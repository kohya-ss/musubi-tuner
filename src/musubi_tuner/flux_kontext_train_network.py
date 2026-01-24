import argparse
from typing import Optional


from einops import rearrange
import torch
from tqdm import tqdm
from accelerate import Accelerator

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_FLUX_KONTEXT, ARCHITECTURE_FLUX_KONTEXT_FULL
from musubi_tuner.flux import flux_models, flux_utils
from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    load_prompts,
    clean_memory_on_device,
    setup_parser_common,
    read_config_from_file,
)

import logging

from musubi_tuner.utils import model_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FluxKontextNetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_FLUX_KONTEXT

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_FLUX_KONTEXT_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
        self._i2v_training = False
        self._control_training = False  # this means video training, not control image training
        self.default_guidance_scale = 2.5  # embeded guidance scale for inference

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        device = accelerator.device

        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        # Load T5 and CLIP text encoders
        t5_dtype = torch.float8e4m3fn if args.fp8_t5 else torch.bfloat16
        tokenizer1, text_encoder1 = flux_utils.load_t5xxl(args.text_encoder1, dtype=t5_dtype, device=device, disable_mmap=True)
        tokenizer2, text_encoder2 = flux_utils.load_clip_l(
            args.text_encoder2, dtype=torch.bfloat16, device=device, disable_mmap=True
        )

        # Encode with T5 and CLIP text encoders
        logger.info("Encoding with T5 and CLIP text encoders")

        sample_prompts_te_outputs = {}  # (prompt) -> (t5, clip)
        with torch.amp.autocast(device_type=device.type, dtype=t5_dtype), torch.no_grad():
            for prompt_dict in prompts:
                prompt = prompt_dict.get("prompt", "")
                if prompt is None or prompt in sample_prompts_te_outputs:
                    continue

                # encode prompt
                logger.info(f"cache Text Encoder outputs for prompt: {prompt}")

                t5_tokens = tokenizer1(
                    prompt,
                    max_length=flux_models.T5XXL_MAX_LENGTH,
                    padding="max_length",
                    return_length=False,
                    return_overflowing_tokens=False,
                    truncation=True,
                    return_tensors="pt",
                )["input_ids"]
                l_tokens = tokenizer2(prompt, max_length=77, padding="max_length", truncation=True, return_tensors="pt")[
                    "input_ids"
                ]

                with torch.autocast(device_type=device.type, dtype=text_encoder1.dtype), torch.no_grad():
                    t5_vec = text_encoder1(
                        input_ids=t5_tokens.to(text_encoder1.device), attention_mask=None, output_hidden_states=False
                    )["last_hidden_state"]
                    assert torch.isnan(t5_vec).any() == False, "T5 vector contains NaN values"
                    t5_vec = t5_vec.cpu()

                with torch.autocast(device_type=device.type, dtype=text_encoder2.dtype), torch.no_grad():
                    clip_l_pooler = text_encoder2(l_tokens.to(text_encoder2.device))["pooler_output"]
                    clip_l_pooler = clip_l_pooler.cpu()

                # save prompt cache
                sample_prompts_te_outputs[prompt] = (t5_vec, clip_l_pooler)

        del tokenizer1, text_encoder1, tokenizer2, text_encoder2
        clean_memory_on_device(device)

        # prepare sample parameters
        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()

            prompt = prompt_dict.get("prompt", "")
            prompt_dict_copy["t5_vec"] = sample_prompts_te_outputs[prompt][0]
            prompt_dict_copy["clip_l_pooler"] = sample_prompts_te_outputs[prompt][1]

            sample_parameters.append(prompt_dict_copy)

        clean_memory_on_device(accelerator.device)

        return sample_parameters

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
        control_video_path=None,
    ):
        """architecture dependent inference"""
        model: flux_models.Flux = transformer
        device = accelerator.device

        # Get embeddings
        t5_vec = sample_parameter["t5_vec"].to(device=device, dtype=torch.bfloat16)
        clip_l_pooler = sample_parameter["clip_l_pooler"].to(device=device, dtype=torch.bfloat16)

        txt_ids = torch.zeros(t5_vec.shape[0], t5_vec.shape[1], 3, device=t5_vec.device)

        # Initialize latents
        packed_latent_height, packed_latent_width = height // 16, width // 16
        noise_dtype = torch.float32
        noise = torch.randn(
            1,
            packed_latent_height * packed_latent_width,
            16 * 2 * 2,
            generator=generator,
            dtype=noise_dtype,
            device=device,
        ).to(device, dtype=torch.bfloat16)

        img_ids = flux_utils.prepare_img_ids(1, packed_latent_height, packed_latent_width).to(device)

        vae.to(device)
        vae.eval()

        # prepare control latent
        control_latent = None
        control_latent_ids = None
        if "control_image_path" in sample_parameter:
            control_image_path = sample_parameter["control_image_path"][0]  # only use the first control image
            control_image_tensor, _, _ = flux_utils.preprocess_control_image(control_image_path, resize_to_prefered=False)

            with torch.no_grad():
                control_latent = vae.encode(control_image_tensor.to(device, dtype=vae.dtype))

            # pack control_latent
            ctrl_packed_height = control_latent.shape[2] // 2
            ctrl_packed_width = control_latent.shape[3] // 2
            control_latent = rearrange(control_latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            control_latent_ids = flux_utils.prepare_img_ids(1, ctrl_packed_height, ctrl_packed_width, is_ctrl=True).to(device)

            control_latent = control_latent.to(torch.bfloat16)

        vae.to("cpu")
        clean_memory_on_device(device)

        # denoise
        discrete_flow_shift = discrete_flow_shift if discrete_flow_shift != 0 else None  # None means no shift
        timesteps = flux_utils.get_schedule(
            num_steps=sample_steps, image_seq_len=packed_latent_height * packed_latent_width, shift_value=discrete_flow_shift
        )

        x = noise
        del noise
        guidance_vec = torch.full((x.shape[0],), guidance_scale, device=x.device, dtype=x.dtype)

        for t_curr, t_prev in zip(tqdm(timesteps[:-1]), timesteps[1:]):
            t_vec = torch.full((x.shape[0],), t_curr, dtype=x.dtype, device=x.device)

            img_input = x
            img_input_ids = img_ids
            if control_latent is not None:
                # if control_latent is provided, concatenate it to the input
                img_input = torch.cat((img_input, control_latent), dim=1)
                img_input_ids = torch.cat((img_input_ids, control_latent_ids), dim=1)

            with torch.no_grad():
                pred = model(
                    img=img_input,
                    img_ids=img_input_ids,
                    txt=t5_vec,
                    txt_ids=txt_ids,
                    y=clip_l_pooler,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
            pred = pred[:, : x.shape[1]]

            x = x + (t_prev - t_curr) * pred

        # unpack
        x = x.float()
        x = rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=packed_latent_height, w=packed_latent_width, ph=2, pw=2)
        latent = x.to(vae.dtype)
        del x

        # Move VAE to the appropriate device for sampling
        vae.to(device)
        vae.eval()

        # Decode latents to video
        logger.info(f"Decoding video from latents: {latent.shape}")
        with torch.no_grad():
            pixels = vae.decode(latent)  # decode to pixels
        del latent

        logger.info("Decoding complete")
        pixels = pixels.to(torch.float32).cpu()
        pixels = (pixels / 2 + 0.5).clamp(0, 1)  # -1 to 1 -> 0 to 1

        vae.to("cpu")
        clean_memory_on_device(device)

        pixels = pixels.unsqueeze(2)  # add a dummy dimension for video frames, B C H W -> B C 1 H W
        return pixels

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        vae_path = args.vae

        logger.info(f"Loading AE model from {vae_path}")
        ae = flux_utils.load_ae(vae_path, dtype=torch.float32, device="cpu", disable_mmap=True)
        return ae

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        model = flux_utils.load_flow_model(
            ckpt_path=args.dit,
            dtype=None,
            device=loading_device,
            disable_mmap=True,
            attn_mode=attn_mode,
            split_attn=split_attn,
            loading_device=loading_device,
            fp8_scaled=args.fp8_scaled,
        )
        return model

    def compile_transformer(self, args, transformer):
        transformer: flux_models.Flux = transformer
        return model_utils.compile_transformer(
            args, transformer, [transformer.double_blocks, transformer.single_blocks], disable_linear=self.blocks_to_swap > 0
        )

    def scale_shift_latents(self, latents):
        return latents

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ):
        model: flux_models.Flux = transformer

        bsize = latents.shape[0]
        latents = batch["latents"]  # B, C, H, W
        control_latents = batch["latents_control"]  # B, C, H, W

        # pack latents
        packed_latent_height = latents.shape[2] // 2
        packed_latent_width = latents.shape[3] // 2
        noisy_model_input = rearrange(noisy_model_input, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        img_ids = flux_utils.prepare_img_ids(bsize, packed_latent_height, packed_latent_width)

        # pack control latents
        packed_control_latent_height = control_latents.shape[2] // 2
        packed_control_latent_width = control_latents.shape[3] // 2
        control_latents = rearrange(control_latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        control_latent_lengths = [control_latents.shape[1]] * bsize

        control_ids = flux_utils.prepare_img_ids(bsize, packed_control_latent_height, packed_control_latent_width, is_ctrl=True)

        # context
        t5_vec = batch["t5_vec"]  # B, T, D
        clip_l_pooler = batch["clip_l_pooler"]  # B, T, D
        txt_ids = torch.zeros(t5_vec.shape[0], t5_vec.shape[1], 3, device=accelerator.device)

        # ensure the hidden state will require grad
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            control_latents.requires_grad_(True)
            t5_vec.requires_grad_(True)
            clip_l_pooler.requires_grad_(True)

        # call DiT
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
        img_ids = img_ids.to(device=accelerator.device)
        control_latents = control_latents.to(device=accelerator.device, dtype=network_dtype)
        control_ids = control_ids.to(device=accelerator.device)
        t5_vec = t5_vec.to(device=accelerator.device, dtype=network_dtype)
        clip_l_pooler = clip_l_pooler.to(device=accelerator.device, dtype=network_dtype)

        # use 1.0 as guidance scale for FLUX.1 Kontext training
        guidance_vec = torch.full((bsize,), 1.0, device=accelerator.device, dtype=network_dtype)

        img_input = torch.cat((noisy_model_input, control_latents), dim=1)
        img_input_ids = torch.cat((img_ids, control_ids), dim=1)

        timesteps = timesteps / 1000.0
        model_pred = model(
            img=img_input,
            img_ids=img_input_ids,
            txt=t5_vec,
            txt_ids=txt_ids,
            y=clip_l_pooler,
            timesteps=timesteps,
            guidance=guidance_vec,
            control_lengths=control_latent_lengths,
        )
        model_pred = model_pred[:, : noisy_model_input.shape[1]]  # remove control latents

        # unpack latents
        model_pred = rearrange(
            model_pred, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=packed_latent_height, w=packed_latent_width, ph=2, pw=2
        )

        # flow matching loss
        target = noise - latents

        return model_pred, target

    # endregion model specific


def flux_kontext_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Flux-Kontext specific parser setup"""
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument("--text_encoder1", type=str, default=None, help="text encoder (T5) checkpoint path")
    parser.add_argument("--fp8_t5", action="store_true", help="use fp8 for Text Encoder model")
    parser.add_argument(
        "--text_encoder2",
        type=str,
        default=None,
        help="text encoder (CLIP) checkpoint path, optional. If training I2V model, this is required",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = flux_kontext_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = None  # set from mixed_precision
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"  # make bfloat16 as default for VAE

    trainer = FluxKontextNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
