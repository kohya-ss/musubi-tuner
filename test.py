from musubi_tuner.wan.modules import t5

#`t2v-A14B`, `i2v-A14B` (for Wan2.2 14B models).


accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 wan_train_network.py
--task i2v-A14B --dit /workspace/ComfyUI/models/diffusion_models/wan2.1_i2v_720p_14B_bf16.safetensors --dataset_config /workspace/musubi-tuner/wan/configs/config.toml --sdpa --mixed_precision bf16 --fp8_base --optimizer_type adamw8bit --learning_rate 3e-5 --gradient_checkpointing --max_data_loader_n_workers 2 --persistent_data_loader_workers --network_module networks.lora_wan --network_dim 128 --network_alpha 1 --timestep_sampling shift --discrete_flow_shift 3.0 --max_train_epochs 1000 --save_every_n_epochs 10 --t5 /workspace/diffusion-pipe/ckpt_path/wan2/models_t5_umt5-xxl-enc-bf16.pth --output_dir /workspace/ComfyUI/models/loras/WAN2_LORAS/14B-musubi-i2v/nolan_musubi-i2v_r128-1_lr3e5 --output_name nolan_i2v_r128-1_1024x1024


python --i2v  C:\workspace\world\musubi-tuner\src\musubi_tuner\wan_cache_latents.py --dataset_config C:\workspace\world\musubi-tuner\dataset_config.toml --vae C:\workspace\world\ComfyUI\models\vae\wan_2.1_vae.safetensors --clip C:\workspace\world\ComfyUI\models\clip\models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
python wan_cache_text_encoder_outputs.py  --dataset_config ./dataset_config.toml  --t5  C:\workspace\world\ComfyUI\models\text_encoders\umt5-xxl-enc-bf16.safetensors --batch_size 16 

/usr/bin/python wan_cache_latents.py  --i2v --dataset_config /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/config/dataset_config.toml --vae /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/models1/vae/wan_2.1_vae.safetensors  --clip /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/models1/clip/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth

/usr/bin/python wan_cache_text_encoder_outputs.py  --dataset_config /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/config/dataset_config.toml   --t5  /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/models1/text_encoders/umt5-xxl-enc-bf16.safetensors --batch_size 16 


#If you train I2V models, add --i2v option to the above command. For Wan2.1, add --clip path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth to specify the CLIP model. If not specified, the training will raise an error. For Wan2.2, CLIP model is not required.

#  'C:\\workspace\\world\\ComfyUI\\models\\diffusion_models\\Wan2.2\\wan2.2_i2v_low_noise_14B_fp16.safetensors'

python wan_train_network.py \
--task i2v-A14B \
  --dit "C:\workspace\world\ComfyUI\models\diffusion_models\Wan2.2\wan2.2_i2v_low_noise_14B_fp16.safetensors" \
  --vae "C:\workspace\world\ComfyUI\models\vae\wan_2.1_vae.safetensors" \
  --t5 "C:\workspace\world\ComfyUI\models\text_encoders\umt5-xxl-enc-bf16.safetensors" \
  --output_dir "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/out2" \
  --output_name "wan2.2-lora-v1" \
  --dataset_config dataset_config.toml \
  --sdpa  --optimizer_type adamw --learning_rate 3e-4 --gradient_checkpointing --gradient_accumulation_steps 1 \
 --max_data_loader_n_workers 1 --network_module networks.lora_wan --network_dim 16 --network_alpha 16 --timestep_sampling shift --discrete_flow_shift 1.0 --max_train_epochs 100 \
 --save_every_n_epochs 10 --seed 5 --optimizer_args weight_decay=0.1 --max_grad_norm 0 --lr_scheduler polynomial --lr_scheduler_power 8  --min_timestep 875 --max_timestep 1000 \
 --mixed_precision fp16 --fp8_base  --network_dim 2 --network_alpha 2 \
  --dit_high_noise "C:\workspace\world\ComfyUI\models\diffusion_models\Wan2.2\wan2.2_i2v_high_noise_fp16.safetensors"  \
# python wan_train_network.py --task i2v-A14B --dit "C:\workspace\world\ComfyUI\models\diffusion_models\Wan2.2\wan2.2_i2v_low_noise_14B_fp16.safetensors" --vae "C:\workspace\world\ComfyUI\models\vae\wan_2.1_vae.safetensors" --dit_high_noise "C:\workspace\world\ComfyUI\models\diffusion_models\Wan2.2\wan2.2_i2v_high_noise_fp16.safetensors" --t5 "C:\workspace\world\ComfyUI\models\text_encoders\umt5-xxl-enc-bf16.safetensors" --dataset_config dataset_config.toml --output_dir "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/out2" --output_name "wan2.2-lora-v1" --sdpa --optimizer_type adamw --learning_rate 3e-4 --gradient_checkpointing --gradient_accumulation_steps 2 --max_data_loader_n_workers 1 --network_module networks.lora_wan --network_dim 8 --network_alpha 8 --timestep_sampling shift --discrete_flow_shift 1.0 --max_train_epochs 20 --save_every_n_epochs 10 --seed 5 --optimizer_args weight_decay=0.1 --max_grad_norm 0 --lr_scheduler polynomial --lr_scheduler_power 8 --min_timestep 875 --max_timestep 1000 --mixed_precision fp16 --fp8_base
#python wan_train_network.py --task i2v-A14B --dit "C:\workspace\world\ComfyUI\models\diffusion_models\Wan2.2\wan2.2_i2v_low_noise_14B_fp16.safetensors" --vae "C:\workspace\world\ComfyUI\models\vae\wan_2.1_vae.safetensors" --dit_high_noise "C:\workspace\world\ComfyUI\models\diffusion_models\Wan2.2\wan2.2_i2v_high_noise_fp16.safetensors" --t5 "C:\workspace\world\ComfyUI\models\text_encoders\umt5-xxl-enc-bf16.safetensors" --dataset_config dataset_config.toml --output_dir "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/out2" --output_name "wan2.2-lora-v1" --sdpa --optimizer_type adamw --learning_rate 3e-4 --gradient_checkpointing --gradient_accumulation_steps 2 --max_data_loader_n_workers 1 --network_module networks.lora_wan --network_dim 4 --network_alpha 4 --timestep_sampling shift --discrete_flow_shift 1.0 --max_train_epochs 20 --save_every_n_epochs 10 --seed 5 --optimizer_args weight_decay=0.1 --max_grad_norm 1 --lr_scheduler polynomial --lr_scheduler_power 8 --min_timestep 875 --max_timestep 1000 --mixed_precision fp16 --batch_size 1
# python wan_train_network.py --task i2v-A14B --dit "C:\workspace\world\ComfyUI\models\diffusion_models\Wan2.2\wan2.2_i2v_low_noise_14B_fp16.safetensors" --vae "C:\workspace\world\ComfyUI\models\vae\wan_2.1_vae.safetensors" --t5 "C:\workspace\world\ComfyUI\models\text_encoders\umt5-xxl-enc-bf16.safetensors" --output_dir "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/out2" --output_name "wan2.2-lora-v1" --xformers --optimizer_type adamw --learning_rate 3e-4 --gradient_checkpointing --gradient_accumulation_steps 1 --max_data_loader_n_workers 2 --network_module networks.lora_wan --network_dim 16 --network_alpha 16 --timestep_sampling shift --discrete_flow_shift 1.0 --max_train_epochs 100 --save_every_n_epochs 10 --seed 5 --optimizer_args weight_decay=0.1 --max_grad_norm 0 --lr_scheduler polynomial --lr_scheduler_power 8 --min_timestep 875 --max_timestep 1000 --mixed_precision fp16 --fp8_base --network_dim 2 --network_alpha 2
#python wan_train_network.py --task i2v-A14B --dit "C:\workspace\world\ComfyUI\models\diffusion_models\Wan2.2\wan2.2_i2v_low_noise_14B_fp16.safetensors" --vae "C:\workspace\world\ComfyUI\models\vae\wan_2.1_vae.safetensors" --t5 "C:\workspace\world\ComfyUI\models\text_encoders\umt5-xxl-enc-bf16.safetensors" --output_dir "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/out2" --output_name "wan2.2-lora-v1" --dataset_config dataset_config.toml --optimizer_type adamw --learning_rate 3e-4 --gradient_checkpointing --gradient_accumulation_steps 2 --max_data_loader_n_workers 1 --network_module networks.lora_wan --network_dim 4 --network_alpha 4 --timestep_sampling shift --discrete_flow_shift 1.0 --max_train_epochs 20 --save_every_n_epochs 10 --seed 5 --optimizer_args weight_decay=0.1 --max_grad_norm 1 --lr_scheduler polynomial --lr_scheduler_power 8 --min_timestep 875 --max_timestep 1000 --mixed_precision fp16 --sdpa


Lower --max_train_epochs (e.g., 20 instead of 100)
Lower --network_dim (e.g., 8 or 4 instead of 16)
Lower --max_data_loader_n_workers (e.g., 1)
Remove --xformers if not needed (it can increase VRAM usage)
Lower batch size if your script supports it (look for --batch_size)
Use --gradient_accumulation_steps > 1 (if supported)
Use --mixed_precision bf16 or --mixed_precision fp16 (already set)
Set environment variable to limit CUDA devices

#Use --fp8_base and/or --fp8_scaled flags for FP8 training.
#wget https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors


/usr/bin/python wan_train_network.py \
--task i2v-A14B \
  --dit "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/models1/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
  --vae "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/models1/diffusion_models/vae/wan_2.1_vae.safetensors" \
   --sdpa --t5 "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/models1/text_encoders/umt5-xxl-enc-bf16.safetensors" \
  --dataset_config /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/dataset_config1.toml \
  --output_dir "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/out2" \
  --output_name "wan2.2-lora-v1" \
  --optimizer_type adamw --learning_rate 3e-4 --gradient_checkpointing --gradient_accumulation_steps 1 \
 --max_data_loader_n_workers 2 --network_module networks.lora_wan --network_dim 16 --network_alpha 16 --timestep_sampling shift --discrete_flow_shift 1.0 --max_train_epochs 100 \
 --save_every_n_epochs 10 --seed 5 --optimizer_args weight_decay=0.1 --max_grad_norm 0 --lr_scheduler polynomial --lr_scheduler_power 8  --min_timestep 875 --max_timestep 1000 \
 --mixed_precision fp16  \
  --dit_high_noise "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/models1/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors" \
   --fp8_base  --timestep_boundary 1


python src/musubi_tuner/wan_generate_video.py --fp8 --task i2v-14B --video_size 832 480 --video_length 81 --infer_steps 20 \
--prompt "prompt for the video" --save_path path/to/save.mp4 --output_type both \
--dit path/to/wan2.1_i2v_480p_14B_bf16_etc.safetensors --vae path/to/wan_2.1_vae.safetensors \
--t5 path/to/models_t5_umt5-xxl-enc-bf16.pth --clip path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
--attn_mode torch --image_path path/to/image.jpg


High-noise training command:

accelerate launch --num_cpu_threads_per_process 1 src/musubi_tuner/wan_train_network.py 
--task t2v-A14B 
--dit /workspace/musubi-tuner/models/diffusion_models/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors 
--vae /workspace/musubi-tuner/models/vae/split_files/vae/wan_2.1_vae.safetensors --t5 /workspace/musubi-tuner/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth 
--dataset_config /workspace/musubi-tuner/dataset/dataset.toml1``
--xformers --mixed_precision fp16 --fp8_base --optimizer_type adamw --learning_rate 3e-4 --gradient_checkpointing --gradient_accumulation_steps 1 \
--max_data_loader_n_workers 2 --network_module networks.lora_wan --network_dim 16 --network_alpha 16 --timestep_sampling shift --discrete_flow_shift 1.0 --max_train_epochs 100 \
--save_every_n_epochs 100 --seed 5 --optimizer_args weight_decay=0.1 --max_grad_norm 0 --lr_scheduler polynomial --lr_scheduler_power 8  --min_timestep 875 --max_timestep 1000 \
--lr_scheduler_min_lr_ratio="5e-5" --output_dir /workspace/musubi-tuner/output 
--output_name WAN2.2-HighNoise_SmartphoneSnapshotPhotoReality_v3_by-AI_Characters \
--metadata_title WAN2.2-HighNoise_SmartphoneSnapshotPhotoReality_v3_by-AI_Characters --metadata_author AI_Characters --preserve_distribution_shape 

Low-noise training command:

accelerate launch --num_cpu_threads_per_process 1 src/musubi_tuner/wan_train_network.py 
--task t2v-A14B 
--dit /workspace/musubi-tuner/models/diffusion_models/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors 
--vae /workspace/musubi-tuner/models/vae/split_files/vae/wan_2.1_vae.safetensors --t5 /workspace/musubi-tuner/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth 
--dataset_config /workspace/musubi-tuner/dataset/dataset.toml --xformers --mixed_precision fp16 --fp8_base --optimizer_type adamw --learning_rate 3e-4 
--gradient_checkpointing --gradient_accumulation_steps 1 --max_data_loader_n_workers 2 --network_module networks.lora_wan --network_dim 16 --network_alpha 16 
--timestep_sampling shift --discrete_flow_shift 1.0 --max_train_epochs 100 --save_every_n_epochs 100 --seed 5 --optimizer_args weight_decay=0.1 --max_grad_norm 0 
--lr_scheduler polynomial --lr_scheduler_power 8 --lr_scheduler_min_lr_ratio="5e-5" --output_dir /workspace/musubi-tuner/output 
--output_name WAN2.2-LowNoise_SmartphoneSnapshotPhotoReality_v3_by-AI_Characters --metadata_title WAN2.2-LowNoise_SmartphoneSnapshotPhotoReality_v3_by-AI_Characters
--metadata_author AI_Characters --preserve_distribution_shape --min_timestep 0 --max_timestep 875


python -m accelerate.commands.launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/wan_train_network.py --task ti2v-A14B --dit "C:\workspace\world\ComfyUI\models\diffusion_models\wan2.1_t2v_14B_bf16.safetensors" --dataset_config dataset_config.toml --sdpa --mixed_precision bf16 --fp8_base --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing --max_data_loader_n_workers 2 --persistent_data_loader_workers --network_module networks.lora_wan --network_dim 32 --timestep_sampling shift --discrete_flow_shift 1.0 --max_train_epochs 15 --save_every_n_steps 200 --seed 7626 --output_dir "C:\workspace\world\ComfyUI\models\loras" --output_name "my-wan-lora-v1" --blocks_to_swap 20 
accelerate launch  --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/wan_train_network.py --task ti2v-A14B  --dit "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/wan2.1_t2v_14B_bf16.safetensors" --dataset_config dataset_config.toml --sdpa --mixed_precision bf16 --fp8_base --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing --max_data_loader_n_workers 2 --persistent_data_loader_workers --network_module networks.lora_wan --network_dim 32 --timestep_sampling shift --discrete_flow_shift 1.0 --max_train_epochs 15 --save_every_n_steps 200 --seed 7626 --output_dir "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/out" --output_name "my-wan-lora-v1" --blocks_to_swap 20 

#--network_weights "C:/ai/sd-models/loras/WAN/experimental/ANYBASELORA.safetensors"
#i think this needed only for resume training

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/wan_train_network.py `
  --task ti2v-A14B `
  --dit "C:/ai/sd-models/checkpoints/Wan/wan2.1_t2v_14B_bf16.safetensors" `
  --dataset_config dataset_config.toml `
  --sdpa --mixed_precision bf16 --fp8_base `
  --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing `
  --max_data_loader_n_workers 2 --persistent_data_loader_workers `
  --network_module networks.lora_wan --network_dim 64 --network_alpha 4 `
  --timestep_sampling shift --discrete_flow_shift 1.0 `
  --max_train_epochs 5 --save_every_n_steps 200 --seed 7626 `
  --output_dir "C:/ai/sd-models/loras/WAN/experimental" `
  --output_name "my-wan-lora-v2" --blocks_to_swap 25 `
  --network_weights "C:/ai/sd-models/loras/WAN/experimental/my-wan-lora-v1.safetensors"



python -m accelerate.commands.launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/wan_train_network.py --task ti2v-A14B 
--dit
"C:\workspace\world\ComfyUI\models\diffusion_models\wan2.1_t2v_14B_bf16.safetensors" 
--dataset_config dataset_config.toml --sdpa --mixed_precision bf16 --fp8_base --optimizer_type adamw8bit --learning_rate 2e-4 
--gradient_checkpointing --max_data_loader_n_workers 2 --persistent_data_loader_workers --network_module networks.lora_wan 
--network_dim 32 --timestep_sampling shift --discrete_flow_shift 1.0 --max_train_epochs 15 --save_every_n_steps 200 --seed 7626 
--output_dir 
"C:\workspace\world\ComfyUI\models\loras" --output_name "my-wan-lora-v1" --blocks_to_swap 20 

#accelerate launch  --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/wan_train_network.py --task ti2v-A14B  
# --dit "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/wan2.1_t2v_14B_bf16.safetensors" --dataset_config dataset_config.toml --sdpa --mixed_precision bf16 --fp8_base --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing --max_data_loader_n_workers 2 --persistent_data_loader_workers --network_module networks.lora_wan --network_dim 32 --timestep_sampling shift --discrete_flow_shift 1.0 --max_train_epochs 15 --save_every_n_steps 200 --seed 7626 --output_dir "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/out" --output_name "my-wan-lora-v1" --blocks_to_swap 20 
# srun -A ai4gaming_isekai --partition interactive --time=04:00:00 --gpus 1 --container-image=/lustre/fsw/portfolios/ai4gaming/users/kschmid/synthetic_data/sdg_musubi.sqsh --container-mounts=/lustre:/lustre --container-save /lustre/fsw/portfolios/ai4gaming/users/kschmid/synthetic_data/sdg_musubi.sqsh  --pty /bin/bash
cd musubi-tuner/
/usr/bin/python src/musubi_tuner/wan_train_network.py \
  --task ti2v-A14B \
  --dit "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/wan2.1_t2v_14B_bf16.safetensors" \
  --dataset_config dataset_config.toml \
  --sdpa   --network_dim 32 \ --mixed_precision bf16 --fp8_base \
  --optimizer_type adamw8bit --learning_rate 2e-4 \
  --gradient_checkpointing --max_data_loader_n_workers 2 \
  --persistent_data_loader_workers --network_module networks.lora_wan \
  --timestep_sampling shift \
  --discrete_flow_shift 1.0 \
  --max_train_epochs 20 \
  --save_every_n_steps 20 --seed 7626 \
  --output_dir "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/out" \
  --output_name "my-wan-lora-v1" \

--blocks_to_swap 20 


The task for Wan2.2 I2V is i2v-A14B
/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/models1/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/models1/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
kschmid@cs-oci-ord-login-01:~$ ls /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/models1/text_encoders/umt5-xxl-enc-bf16.safetensors  umt5_xxl_fp8_e4m3fn_scaled.safetensors

- Value not in list: unet_name: 'wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors' not in ['RMBG-2.0.safetensors', 'Wan2.2\\wan2.2_t2v_low_noise_14B_fp16.safetensors', 'Wan2_1-Ti2v-A14B_fp8_e4m3fn.safetensors', 'Wan2_1-T2V-1_3B_fp8_e4m3fn.safetensors', 'Wan2_1-VACE_module_14B_fp8_e4m3fn.safetensors', 'Wan2_2-I2V-A14B-HIGH_fp8_e5m2_scaled_KJ.safetensors', 'Wan2_2-I2V-A14B-LOW_fp8_e5m2_scaled_KJ.safetensors', 'flux1-dev-fp8.safetensors', 'wan2.1_t2v_14B_bf16.safetensors']
/usr/bin/python wan_train_network.py \
  --task i2v-A14B \
  --dit "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/models1/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" \
  --dit_high_noise "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/models1/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" \
  --vae "/lustre/fsw/portfolios/ai4gaming/users/kschmid/models1/vae/wan_2.1_vae.safetensors" \
  --t5 "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/models1/text_encoders/umt5-xxl-enc-bf16.safetensors" \
  --dataset_config dataset_config.toml \
  --output_dir "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/out2" \
  --output_name "wan2.2-lora-v1" \
  --sdpa \
  --network_dim 32 \
  --optimizer_type adamw8bit \
  --learning_rate 2e-4 \
  --gradient_checkpointing \
  --max_data_loader_n_workers 2 \
  --persistent_data_loader_workers \
  --network_module networks.lora_wan \
  --timestep_sampling shift \
  --discrete_flow_shift 1.0 \
  --max_train_epochs 20 \
  --save_every_n_steps 20 \
  --seed 7626 \
  --timestep_boundary 1 \
  --blocks_to_swap 20 \
  --log_with tensorboard \
  --logging_dir "/lustre/fsw/portfolios/ai4gaming/users/kschmid/tb" \
  --fp8_base
 # --mixed_precision fp16 \
  

/usr/bin/python wan_train_network.py \
--max_train_epochs 50 --save_every_n_epochs 1 --seed 42 \
--output_dir "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/out2" \
--output_name "wan2.2-lora-v1" \
--network_module networks.lora_wan \
--task i2v-A14B   --sdpa  \
--dataset_config dataset_config.toml \
--dit "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/models1/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" \
--dit_high_noise "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/models1/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" \
--vae "/lustre/fsw/portfolios/ai4gaming/users/kschmid/models1/vae/wan_2.1_vae.safetensors" \
--t5 "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/models1/text_encoders/umt5-xxl-enc-bf16.safetensors" \
--discrete_flow_shift 5.0 --timestep_sampling shift --timestep_boundary 900 \
--fp8_base  --mixed_precision fp16  \
--max_data_loader_n_workers 16 --persistent_data_loader_workers  \
--network_module networks.lora_wan --network_dim 16 --network_alpha 16 --network_args loraplus_lr_ratio=4  \
--optimizer_type schedulefree.RAdamScheduleFree --learning_rate 0.002 \
--gradient_checkpointing \
--rope_func comfy
--fp8_scaled


/usr/bin/python wan_train_network.py \
--dit "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/models1/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" \
--dataset_config dataset_config.toml \
--flash_attn --mixed_precision fp16 --optimizer_type came_pytorch.CAME --lr_scheduler constant_with_warmup \
--optimizer_args weight_decay=0.01 eps=\(1e-30,1e-16\) betas=\(0.9,0.999,0.9999\) --learning_rate 2e-5 \
--gradient_checkpointing --max_data_loader_n_workers 16 --persistent_data_loader_workers \
--network_module=networks.lora_wan --network_dim=16 --network_alpha=16 --timestep_sampling shift \
--discrete_flow_shift 3.0 --gradient_accumulation_steps 1 --max_train_steps 2400 --save_every_n_epochs=1 \
--seed 032424 --output_dir ./WanKisses --output_name WanKisses --blocks_to_swap 28 --log_with tensorboard \
--logging_dir ./logs --save_state_on_train_end --fp8_base --lr_warmup_steps 100 --network_args loraplus_lr_ratio=4 \
--network_dropout 0.05  --mixed_precision_transformer \
--preserve_distribution_shape --min_timestep 875 --max_timestep 1000 --rope_func comfy \
--optimized_compile --fp8_scaled
#--upcast_quantization --upcast_linear 

accelerate launch --num_cpu_threads_per_process 1 src\musubi_tuner\wan_train_network.py
--dataset_config "E:\AI\musubi-tuner\train\2.toml"
--vae "E:\AI\musubi-tuner\wan\wan_2.1_vae.safetensors"
--flash_attn --split_attn
--fp8_base --fp8_scaled
--gradient_checkpointing
--network_module networks.lora_wan --network_dim 16 --network_alpha 16
--min_timestep 900 --max_timestep 1000
--max_train_epochs 20 --save_every_n_epochs 2 --seed 42
--log_with tensorboard


  accelerate launch --num_cpu_threads_per_process 1 --mixed_precision fp16 
src/musubi_tuner/wan_train_network.py
--max_train_epochs 50 --save_every_n_epochs 1 --seed 42
--output_dir /workspace/output --output_name motionv1
--network_module networks.lora_wan
--task i2v-A14B
--dataset_config /workspace/dataset.toml
--dit /workspace/models/wan2.2_i2v_low_noise_14B_fp16.safetensors
--dit_high_noise /workspace/models/wan2.2_i2v_high_noise_14B_fp16.safetensors
--t5 /workspace/models/umt5-xxl-enc-bf16.safetensors
--vae /workspace/models/wan_2.1_vae.safetensors
--discrete_flow_shift 5.0 --timestep_sampling shift --timestep_boundary 900
--gradient_checkpointing --rope_func comfy --sdpa 
--fp8_base --fp8_scaled --mixed_precision fp16
--max_data_loader_n_workers 16 --persistent_data_loader_workers
--network_module networks.lora_wan --network_dim 16 --network_alpha 16 --network_args loraplus_lr_ratio=4
--optimizer_type schedulefree.RAdamScheduleFree --learning_rate 0.002

python wan_train_network.py ^
--task ii2v-A14B ^
--dit "H:\ComfyStudio\App\ComfyUI\models\diffusion_models\wan2.2_i2v_high_noise_14B_fp16.safetensors" ^
--vae "H:\ComfyStudio\App\ComfyUI\models\vae\wan_2.1_vae.safetensors" ^
--t5 "H:\ComfyStudio\App\ComfyUI\models\clip\models_t5_umt5-xxl-enc-bf16.pth" ^
--dataset_config "dataset\PicData\dataset.toml" ^
--sdpa ^
--mixed_precision fp16 ^
--fp8_base ^
--optimizer_type adamw8bit ^
--learning_rate 1e-4 ^
--gradient_checkpointing ^
--max_data_loader_n_workers 2 ^
--network_module networks.lora_wan ^
--network_dim 32 ^
--network_alpha 1 ^
--timestep_sampling shift ^
--discrete_flow_shift 7.0 ^
--min_timestep 875 ^
--max_timestep 1000 ^
--max_train_epochs 60 ^
--logging_dir logs ^
--log_with tensorboard

/lustre/fsw/portfolios/ai4gaming/users/kschmid/models1/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
/lustre/fsw/portfolios/ai4gaming/users/kschmid/models1/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
/lustre/fsw/portfolios/ai4gaming/users/kschmid/models1/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors

wan2.2_t2v_low_noise_14B_fp16.safetensors


Wan2.2-T2V-A14B-4steps-250928-dyno-NativeComfy.json, 
Wan2_2_native.json

Wan2_1_VAE_bf16.safetensors  Wan2_2_VAE_bf16.safetensors  put_vae_here  qwen_image_vae.safetensors  wan_2.1_vae.safetensors
umt5-xxl-enc-bf16.safetensors  umt5_xxl_fp8_e4m3fn_scaled




Wan2.2-T2V-A14B-4steps-250928-dyno-high-lightx2v.safetensors  Wan2_2-T2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors
Wan2_1-Ii2v-A14B-720P_fp8_e5m2.safetensors                      Wan2_2-T2V-A14B-LOW_fp8_e5m2_scaled_KJ.safetensors
Wan2_1-Ti2v-A14B_fp8_e4m3fn.safetensors                         Wan2_2-T2V-A14B-LOW_fp8_e5m2_scaled_KJ.safetensors.1
Wan2_1-VACE_module_14B_fp8_e4m3fn.safetensors                 Wan2_2-T2V-A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors
Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors         put_diffusion_model_files_here
Wan2_2-I2V-A14B-HIGH_fp8_e5m2_scaled_KJ.safetensors           qwen_image_edit_2509_fp8_e4m3fn.safetensors
Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors          qwen_image_fp8_e4m3fn.safetensors
Wan2_2-I2V-A14B-LOW_fp8_e5m2_scaled_KJ.safetensors            wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
Wan2_2-T2V-A14B-HIGH_fp8_e5m2_scaled_KJ.safetensors           wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
Wan2_2-T2V-A14B-HIGH_fp8_e5m2_scaled_KJ.safetensors.1         wan2.2_t2v_low_noise_14B_fp16.safetensors

/usr/bin/python src/musubi_tuner/wan_train_network.py --task ti2v-A14B --dit "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/wan2.1_t2v_14B_bf16.safetensors" --dataset_config dataset_config.toml --sdpa --mixed_precision bf16 --fp8_base --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing --max_data_loader_n_workers 2 --persistent_data_loader_workers --network_module networks.lora_wan --network_dim 32 --timestep_sampling shift --discrete_flow_shift 1.0 --max_train_epochs 20 --save_every_n_steps 10 --seed 7626 --output_dir "/lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/out" --output_name "my-wan-lora-v1" --blocks_to_swap 20

ti2v-A14B (text-to-video, 14B model)
t2v-1.3B (text-to-video, 1.3B model)
t2v-A14B (text-to-video, advanced 14B model)

https://github.com/kohya-ss/musubi-tuner/issues/560

# srun -A ai4gaming_isekai --partition interactive --time=04:00:00 --gpus 1 --container-image=/lustre/fsw/portfolios/ai4gaming/users/kschmid/synthetic_data/sdg_musubi.sqsh --container-mounts=/lustre:/lustre --container-save /lustre/fsw/portfolios/ai4gaming/users/kschmid/synthetic_data/sdg_musubi.sqsh  --pty /bin/bash
cd musubi-tuner/
/usr/bin/python src/musubi_tuner/wan_generate_video.py \
--fp8 --task ti2v-A14B --video_size 720 1280 --video_length 81 --infer_steps 20 \
--from_file /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/synthetic_data/sdg_game_descriptions_pnc_puzzle_snow_cartoony_character_short.txt \
--save_path /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/video/run2.0 \
--output_type video \
--dit /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/wan2.1_t2v_14B_bf16.safetensors \
--vae /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/Wan2_1_VAE_bf16.safetensors \
--t5 /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/models_t5_umt5-xxl-enc-bf16.pth \
--lora_weight /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/out/my-wan-lora-v1-step00000060.safetensors \
--lora_multiplier 1.0 \
--lora_weight_high_noise /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/out/my-wan-lora-v1-step00000060.safetensors 
#--output_type both \
wanvideo2_2_I2V_A14B.json,
wan2.1_t2v_14B_bf16.safetensors
Wan2.2-T2V-A14B-4steps-250928-dyno-NativeComfy.json, Wan2_2_native.json,

python wan_generate_video.py \
    --task ii2v-A14B \
    --video_size 1280 720 \
    --video_length 21 \
    --infer_steps 20 \
    --prompt "动态实景拍摄，开阔草地上的自然风光，前景聚焦一棵枝叶繁茂的大树，其绿叶在微风中轻轻摇曳。背景延伸至一片树林，树木与灌木错落有致，构成层次丰富的绿色画卷。天空布满厚重阴云，营造出一种宁静而略带压抑的氛围。远处，隐约可见行人穿梭其间，增添了几分生活气息。镜头采用流畅的右转移动，捕捉每一处细节，强化了场景的动感与立体感。全景，自然风光记录，阴天下的户外生态景象。" \
    --save_path ./output/save \
    --output_type video \
    --dit ckpts/wan/wan2.1_i2v_720p_14B_bf16.safetensors \
    --vae ckpts/wan/wan_2.1_vae.safetensors \
    --t5 ckpts/wan/models_t5_umt5-xxl-enc-bf16.pth \
    --clip ckpts/wan/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    --attn_mode torch \
    --image_path test/test.jpg \
    --lora_weight output/720_hq/1/wan-lora-720-step00000850.safetensors \
    --lora_multiplier 1.7 \
    --seed 42 \
    --fp8

/usr/bin/python src/musubi_tuner/wan_generate_video.py \
--fp8 \
--task ti2v-A14B \
--video_size 1280 720 \
--video_length 81 \
--infer_steps 20 \
--from_file /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/synthetic_data/sdg_game_descriptions_pnc_puzzle_snow_cartoony_character.txt \
--save_path /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/video/out_lora.mp4 \
--output_type both \
--dit /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/wan2.1_t2v_14B_bf16.safetensors \
--vae /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/Wan2.1_VAE.pth \
--t5 /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/models_t5_umt5-xxl-enc-bf16.pth \
--lora_weight /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/out/my-wan-lora-v1-step00000060.safetensors \
--lora_multiplier 1.0 \
--lora_weight_high_noise /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/out/my-wan-lora-v1-step00000060.safetensors

--prompt "Third-person view snowboarding gameplay. The camera initially faced North, very wide angle shot show the rider from behind not looking at the camera. The rider wearing a neon yellow helmet, transparent goggles, mid-length red hair, white skiing pants, and a dark jacket with hexagon patterns, no poles just gloves, The rider is focused on boarding.     camera dramatically capturing the moment you beat your rival while laughing maniacally in slow motion as the announcer yells, \"IT'S TRICKYYYY!\" with the words materializing in giant 3D letters made of snow." \

models_t5_umt5-xxl-enc-bf16.pth

/usr/bin/python src/musubi_tuner/wan_generate_video.py
--task t2v-A14B
--video_size 1280 720
--vae_cache_cpu
--offload_inactive_dit
--video_length 81
--infer_steps 20
--prompt "Third-person view snowboarding gameplay. The camera initially faced North, very wide angle shot show the rider from behind not looking at the camera. The rider wearing a neon yellow helmet, transparent goggles, mid-length red hair, white skiing pants, and a dark jacket with hexagon patterns, no poles just gloves, The rider is focused on boarding.     camera dramatically capturing the moment you beat your rival while laughing maniacally in slow motion as the announcer yells, \"IT'S TRICKYYYY\!\" with the words materializing in giant 3D letters made of snow."
--output_type video
--dit /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/wan2.1_t2v_14B_bf16.safetensors
--vae /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/Wan2.1_VAE.pth
--t5 /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/--t5 /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/models_t5_umt5-xxl-enc-bf16.pth \
--lora_weight /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/out/my-wan-lora-v1-step00000060.safetensors
--lora_multiplier 1.0
--lora_weight_high_noise /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/out/my-wan-lora-v1-step00000060.safetensors
--lora_multiplier_high_noise 1.0
--save_path /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/video/out_lora.mp4
--seed 42
#--attn_mode flash2
#--dit_high_noise /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/wan2.2_t2v_high_noise_14B_fp16.safetensors
--fp8
--fp8_scaled
--fp8_t5

/usr/bin/python src/musubi_tuner/wan_generate_video.py --task t2v-A14B --video_size 1280 720  --vae_cache_cpu --offload_inactive_dit --video_length 81 --infer_steps 20 --prompt 'Third-person view snowboarding gameplay. The camera initially faced North, very wide angle shot show the rider from behind not looking at the camera. The rider wearing a neon yellow helmet, transparent goggles, mid-length red hair, white skiing pants, and a dark jacket with hexagon patterns, no poles just gloves, The rider is focused on boarding.     camera dramatically capturing the moment you beat your rival while laughing maniacally in slow motion as the announcer yells, "IT'\''S TRICKYYYY!" with the words materializing in giant 3D letters made of snow.' --output_type video --dit /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/wan2.1_t2v_14B_bf16.safetensors --vae /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/Wan2.1_VAE.pth --t5 /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/models_t5_umt5-xxl-enc-bf16.pth --lora_weight /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/out/my-wan-lora-v1-step00000060.safetensors --lora_multiplier 1.0 --lora_weight_high_noise /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/out/my-wan-lora-v1-step00000060.safetensors --lora_multiplier_high_noise 1.0 --save_path /lustre/fs11/portfolios/ai4gaming/projects/ai4gaming_isekai/users/kschmid/LORA/video/out_lora.mp4 --seed 42

python src/musubi_tuner/wan_generate_video.py --fp8 --task t2v-1.3B --video_size  832 480 --video_length 81 --infer_steps 20 \
--prompt "prompt for the video" --save_path path/to/save.mp4 --output_type both --dit path/to/wan2.1_t2v_1.3B_bf16_etc.safetensors --vae path/to/wan_2.1_vae.safetensors -t5 path/to/models_t5_umt5-xxl-enc-bf16.pth --attn_mode torch

--include_patterns and --exclude_patterns can be used to specify which LoRA modules to apply or exclude during training. If not specified, all modules are applied by default. These options accept regular expressions.

--include_patterns specifies the modules to be applied, and --exclude_patterns specifies the modules to be excluded. The regular expression is matched against the LoRA key name, and include takes precedence.

The key name to be searched is in sd-scripts format (lora_unet_<module_name with dot replaced by _>). For example, lora_unet_blocks_9_cross_attn_k.

For example, if you specify --exclude_patterns "blocks_[23]\d_" , it will exclude modules containing blocks_20 to blocks_39. If you specify --include_patterns "cross_attn" --exclude_patterns "blocks_(0|1|2|3|4)_", it will apply LoRA to modules containing cross_attn and not containing blocks_0 to blocks_4.

If you specify multiple LoRA weights, please specify them with multiple arguments. For example: --include_patterns "cross_attn" ".*" --exclude_patterns "dummy_do_not_exclude" "blocks_(0|1|2|3|4)". ".*" is a regex that matches everything. dummy_do_not_exclude is a dummy regex that does not match anything.