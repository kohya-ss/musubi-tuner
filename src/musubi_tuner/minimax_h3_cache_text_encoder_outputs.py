from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import logging
from pathlib import Path

import torch

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_minimax_h3
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.text_encoder import (
    H3Presentation,
    H3TextVisual,
    TEACHER_CONDITIONS_REF,
    TEACHER_CONDITIONS_SUBJECT_REF,
    TEXT_CACHE_FORMAT,
    build_presentation,
    encode_h3_presentation,
    load_h3_processor,
    load_h3_text_encoder,
    normalize_teacher_conditions,
    presentation_fingerprint,
    processor_fingerprint,
    save_h3_uncond_cache,
    wrap_ref_teacher_caption,
    wrap_subject_reference_caption,
)
from musubi_tuner.minimax_h3.packing import one_frame_condition_role
from musubi_tuner.minimax_h3.args import add_h3_text_encoder_args
from musubi_tuner.minimax_h3.media import (
    H3_TASKS,
    ONE_FRAME_REFERENCE_FRAME_CAP,
    TEXT_VISUAL_FPS,
    TEXT_VISUAL_FRAME_STRIDE,
    H3AudioSource,
    H3MediaDecoder,
    H3Record,
    H3Reference,
    PyAVH3MediaDecoder,
    adapt_reference_canvas,
    fingerprint_file,
    reject_one_frame_audio_references,
    resize_frames,
    validate_subject_reference_record,
)
from musubi_tuner.minimax_h3.cache_plan import (
    H3DatasetPlan,
    cache_metadata_matches,
    item_control_paths,
    item_crop_start,
    item_plan,
    item_record,
    plan_h3_datasets,
)
from musubi_tuner.minimax_h3.checkpoint import fingerprint_checkpoint


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _text_media_paths(record, task: str, teacher_conditions: str | None = None) -> set[Path]:
    if teacher_conditions == TEACHER_CONDITIONS_SUBJECT_REF:
        # the subject-reference teacher presentation embeds the item's reference pictures
        return {reference.path for reference in record.references if reference.type in {"image", "video"}}
    if task == "fl2va" or teacher_conditions:
        # the FL2VA (or teacher) presentation embeds the first/last frames of the target video
        return {record.video_path}
    if task == "ref2va":
        return {reference.path for reference in record.references if reference.type in {"image", "video"}}
    return set()


def _build_visuals(
    record,
    task: str,
    item: ItemInfo,
    decoder: H3MediaDecoder,
    decoded_reference_cache: dict[tuple, torch.Tensor],
) -> dict[object, H3TextVisual]:
    if task == "t2va":
        return {}
    target_frames = torch.as_tensor(item.content)
    if target_frames.ndim != 4:
        raise ValueError(f"MiniMax-H3 target frames must be [F,H,W,C], got {tuple(target_frames.shape)}")
    if task == "fl2va":
        return {
            "first": H3TextVisual(target_frames[:1]),
            "last": H3TextVisual(target_frames[-1:]),
        }

    target_size = (int(target_frames.shape[2]), int(target_frames.shape[1]))
    return _reference_visuals(record, item.frame_count, target_size, decoder, decoded_reference_cache)


def _reference_visuals(
    record,
    reference_frame_cap: int,
    target_size: tuple[int, int],
    decoder: H3MediaDecoder,
    decoded_reference_cache: dict[tuple, torch.Tensor],
) -> dict[object, H3TextVisual]:
    """Text visuals of the record's references: image references as single frames, video
    references sampled at 2 fps, decoded through the same reference canvas policy as the
    latent cache (target_size caps images, reference_frame_cap caps videos)."""
    visuals = {}
    for reference in record.references:
        if reference.type not in {"image", "video"}:
            continue
        cache_key = (reference.path, reference.type, reference_frame_cap, target_size)
        frames = decoded_reference_cache.get(cache_key)
        if frames is None:
            frames = decoder.decode_reference_visual(
                reference,
                target_frame_count=reference_frame_cap,
                target_size=target_size,
            )
            decoded_reference_cache[cache_key] = frames
        if reference.type == "image":
            visuals[reference.path] = H3TextVisual(frames)
        else:
            sampled = frames[::TEXT_VISUAL_FRAME_STRIDE]
            timestamps = tuple(index / TEXT_VISUAL_FPS for index in range(sampled.shape[0]))
            visuals[reference.path] = H3TextVisual(sampled, timestamps)
    return visuals


def _ref_teacher_presentation(record, item: ItemInfo) -> H3Presentation:
    """Ref2VA teacher presentation of the item: the training crop itself as the copy-source reference.

    The visuals come from the already-decoded target crop, so the teacher sees exactly the
    trained window (a re-decode from the file would miss the crop start). The audio track is
    always declared: its latent condition at training time is the cached target audio, which is
    encoded silence for audio-less items, and the audio loss stays presence-gated there.
    """
    target_frames = torch.as_tensor(item.content)
    if target_frames.ndim != 4:
        raise ValueError(f"MiniMax-H3 target frames must be [F,H,W,C], got {tuple(target_frames.shape)}")
    reference = H3Reference(
        type="video",
        path=record.video_path,
        audio=H3AudioSource(path=record.video_path, embedded=True),
    )
    teacher_record = H3Record(
        video_path=record.video_path,
        caption=wrap_ref_teacher_caption(record.caption),
        references=(reference,),
        label=record.label,
    )
    sampled = target_frames[::TEXT_VISUAL_FRAME_STRIDE]
    # the same downscale-only canvas cap that decode_reference_visual applies to reference
    # videos: identity for targets within the released canvas (typical buckets), so only
    # oversized targets are resized and normal cache fingerprints are unaffected
    source_height, source_width = int(sampled.shape[1]), int(sampled.shape[2])
    width, height = adapt_reference_canvas(source_width, source_height)
    if source_width * source_height > width * height:
        sampled = resize_frames(sampled.numpy(), (width, height))
    timestamps = tuple(index / TEXT_VISUAL_FPS for index in range(sampled.shape[0]))
    return build_presentation(teacher_record, "ref2va", {reference.path: H3TextVisual(sampled, timestamps)})


def _subject_ref_teacher_presentation(
    record: H3Record,
    *,
    reference_frame_cap: int,
    target_size: tuple[int, int],
    still_image: bool,
    decoder: H3MediaDecoder,
    decoded_reference_cache: dict[tuple, torch.Tensor],
) -> H3Presentation:
    """Ref2VA teacher presentation of the item's own reference pictures (subject references).

    The caption is the item's ``teacher_caption`` when given, else the student caption wrapped
    in the subject-reference declaration boilerplate. The student rows never see the
    references: the same record minus references is the plain T2VA presentation.
    """
    validate_subject_reference_record(record, record.context)
    if record.teacher_caption is not None:
        caption = record.teacher_caption
    else:
        caption = wrap_subject_reference_caption(record.caption, len(record.references), still_image=still_image)
    visuals = _reference_visuals(record, reference_frame_cap, target_size, decoder, decoded_reference_cache)
    return build_presentation(replace(record, caption=caption), "ref2va", visuals)


@dataclass(frozen=True)
class _TextCacheInputs:
    """Everything the text cache of one item is built from, prepared once per item."""

    record: H3Record
    crop_start: int
    frame_count: int
    reference_frame_cap: int
    target_size: tuple[int, int]
    presentation: H3Presentation
    record_media_fingerprints: dict[Path, str]
    teacher_presentation: H3Presentation | None = None
    metadata: dict[str, str] | None = None


def _text_cache_metadata(
    *,
    task: str,
    crop_start: int,
    processor_identity: str,
    text_encoder_identity: str,
    presentation_identity: str,
    cache_dtype: str,
    teacher_conditions: str | None = None,
    teacher_presentation_identity: str | None = None,
) -> dict[str, str]:
    # cache_dtype and crop_start_frame stay so --skip_existing rebuilds when --text_cache_dtype or the
    # FL2VA crop window changes; frame_count is folded into the presentation fingerprint and the
    # behavior tags into TEXT_CACHE_FORMAT.
    metadata = {
        "task": task,
        "crop_start_frame": str(crop_start),
        "cache_format": TEXT_CACHE_FORMAT,
        "text_encoder_fingerprint": text_encoder_identity,
        "processor_fingerprint": processor_identity,
        "presentation_fingerprint": presentation_identity,
        "cache_dtype": cache_dtype,
    }
    if teacher_conditions:
        metadata["teacher_conditions"] = teacher_conditions
        metadata["teacher_presentation_fingerprint"] = teacher_presentation_identity or ""
    return metadata


def _cache_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported MiniMax-H3 text cache dtype: {name}")


def setup_parser() -> argparse.ArgumentParser:
    parser = cache_text_encoder_outputs.setup_parser_common()
    add_h3_text_encoder_args(parser, required=True)
    parser.add_argument("--task", choices=H3_TASKS, required=True)
    parser.add_argument(
        "--one_frame",
        action="store_true",
        help="experimental one-frame (image) training caches: accept image datasets. --task t2va encodes plain"
        " caption presentations; --task fl2va embeds the bucket-resized control images as <Picture i> visuals (in"
        " fp_1f_clean_indices order);"
        " --task ref2va embeds the per-item references (image_jsonl_file references, or control images without"
        " fp_1f_clean_indices) in the Ref2VA presentation",
    )
    parser.add_argument(
        "--teacher_conditions",
        type=str,
        default=None,
        help="also cache a teacher presentation for --h3_teacher_matching training (--task t2va only)."
        " 'first,last' stores the FL2VA presentation with the crop endpoints; 'ref' stores the Ref2VA"
        " presentation with the training crop itself (video + audio copy declaration) as the reference;"
        " 'subject_ref' stores the Ref2VA presentation with the item's own image references (JSONL references, or"
        " control images without fp_1f_clean_indices; subject declaration wrapped around the caption, or the item's"
        " teacher_caption), for image or video targets",
    )
    parser.add_argument("--text_cache_dtype", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--disable_numpy_memmap", action="store_true", help="disable numpy memmap while loading safetensors")
    parser.add_argument(
        "--uncond_output",
        type=str,
        default=None,
        help="also write the guidance-loss uncond probe embedding (--uncond_text) to this safetensors path,"
        " for --h3_guidance_loss_uncond_cache in training",
    )
    parser.add_argument(
        "--uncond_text",
        type=str,
        default=" ",
        help='text of the uncond probe for --uncond_output (default: a single space, the screened "space" probe)',
    )
    return parser


def main() -> None:
    args = setup_parser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    teacher_conditions = None
    if args.teacher_conditions is not None:
        if args.task != "t2va":
            raise ValueError("--teacher_conditions requires --task t2va (teacher matching trains a T2VA student)")
        teacher_conditions = normalize_teacher_conditions(args.teacher_conditions)
        if args.one_frame and teacher_conditions != TEACHER_CONDITIONS_SUBJECT_REF:
            raise ValueError("--teacher_conditions supports --one_frame with 'subject_ref' only")
    # the subject-reference teacher reads the records' references, so they are parsed as Ref2VA
    # records even though the student (and the cache task) is T2VA
    record_task = "ref2va" if teacher_conditions == TEACHER_CONDITIONS_SUBJECT_REF else args.task

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info("Loading dataset config from %s", args.dataset_config)
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_MINIMAX_H3)
    dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = dataset_group.datasets

    decoder = PyAVH3MediaDecoder()
    plans = plan_h3_datasets(datasets, task=args.task, one_frame=args.one_frame, record_task=record_task)
    if teacher_conditions == TEACHER_CONDITIONS_SUBJECT_REF:
        # fail on the data contract before any model is loaded
        for plan in plans:
            for record in plan.records:
                validate_subject_reference_record(record, record.context)

    all_cache_files, all_cache_paths = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)
    # the presentations embed the FL2VA endpoints / the reference visuals (video and one-frame
    # ref2va or subject-reference teacher alike), so those files join the identity
    text_paths = {
        path for plan in plans for record in plan.records for path in _text_media_paths(record, args.task, teacher_conditions)
    }
    # ... and one-frame fl2va presentations embed the control images
    text_paths.update(Path(path).resolve() for plan in plans for paths in plan.control_paths.values() for path in paths)
    media_fingerprints = {path: fingerprint_file(path) for path in text_paths}

    logger.info("Loading MiniMax-H3 Qwen3-VL processor")
    processor = load_h3_processor()
    processor_identity = processor_fingerprint(processor)
    text_encoder_identity = fingerprint_checkpoint(args.text_encoder)
    logger.info("Loading MiniMax-H3 text encoder from %s", args.text_encoder)
    text_encoder = load_h3_text_encoder(
        args.text_encoder,
        device=device,
        dtype=torch.bfloat16,
        disable_numpy_memmap=args.disable_numpy_memmap,
        nvfp4_scaled_mm=args.nvfp4_scaled_mm,
        blocks_to_swap=args.text_encoder_blocks_to_swap,
        attn_mode=args.text_encoder_attn_mode,
    )

    if args.uncond_output:
        presentation = H3Presentation(text=args.uncond_text, processor_text=args.uncond_text)
        hidden_states, token_tags = encode_h3_presentation(processor, text_encoder, presentation)
        save_h3_uncond_cache(
            args.uncond_output,
            hidden_states.to(_cache_dtype(args.text_cache_dtype)).cpu(),
            token_tags.cpu(),
            metadata={
                "text": args.uncond_text,
                "text_encoder_fingerprint": text_encoder_identity,
                "processor_fingerprint": processor_identity,
                "cache_dtype": args.text_cache_dtype,
            },
        )
        logger.info(
            "Saved MiniMax-H3 guidance-loss uncond cache (%d rows, text=%r): %s",
            hidden_states.shape[0],
            args.uncond_text,
            args.uncond_output,
        )

    decoded_reference_cache = {}
    # the presentations and identity metadata of the items seen so far, keyed by cache path:
    # the --skip_existing check builds them first and the encoder consumes them right after
    prepared_items: dict[str, _TextCacheInputs] = {}

    def one_frame_inputs(item: ItemInfo, plan: H3DatasetPlan) -> _TextCacheInputs:
        # one-frame image item: the caption as a T2VA presentation, an FL2VA presentation
        # embedding the bucket-resized control images, or a Ref2VA presentation embedding the
        # item's references; all time indices live in the latent cache, never in the text rows
        record = item_record(plan, item)
        visuals = {}
        record_media_fingerprints = {}
        target = torch.as_tensor(item.content)
        if target.ndim != 3:
            raise ValueError(f"MiniMax-H3 one-frame target must be [H,W,C], got {tuple(target.shape)}")
        target_size = (int(target.shape[1]), int(target.shape[0]))
        if record.references:
            # ref2va targets or subject-reference teachers (the records are parsed as Ref2VA)
            reject_one_frame_audio_references(record)
        if args.task == "ref2va":
            # the same canvas policy as the latent cache: images capped to the target
            # area, videos to the released 15 s span with 2 fps text sampling
            visuals = _reference_visuals(record, ONE_FRAME_REFERENCE_FRAME_CAP, target_size, decoder, decoded_reference_cache)
            record_media_fingerprints = {path: media_fingerprints[path] for path in _text_media_paths(record, args.task)}
        elif args.task == "fl2va":
            controls = item.control_content
            control_indices = item.fp_1f_clean_indices
            if not control_indices or controls is None or len(controls) != len(control_indices):
                raise ValueError(f"MiniMax-H3 fl2va one-frame item is missing its control images: {item.item_key}")
            for index, control in enumerate(controls):
                # the dataset keeps RGBA controls as-is; drop alpha the same way the
                # latent path does (prepare_pixels), the processor accepts only RGB
                visuals[one_frame_condition_role(index)] = H3TextVisual(torch.as_tensor(control)[..., :3].unsqueeze(0))
            record_media_fingerprints = {
                control_path: media_fingerprints[control_path]
                for control_path in item_control_paths(plan, item, len(control_indices))
            }
        # the student rows never carry the subject-reference teacher's references
        student_record = replace(record, references=()) if teacher_conditions == TEACHER_CONDITIONS_SUBJECT_REF else record
        return _TextCacheInputs(
            record=record,
            crop_start=0,
            frame_count=1,
            reference_frame_cap=ONE_FRAME_REFERENCE_FRAME_CAP,
            target_size=target_size,
            presentation=build_presentation(student_record, args.task, visuals),
            record_media_fingerprints=record_media_fingerprints,
        )

    def video_inputs(item: ItemInfo, plan: H3DatasetPlan) -> _TextCacheInputs:
        record = item_record(plan, item)
        target_frames = torch.as_tensor(item.content)
        target_size = (int(target_frames.shape[2]), int(target_frames.shape[1]))
        student_record = replace(record, references=()) if teacher_conditions == TEACHER_CONDITIONS_SUBJECT_REF else record
        visuals = _build_visuals(student_record, args.task, item, decoder, decoded_reference_cache)
        return _TextCacheInputs(
            record=record,
            crop_start=item_crop_start(item),
            frame_count=item.frame_count,
            reference_frame_cap=item.frame_count,
            target_size=target_size,
            presentation=build_presentation(student_record, args.task, visuals),
            record_media_fingerprints={path: media_fingerprints[path] for path in _text_media_paths(record, args.task)},
        )

    def prepare(item: ItemInfo) -> _TextCacheInputs:
        inputs = prepared_items.get(item.text_encoder_output_cache_path)
        if inputs is not None:
            return inputs
        plan = item_plan(plans, item)
        inputs = one_frame_inputs(item, plan) if item.frame_count is None else video_inputs(item, plan)
        record = inputs.record
        presentation_identity = presentation_fingerprint(
            inputs.presentation, inputs.record_media_fingerprints, frame_count=inputs.frame_count
        )
        teacher_presentation = None
        teacher_presentation_identity = None
        if teacher_conditions == TEACHER_CONDITIONS_SUBJECT_REF:
            # the student rows stay a plain T2VA presentation; the teacher rows are the
            # Ref2VA presentation with the item's own reference pictures (subject references)
            teacher_presentation = _subject_ref_teacher_presentation(
                record,
                reference_frame_cap=inputs.reference_frame_cap,
                target_size=inputs.target_size,
                still_image=inputs.frame_count == 1,
                decoder=decoder,
                decoded_reference_cache=decoded_reference_cache,
            )
            teacher_presentation_identity = presentation_fingerprint(
                teacher_presentation,
                {path: media_fingerprints[path] for path in _text_media_paths(record, args.task, teacher_conditions)},
                frame_count=inputs.frame_count,
            )
        elif teacher_conditions == TEACHER_CONDITIONS_REF:
            # the student rows stay a plain T2VA presentation; the teacher rows are the
            # Ref2VA presentation with the training crop itself as the copy-source reference
            teacher_presentation = _ref_teacher_presentation(record, item)
        elif teacher_conditions:
            # the student rows stay a plain T2VA presentation; the teacher rows are the
            # FL2VA presentation of the same record (first/last frames of the crop window)
            teacher_visuals = _build_visuals(record, "fl2va", item, decoder, decoded_reference_cache)
            teacher_presentation = build_presentation(record, "fl2va", teacher_visuals)
        if teacher_presentation is not None and teacher_presentation_identity is None:
            teacher_presentation_identity = presentation_fingerprint(
                teacher_presentation,
                {record.video_path: media_fingerprints[record.video_path]},
                frame_count=item.frame_count,
            )
        metadata = _text_cache_metadata(
            task=args.task,
            crop_start=inputs.crop_start,
            processor_identity=processor_identity,
            text_encoder_identity=text_encoder_identity,
            presentation_identity=presentation_identity,
            cache_dtype=args.text_cache_dtype,
            teacher_conditions=teacher_conditions,
            teacher_presentation_identity=teacher_presentation_identity,
        )
        inputs = replace(inputs, teacher_presentation=teacher_presentation, metadata=metadata)
        prepared_items[item.text_encoder_output_cache_path] = inputs
        return inputs

    def cache_is_current(item: ItemInfo) -> bool:
        # --skip_existing: a cache counts as existing only when its identity metadata matches
        if not Path(item.text_encoder_output_cache_path).is_file():
            return False
        if cache_metadata_matches(item.text_encoder_output_cache_path, prepare(item).metadata):
            logger.info("Skipping matching MiniMax-H3 text cache: %s", item.text_encoder_output_cache_path)
            prepared_items.pop(item.text_encoder_output_cache_path)  # the decoded visuals are not needed
            return True
        logger.info("Rebuilding stale MiniMax-H3 text cache: %s", item.text_encoder_output_cache_path)
        return False

    def encode(batch: list[ItemInfo]) -> None:
        for item in batch:
            inputs = prepare(item)
            prepared_items.pop(item.text_encoder_output_cache_path, None)
            presentation = inputs.presentation
            teacher_presentation = inputs.teacher_presentation
            metadata = inputs.metadata

            hidden_states, token_tags = encode_h3_presentation(processor, text_encoder, presentation)
            hidden_states = hidden_states.to(_cache_dtype(args.text_cache_dtype))
            payload_mib = hidden_states.numel() * hidden_states.element_size() / (1024**2)
            teacher_note = ""
            teacher_hidden = teacher_tags = None
            if teacher_presentation is not None:
                teacher_hidden, teacher_tags = encode_h3_presentation(processor, text_encoder, teacher_presentation)
                teacher_hidden = teacher_hidden.to(_cache_dtype(args.text_cache_dtype))
                payload_mib += teacher_hidden.numel() * teacher_hidden.element_size() / (1024**2)
                teacher_note = f", teacher_rows={teacher_hidden.shape[0]}"
            logger.info(
                "Saving MiniMax-H3 text cache for %s: rows=%d, vision_rows=%d%s, payload=%.1f MiB",
                item.item_key,
                hidden_states.shape[0],
                int((token_tags == 0).sum().item()),
                teacher_note,
                payload_mib,
            )
            save_text_encoder_output_cache_minimax_h3(
                item,
                hidden_states=hidden_states,
                token_tags=token_tags,
                teacher_kind=teacher_conditions if teacher_presentation is not None else None,
                teacher_hidden_states=teacher_hidden,
                teacher_token_tags=teacher_tags,
                metadata=metadata,
            )

    # the text caches are per crop (the FL2VA presentations embed the crop endpoints), so every
    # task walks the latent batches; a per-record text cache for t2va/ref2va is a follow-up
    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files,
        all_cache_paths,
        encode,
        requires_content=True,
        cache_is_current=cache_is_current,
    )
    cache_text_encoder_outputs.post_process_cache_files(datasets, all_cache_files, all_cache_paths, args.keep_cache)


if __name__ == "__main__":
    main()
