import argparse

from musubi_tuner.lens import lens_utils


def main():
    parser = argparse.ArgumentParser(description="Download Lens MVP components")
    parser.add_argument("--output_dir", type=str, required=True, help="directory to download model components into")
    parser.add_argument("--repo_id", type=str, default=lens_utils.COMFY_LENS_REPO_ID, help="Comfy Lens weights repo")
    parser.add_argument("--metadata_repo_id", type=str, default=lens_utils.OFFICIAL_LENS_REPO_ID, help="official Lens metadata repo")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token")
    parser.add_argument("--dry_run", action="store_true", help="list files without downloading")
    args = parser.parse_args()

    downloads = lens_utils.download_lens_components(
        args.output_dir,
        repo_id=args.repo_id,
        metadata_repo_id=args.metadata_repo_id,
        dry_run=args.dry_run,
        token=args.token,
    )
    for repo_id, filename in downloads:
        print(f"{repo_id}/{filename}")

    if not args.dry_run:
        print("\nReady-to-use paths:")
        print(f"  --dit {args.output_dir}/diffusion_models/lens_bf16.safetensors")
        print(f"  --text_encoder {args.output_dir}/text_encoders/gpt_oss_20b_nvfp4.safetensors")
        print(f"  --text_encoder_config {args.output_dir}/text_encoder")
        print(f"  --tokenizer {args.output_dir}/tokenizer")
        print(f"  --vae {args.output_dir}/vae/flux2-vae.safetensors")


if __name__ == "__main__":
    main()
