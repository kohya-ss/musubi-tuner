from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_top_level_entrypoints_are_exact_shims():
    expected = {
        "flux_2_train.py": "musubi_tuner.flux_2_train",
        "flux_kontext_train.py": "musubi_tuner.flux_kontext_train",
        "ideogram4_cache_latents.py": "musubi_tuner.ideogram4_cache_latents",
        "ideogram4_cache_text_encoder_outputs.py": "musubi_tuner.ideogram4_cache_text_encoder_outputs",
        "ideogram4_generate_image.py": "musubi_tuner.ideogram4_generate_image",
        "ideogram4_train.py": "musubi_tuner.ideogram4_train",
        "ideogram4_train_network.py": "musubi_tuner.ideogram4_train_network",
        "krea2_train.py": "musubi_tuner.krea2_train",
    }

    for script_name, module_name in expected.items():
        script = ROOT / script_name
        assert script.exists(), f"missing top-level entrypoint: {script_name}"
        assert script.read_text(encoding="utf-8") == (f'from {module_name} import main\n\nif __name__ == "__main__":\n    main()\n')
