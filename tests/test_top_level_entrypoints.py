from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ideogram4_top_level_entrypoints_exist():
    expected = {
        "ideogram4_cache_latents.py": "musubi_tuner.ideogram4_cache_latents",
        "ideogram4_cache_text_encoder_outputs.py": "musubi_tuner.ideogram4_cache_text_encoder_outputs",
        "ideogram4_generate_image.py": "musubi_tuner.ideogram4_generate_image",
        "ideogram4_train_network.py": "musubi_tuner.ideogram4_train_network",
    }

    for script_name, module_name in expected.items():
        script = ROOT / script_name
        assert script.exists(), f"missing top-level entrypoint: {script_name}"
        assert script.read_text(encoding="utf-8") == (
            f"from {module_name} import main\n"
            "\n"
            'if __name__ == "__main__":\n'
            "    main()\n"
        )
