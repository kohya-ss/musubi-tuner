import io
from pathlib import Path

import pytest
from tqdm import tqdm

from musubi_tuner.utils import train_utils


class ManualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def make_progress_bar(*, smoothing: float = 0.0):
    clock = ManualClock()
    output = io.StringIO()
    progress_bar = tqdm(total=10, smoothing=smoothing, mininterval=0, miniters=1, file=output)
    progress_bar._time = clock
    progress_bar.start_t = clock.value
    progress_bar.last_print_t = clock.value
    return progress_bar, clock, output


def test_first_step_is_baseline_for_rate_and_eta():
    progress_bar, clock, _ = make_progress_bar()
    try:
        clock.value = 10.0
        progress_bar.update(1)
        train_utils.reset_progress_bar_timing(progress_bar)

        assert progress_bar.n == 1
        assert progress_bar.initial == 1
        assert progress_bar.total == 10
        assert progress_bar.format_dict["elapsed"] == 0
        assert "?it/s" in tqdm.format_meter(**progress_bar.format_dict)

        clock.value = 14.0
        progress_bar.update(1)
        assert progress_bar.format_dict["elapsed"] == 4
        meter = tqdm.format_meter(**progress_bar.format_dict)
        assert "4.00s/it" in meter
        assert "<00:32" in meter

        clock.value = 22.0
        progress_bar.update(2)
        meter = tqdm.format_meter(**progress_bar.format_dict)
        assert "4.00s/it" in meter
        assert "<00:24" in meter
    finally:
        progress_bar.close()


def test_timing_reset_clears_exponential_average_history():
    progress_bar, clock, _ = make_progress_bar(smoothing=0.3)
    try:
        clock.value = 10.0
        progress_bar.update(1)
        assert progress_bar._ema_dt.calls == 1

        train_utils.reset_progress_bar_timing(progress_bar)
        assert progress_bar._ema_dn.calls == 0
        assert progress_bar._ema_dt.calls == 0
        assert progress_bar._ema_miniters.calls == 0

        clock.value = 14.0
        progress_bar.update(1)
        assert progress_bar.format_dict["rate"] == pytest.approx(0.25)
    finally:
        progress_bar.close()


def test_disabled_progress_bar_is_unchanged():
    progress_bar = tqdm(total=10, disable=True)
    train_utils.reset_progress_bar_timing(progress_bar)
    assert progress_bar.n == 0
    assert progress_bar.total == 10


def test_all_training_loops_reset_after_first_synchronized_loss():
    source_root = Path(__file__).resolve().parents[1] / "src" / "musubi_tuner"
    training_files = (
        source_root / "training" / "trainer_base.py",
        source_root / "hv_train.py",
        source_root / "qwen_image_train.py",
        source_root / "zimage_train.py",
        source_root / "hidream_o1_train.py",
    )

    for path in training_files:
        source = path.read_text(encoding="utf-8")
        update = source.index("progress_bar.update(1)")
        increment = source.index("global_step += 1", update)
        synchronized_loss = source.index("current_loss = loss.detach().item()", increment)
        reset = source.index("train_utils.reset_progress_bar_timing(progress_bar)", synchronized_loss)

        assert update < increment < synchronized_loss < reset, path
        assert "if accelerator.sync_gradients and global_step == 1:" in source[increment:reset]
        assert "progress_bar.reset()" not in source
