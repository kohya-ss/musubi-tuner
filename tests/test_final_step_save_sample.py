from musubi_tuner.training.sampling_prompts import should_sample_at_epoch_end
from musubi_tuner.utils.train_utils import should_save_at_step


def test_should_save_at_step_periodic_hit():
    assert should_save_at_step(save_every_n_steps=500, global_step=500, max_train_steps=2000) is True


def test_should_save_at_step_not_a_multiple():
    assert should_save_at_step(save_every_n_steps=500, global_step=750, max_train_steps=2000) is False


def test_should_save_at_step_disabled():
    assert should_save_at_step(save_every_n_steps=None, global_step=500, max_train_steps=2000) is False


def test_should_save_at_step_skips_final_step_even_if_interval_matches():
    # issue #1048: interval hits exactly on the last step, but the unconditional
    # end-of-training save already writes the same weights as the "last" checkpoint.
    assert should_save_at_step(save_every_n_steps=500, global_step=2000, max_train_steps=2000) is False


def test_should_save_at_step_final_step_non_multiple_still_false():
    assert should_save_at_step(save_every_n_steps=300, global_step=2000, max_train_steps=2000) is False


def test_should_sample_at_epoch_end_no_prior_sample():
    assert should_sample_at_epoch_end(global_step=2000, last_sampled_step=None) is True


def test_should_sample_at_epoch_end_skips_duplicate_of_last_step_sample():
    # issue #1048: the in-loop step-based trigger already sampled this exact step.
    assert should_sample_at_epoch_end(global_step=2000, last_sampled_step=2000) is False


def test_should_sample_at_epoch_end_proceeds_for_different_step():
    assert should_sample_at_epoch_end(global_step=2000, last_sampled_step=1500) is True
