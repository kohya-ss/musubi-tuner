from torch.utils.tensorboard import SummaryWriter
from loguru import logger as loguru_logger


class Logger:
    def __init__(self, model, scheduler, cfg):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.cfg = cfg

    def _print_training_status(self):
        metrics_data = [
            self.running_loss[k] / self.cfg.sum_freq
            for k in sorted(self.running_loss.keys())
        ]
        training_str = "[{:6d}, {}] ".format(
            self.total_steps + 1, self.scheduler.get_last_lr()
        )
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        loguru_logger.info(training_str + metrics_str)

        if self.writer is None:
            if self.cfg.log_dir is None:
                self.writer = SummaryWriter()
            else:
                self.writer = SummaryWriter(self.cfg.log_dir)

        for k in self.running_loss:
            self.writer.add_scalar(
                k, self.running_loss[k] / self.cfg.sum_freq, self.total_steps
            )
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.cfg.sum_freq == self.cfg.sum_freq - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()
