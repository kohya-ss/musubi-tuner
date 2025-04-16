import time
import os
import shutil


def process_transformer_cfg(cfg):
    log_dir = ""
    if "critical_params" in cfg:
        critical_params = [cfg[key] for key in cfg.critical_params]
        for name, param in zip(cfg["critical_params"], critical_params):
            log_dir += "{:s}[{:s}]".format(name, str(param))

    return log_dir


def process_cfg(cfg):
    log_dir = "logs/" + cfg.name + "/" + cfg.transformer + "/"
    critical_params = [cfg.trainer[key] for key in cfg.critical_params]
    for name, param in zip(cfg["critical_params"], critical_params):
        log_dir += "{:s}[{:s}]".format(name, str(param))

    log_dir += process_transformer_cfg(cfg[cfg.transformer])

    now = time.localtime()
    now_time = "{:02d}_{:02d}_{:02d}_{:02d}".format(
        now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min
    )
    log_dir += cfg.suffix + "(" + now_time + ")"
    cfg.log_dir = log_dir
    os.makedirs(log_dir)

    shutil.copytree("configs", f"{log_dir}/configs")
    shutil.copytree("core/FlowFormer", f"{log_dir}/FlowFormer")
