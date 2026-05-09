import torch

class TorchProfiler:
    def __init__(self, filename: str, enabled: bool = True):
        self.filename = filename
        self.enabled = enabled
        self.profiler = None

    def __enter__(self):
        if self.enabled:
            profiler_context = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ],
            )
            self.profiler = profiler_context.__enter__()
            return self.profiler
        else:
            return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiler is not None:
            ret = self.profiler.__exit__(exc_type, exc_val, exc_tb)
            try:
                self.profiler.export_chrome_trace(self.filename)
            except Exception:
                print(f"could not write profiler output {self.filename}")
            return ret
        else:
            return False
