import psutil
import os


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    usage = {"rss": mem_info.rss, "vms": mem_info.vms}
    try:
        import torch

        if torch.cuda.is_available():
            usage["cuda"] = torch.cuda.memory_allocated()
    except ImportError:
        pass
    return usage
