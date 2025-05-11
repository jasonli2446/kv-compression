from evaluation.memory import get_memory_usage


def test_get_memory_usage():
    usage = get_memory_usage()
    assert "rss" in usage
    assert "vms" in usage
