"""Tests for tabpfn.parallel_execute."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing_extensions import override

import torch
from torch import Tensor, nn

from tabpfn.architectures.interface import Architecture
from tabpfn.inference import _PerDeviceModelCache
from tabpfn.parallel_execute import parallel_execute


def test__parallel_execute__single_device__executes_in_current_thread() -> None:
    def test_function(device: torch.device) -> int:  # noqa: ARG001
        return threading.get_ident()

    thread_ids = parallel_execute(
        devices=[torch.device("cpu")], functions=[test_function, test_function]
    )

    current_thread_id = threading.get_ident()
    assert list(thread_ids) == [current_thread_id, current_thread_id]


def test__parallel_execute__single_device__results_in_same_order_as_functions() -> None:
    def a(device: torch.device) -> str:  # noqa: ARG001
        return "a"

    def b(device: torch.device) -> str:  # noqa: ARG001
        return "b"

    def c(device: torch.device) -> str:  # noqa: ARG001
        return "c"

    results = parallel_execute(devices=[torch.device("cpu")], functions=[a, b, c])

    assert list(results) == ["a", "b", "c"]


def test__parallel_execute__multiple_devices__executes_in_worker_threads() -> None:
    def test_function(device: torch.device) -> int:  # noqa: ARG001
        return threading.get_ident()

    thread_ids = parallel_execute(
        devices=[torch.device("cpu"), torch.device("meta")],
        functions=[test_function, test_function],
    )

    current_thread_id = threading.get_ident()
    for thread_id in thread_ids:
        assert thread_id != current_thread_id


def test__parallel_execute__multiple_devices__results_in_same_order_as_functions() -> (
    None
):
    def a(device: torch.device) -> str:  # noqa: ARG001
        return "a"

    def b(device: torch.device) -> str:  # noqa: ARG001
        return "b"

    def c(device: torch.device) -> str:  # noqa: ARG001
        return "c"

    results = parallel_execute(
        devices=[torch.device("meta"), torch.device("meta")], functions=[a, b, c]
    )

    assert list(results) == ["a", "b", "c"]


class _SimpleTestModel(Architecture):
    """Minimal Architecture for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.tensor([1.0]))

    @override
    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        y: Tensor | dict[str, Tensor] | None,
        *,
        only_return_standard_out: bool = True,
        categorical_inds: list[list[int]] | None = None,
        force_recompute_layer: bool = False,
        save_peak_memory_factor: int | None = None,
    ) -> Tensor | dict[str, Tensor]:
        return self.param


class TestPerDeviceModelCacheThreadSafety:
    """Thread-safety tests for _PerDeviceModelCache.

    These tests verify that concurrent access to the model cache from multiple
    threads (as happens during multi-GPU inference via ThreadPool) does not
    cause race conditions.
    """

    @staticmethod
    def _make_cache() -> tuple[_PerDeviceModelCache, torch.device]:
        model = _SimpleTestModel()
        cache = _PerDeviceModelCache(model)
        device = torch.device("cpu")
        cache.to([device])
        return cache, device

    @staticmethod
    def _run_threads(targets: list[tuple[Callable[[], None], int]]) -> None:
        """Run callables in threads. Each tuple is (target, count)."""
        threads = [
            threading.Thread(target=target)
            for target, count in targets
            for _ in range(count)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def test__concurrent_get(self) -> None:
        """Concurrent get() calls should not raise."""
        cache, device = self._make_cache()
        errors: list[Exception] = []

        def reader() -> None:
            for _ in range(100):
                try:
                    cache.get(device)
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

        self._run_threads([(reader, 10)])
        assert not errors

    def test__concurrent_get_and_to(self) -> None:
        """Concurrent get() and to() should not corrupt state."""
        cache, device = self._make_cache()
        errors: list[Exception] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                try:
                    cache.get(device)
                except KeyError:
                    pass  # Expected during transitions
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

        def writer() -> None:
            for _ in range(20):
                cache.to([device])
            stop.set()

        threads = [threading.Thread(target=reader) for _ in range(5)]
        threads.append(threading.Thread(target=writer))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test__concurrent_set_dtype(self) -> None:
        """Concurrent set_dtype() calls should not raise."""
        cache, _ = self._make_cache()
        errors: list[Exception] = []

        def changer(dtype: torch.dtype) -> None:
            for _ in range(20):
                try:
                    cache.set_dtype(dtype)
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

        threads = [
            threading.Thread(target=changer, args=(dtype,))
            for dtype in [torch.float32, torch.float64]
            for _ in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test__get_devices_returns_consistent_list(self) -> None:
        """get_devices() should never return empty during to()."""
        cache, device = self._make_cache()
        empty_results: list[list[torch.device]] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                devices = cache.get_devices()
                if not devices:
                    empty_results.append(devices)

        def writer() -> None:
            for _ in range(20):
                cache.to([device])
            stop.set()

        threads = [threading.Thread(target=reader) for _ in range(5)]
        threads.append(threading.Thread(target=writer))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not empty_results

    def test__high_contention(self) -> None:
        """All operations mixed under high contention should not raise."""
        cache, device = self._make_cache()
        errors: list[Exception] = []

        def mixed_ops(op_type: int) -> None:
            for _ in range(50):
                try:
                    if op_type == 0:
                        cache.get(device)
                    elif op_type == 1:
                        cache.to([device])
                    elif op_type == 2:
                        cache.set_dtype(torch.float32)
                    else:
                        cache.get_devices()
                except KeyError:
                    pass
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(mixed_ops, i % 4) for i in range(20)]
            for f in futures:
                f.result()

        assert not errors

    def test__parallel_execute_pattern(self) -> None:
        """Simulate the actual multi-GPU inference pattern from parallel_execute.

        Without the RLock fix, this would cause RuntimeError (dictionary changed
        size during iteration) or corrupt state when ThreadPool workers call
        get() while another thread calls to().
        """
        cache, device = self._make_cache()
        errors: list[Exception] = []

        def worker() -> None:
            for _ in range(200):
                try:
                    model = cache.get(device)
                    list(model.parameters())  # Simulate forward pass setup
                except KeyError:
                    pass
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

        def switcher() -> None:
            for _ in range(50):
                try:
                    cache.to([device])
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        threads.append(threading.Thread(target=switcher))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
