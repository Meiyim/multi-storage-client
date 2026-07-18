# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import os
import random
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from multistorageclient.cache import CacheManager
from multistorageclient.caching.cache_config import CacheConfig, EvictionPolicyConfig


def worker_write_read(cache_dir, keys, data, barrier, result_queue):
    """
    Worker function that use CacheManager to write and read data at random order.
    """
    try:
        cache_config = CacheConfig(size="10M", cache_line_size="64M", check_source_version=False, location=cache_dir)
        cache_manager = CacheManager(profile="test", cache_config=cache_config)

        # Synchronize all worker processes at this point
        barrier.wait()

        # Write the files at random
        random.shuffle(keys)
        for key in keys:
            cache_manager.set(key, data)

        # Read the files at random
        random.shuffle(keys)
        for key in keys:
            assert data == cache_manager.read(key)

        # Open the files at random
        random.shuffle(keys)
        for key in keys:
            fp = cache_manager.open(key, "rb")
            assert fp is not None
            assert data == fp.read()
            fp.close()

        # Synchronize all worker processes at this point
        barrier.wait()

        cache_manager.refresh_cache()

        result_queue.put(True)
    except Exception as e:
        import traceback

        traceback.print_exc()
        result_queue.put(e)


def worker_write_refresh(cache_dir, keys, data, barrier, return_dict, result_queue):
    """
    Worker function that use CacheManager to write and read data at random order.
    """
    try:
        cache_config = CacheConfig(size="10M", cache_line_size="64M", check_source_version=False, location=cache_dir)
        cache_manager = CacheManager(profile="test", cache_config=cache_config)

        # Synchronize all worker processes at this point
        barrier.wait()

        # Write the files at random
        random.shuffle(keys)
        for key in keys:
            cache_manager.set(key, data)

        # Synchronize all worker processes at this point
        barrier.wait()

        # Refresh the cache and verify the size
        with patch("multistorageclient.cache.CacheManager.evict_files", new=lambda self: time.sleep(5)):
            cache_refreshed = cache_manager.refresh_cache()

        return_dict[os.getpid()] = cache_refreshed
        result_queue.put(True)
    except Exception as e:
        import traceback

        traceback.print_exc()
        result_queue.put(e)


@pytest.fixture
def cache_dir():
    """
    Pytest fixture to create a temporary cache directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.mark.serial
def test_multiprocessing_cache_manager(cache_dir):
    """
    Test the CacheManager with multiple processes reading and writing to the cache.
    """
    num_procs = 8
    max_cache_size = num_procs * 1024 * 1024
    keys = [f"file-{i:04d}.bin" for i in range(num_procs)]
    test_data = b"*" * 1 * 1024 * 1024

    # Queue for capturing the success or failure of each process
    result_queue = multiprocessing.Queue()

    # Create a barrier that will block until all the processes reach it
    barrier = multiprocessing.Barrier(num_procs, timeout=60)

    # Create multiple processes for testing
    processes = []
    for _ in range(num_procs):
        p = multiprocessing.Process(target=worker_write_read, args=(cache_dir, keys, test_data, barrier, result_queue))
        processes.append(p)
        p.start()

    # Wait for all writer processes to finish
    for p in processes:
        p.join()

    # Check the results from the queue for reader processes
    while not result_queue.empty():
        result = result_queue.get()
        if isinstance(result, Exception):
            pytest.fail(f"Worker process failed with error: {result}")

    # Check the final cache size
    cache_config = CacheConfig(size="10M", cache_line_size="64M", check_source_version=False, location=cache_dir)
    cache_manager = CacheManager(profile="test", cache_config=cache_config)
    assert cache_manager.cache_size() <= max_cache_size


@pytest.mark.serial
def test_multiprocessing_cache_manager_single_refresh(cache_dir):
    num_procs = 8
    keys = [f"file-{i:04d}.bin" for i in range(num_procs * 10)]
    test_data = b"*" * 10 * 1024 * 1024

    with multiprocessing.Manager() as manager:
        # Shared dictionary for collecting results from worker processes
        return_dict = manager.dict()

        # Queue for capturing the success or failure of each process
        result_queue = multiprocessing.Queue()

        # Create a barrier that will block until all the processes reach it
        barrier = multiprocessing.Barrier(num_procs, timeout=60)

        # Create multiple processes for testing
        processes = []
        for _ in range(num_procs):
            p = multiprocessing.Process(
                target=worker_write_refresh, args=(cache_dir, keys, test_data, barrier, return_dict, result_queue)
            )
            processes.append(p)
            p.start()

        # Wait for all writer processes to finish
        for p in processes:
            p.join()

        # Check the results from the queue for reader processes
        while not result_queue.empty():
            result = result_queue.get()
            if isinstance(result, Exception):
                pytest.fail(f"Worker process failed with error: {result}")

        # Verify only one process refreshed the cache
        assert len([d for d in return_dict.values() if d is True]) == 1


def worker_overfill(cache_dir, prefix, num_files, data, cap, purge_factor, barrier, result_queue):
    """Concurrently write many unique files (well past the cap) into a shared cache,
    refreshing along the way — mimics many DataLoader workers hammering one tmpfs cache."""
    try:
        cache_config = CacheConfig(
            size=cap,
            cache_line_size="64M",
            check_source_version=False,
            location=cache_dir,
            eviction_policy=EvictionPolicyConfig(policy="fifo", purge_factor=purge_factor),
        )
        cache_manager = CacheManager(profile="test", cache_config=cache_config)
        barrier.wait()
        for i in range(num_files):
            cache_manager.set(f"{prefix}/file-{i:03d}.bin", data)
            # Periodically attempt a refresh, as real workers do on cache ops.
            if i % 4 == 0:
                cache_manager._last_refresh_time = datetime.now() - timedelta(
                    seconds=cache_manager._cache_refresh_interval + 1
                )
                cache_manager.refresh_cache()
        result_queue.put(True)
    except Exception as e:
        import traceback

        traceback.print_exc()
        result_queue.put(e)


@pytest.mark.serial
@pytest.mark.skipif(
    not os.path.isdir("/dev/shm") or not os.access("/dev/shm", os.W_OK),
    reason="tmpfs (/dev/shm) not available",
)
def test_multiprocessing_tmpfs_eviction_bounds_cache():
    """Concurrent writers over-filling a shared tmpfs cache must stay bounded by eviction.

    Scaled-down reproduction of the cook's failure mode: many processes hammering one
    shared /dev/shm cache far past its cap. After the writers finish and an authoritative
    refresh runs, the cache must be bounded to its size cap, and purge_factor must have
    created headroom. Guards process-safe eviction on tmpfs (the real deployment target).
    """
    cache_dir = os.path.join("/dev/shm", f"msc_mp_evict_{uuid.uuid4().hex}")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        num_procs = 6
        files_per_proc = 10
        cap_mb = 15
        purge_factor = 50
        data = b"*" * 1024 * 1024  # 1 MB
        # total unique data = 6 * 10 = 60 MB into a 15 MB cap => 4x concurrent over-fill

        result_queue = multiprocessing.Queue()
        barrier = multiprocessing.Barrier(num_procs, timeout=60)
        processes = []
        for pi in range(num_procs):
            p = multiprocessing.Process(
                target=worker_overfill,
                args=(cache_dir, f"p{pi}", files_per_proc, data, f"{cap_mb}M", purge_factor, barrier, result_queue),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join(timeout=120)

        while not result_queue.empty():
            result = result_queue.get()
            if isinstance(result, Exception):
                pytest.fail(f"Worker process failed with error: {result}")

        # Authoritative final eviction pass from the test process.
        cache_config = CacheConfig(
            size=f"{cap_mb}M",
            cache_line_size="64M",
            check_source_version=False,
            location=cache_dir,
            eviction_policy=EvictionPolicyConfig(policy="fifo", purge_factor=purge_factor),
        )
        cache_manager = CacheManager(profile="test", cache_config=cache_config)
        cache_manager._last_refresh_time = datetime.now() - timedelta(
            seconds=cache_manager._cache_refresh_interval + 1
        )
        cache_manager.refresh_cache()

        cap_bytes = cap_mb * 1024 * 1024
        target_bytes = cap_bytes * (1 - purge_factor / 100.0)  # 7.5 MB
        size_after = cache_manager.cache_size()
        assert size_after <= cap_bytes, f"cache {size_after} exceeds cap {cap_bytes} after concurrent over-fill"
        assert size_after <= target_bytes + 2 * 1024 * 1024, (
            f"purge_factor={purge_factor} should evict to ~{target_bytes} bytes, got {size_after}"
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
