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

import os
import tempfile
import time
import uuid

import pytest

import multistorageclient as msc
from multistorageclient.types import Range

# A real (large, ~14.8 GiB) BOS object used for the download-timing test.
# Credentials are supplied via the BCE_ACCESS_KEY_ID / BCE_SECRET_ACCESS_KEY
# environment variables (read by bosfs).
BOS_URL = "bos://nlp-ernie4/chenxuyi/data/wds/nemotron_phase1/10000200000499/part_000065.tar"

# Writable scratch prefix for the bulk-upload test (cleaned up afterwards).
BOS_UPLOAD_ROOT = os.environ.get("MSC_BOS_UPLOAD_ROOT", "bos://nlp-ernie4/chenxuyi/tmp/msc_ckpt_upload_test")

requires_bos_credentials = pytest.mark.skipif(
    not (os.environ.get("BCE_ACCESS_KEY_ID") and os.environ.get("BCE_SECRET_ACCESS_KEY")),
    reason="BOS credentials (BCE_ACCESS_KEY_ID / BCE_SECRET_ACCESS_KEY) are not configured.",
)


@requires_bos_credentials
def test_bos_open_and_read():
    """Open a real BOS object through the ``bos://`` implicit profile and read the first bytes."""
    with msc.open(BOS_URL, "rb") as f:
        head = f.read(512)

    assert head, "Expected to read a non-empty prefix from the BOS object."


@requires_bos_credentials
def test_bos_ranged_read_matches_reference():
    """Exercise the ranged-read path used by Megatron-Energon's tar readers.

    Energon opens a shard with ``EPath.open("rb")`` (``prefetch_file=False``) and then does
    ``seek(offset)`` + ``read(n)``. In MSC that becomes ``RemoteFileReader`` issuing
    ``StorageClient.read(path, byte_range=Range(offset, size))`` -> provider ``_get_object``.
    This test verifies those ranged reads return exactly ``size`` bytes from ``offset`` and
    match a locally-held reference prefix -- covering the ``offset == 0`` case that bosfs's
    ``cat_file`` mishandles.
    """
    client, path = msc.resolve_storage_client(BOS_URL)

    # Reference: the first 4 MiB of the object, fetched in one whole-range read.
    reference_len = 4 * 1024 * 1024
    reference = client.read(path, byte_range=Range(offset=0, size=reference_len))
    assert len(reference) == reference_len

    # Ranged reads at various offsets (including 0) must return exactly `size` bytes
    # and match the reference slice.
    for offset, size in [(0, 512), (0, 11), (100, 4096), (1_000_000, 65536), (reference_len - 10, 10)]:
        chunk = client.read(path, byte_range=Range(offset=offset, size=size))
        assert len(chunk) == size, f"offset={offset} size={size}: got {len(chunk)} bytes"
        assert chunk == reference[offset : offset + size], f"offset={offset} size={size}: content mismatch"


@requires_bos_credentials
def test_bos_open_seek_read_like_energon():
    """Drive the exact file-object API Energon uses: open -> seek -> read on the same handle."""
    client, path = msc.resolve_storage_client(BOS_URL)
    whole_prefix = client.read(path, byte_range=Range(offset=0, size=8192))

    with msc.open(BOS_URL, "rb") as f:
        # First read starts at offset 0 (the bosfs cat_file bug case).
        first = f.read(512)
        assert first == whole_prefix[:512]

        # Seek forward and read a mid-file region (SubfileReader pattern).
        f.seek(1024)
        mid = f.read(2048)
        assert mid == whole_prefix[1024:3072]

        # Absolute re-seek backwards still works.
        f.seek(0)
        assert f.read(16) == whole_prefix[:16]


@requires_bos_credentials
def test_bos_download_timing(capsys):
    """Download a large BOS object into RAM and report the elapsed time and throughput.

    The download target lives on a RAM-backed tmpfs (``/dev/shm`` by default, overridable
    via the ``MSC_E2E_RAM_DIR`` environment variable) so no large file touches disk.
    """
    expected_size = msc.info(BOS_URL).content_length

    ram_dir = os.environ.get("MSC_E2E_RAM_DIR", "/dev/shm")
    if not os.path.isdir(ram_dir):
        pytest.skip(f"RAM-backed directory {ram_dir!r} is not available.")

    with tempfile.TemporaryDirectory(dir=ram_dir) as tmp_dir:
        local_path = os.path.join(tmp_dir, "part_000065.tar")

        start = time.perf_counter()
        msc.download_file(BOS_URL, local_path)
        elapsed = time.perf_counter() - start

        downloaded_size = os.path.getsize(local_path)

    assert downloaded_size == expected_size, f"Downloaded {downloaded_size} bytes, expected {expected_size}."

    mib = downloaded_size / 1024 / 1024
    throughput = mib / elapsed if elapsed > 0 else float("inf")
    with capsys.disabled():
        print(
            f"\n[BOS download → RAM ({ram_dir})] {BOS_URL}\n"
            f"  size       : {downloaded_size:,} bytes ({mib:,.1f} MiB)\n"
            f"  elapsed    : {elapsed:.2f} s\n"
            f"  throughput : {throughput:,.1f} MiB/s"
        )


@requires_bos_credentials
def test_bos_bulk_checkpoint_upload_like_megatron(capsys):
    """Bulk-upload many large files mirroring Megatron-LM's async checkpoint writer.

    Reproduces the MSC upload call path *and the threading model* in
    ``megatron/core/dist_checkpointing/strategies/filesystem_async.py``
    (``FileSystemWriterAsync.write_preloaded_data_multithread`` /
    ``write_preloaded_data``):

        msc.os.makedirs(checkpoint_dir, exist_ok=True)
        # one thread per bucket, except the last bucket runs on the calling thread;
        # all threads start together, then join -> buckets upload concurrently.
        with msc.open(file_name, "wb") as stream:
            stream.write(<tensor/bytes data>)   # _write_item writes into the stream
            stream.fsync()                       # use_fsync=True (no-op for object stores)
        with msc.open(metadata_path, "wb") as f: # .metadata (written after, single file)
            f.write(...)

    Each bucket is written through the ``bos`` provider's ``_upload_file`` / ``_put_object``
    path. Sizes/contents are verified by reading back, then everything is deleted.

    Scale is overridable: ``MSC_BOS_UPLOAD_SHARDS`` (default 8) and
    ``MSC_BOS_UPLOAD_SHARD_MIB`` (default 32) -> ~256 MiB total by default.
    """
    import threading

    n_shards = int(os.environ.get("MSC_BOS_UPLOAD_SHARDS", "8"))
    shard_mib = int(os.environ.get("MSC_BOS_UPLOAD_SHARD_MIB", "32"))
    shard_size = shard_mib * 1024 * 1024
    shard_mib_f = shard_size / 1024 / 1024

    ckpt_dir = f"{BOS_UPLOAD_ROOT}/{uuid.uuid4().hex}"
    shard_names = [f"{ckpt_dir}/__{i}_0.distcp" for i in range(n_shards)]
    metadata_path = f"{ckpt_dir}/.metadata"
    # Precompute cleanup targets so orphans are removed even if a writer fails.
    written = list(shard_names) + [metadata_path]

    def shard_payload(index: int) -> bytes:
        # 32-byte identifiable header + filler, so we can verify each shard cheaply.
        header = f"msc-bos-shard-{index:04d}".encode().ljust(32, b"\0")
        return header + bytes(shard_size - len(header))

    # Per-bucket writer, identical to write_preloaded_data's inner body.
    shard_result: list = [None] * n_shards  # elapsed seconds, or an Exception

    def _write_bucket(i: int) -> None:
        try:
            payload = shard_payload(i)
            t0 = time.perf_counter()
            with msc.open(shard_names[i], "wb") as stream:
                stream.write(payload)
                if hasattr(stream, "fsync"):
                    stream.fsync()  # upload commits on close() (end of with-block)
            shard_result[i] = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001 — surfaced after join, mirrors Megatron
            shard_result[i] = e

    try:
        # 1. Create the checkpoint directory (no-op marker on object stores).
        msc.os.makedirs(ckpt_dir, exist_ok=True)

        # 2. Concurrent bulk write: one thread per bucket except the last (main thread).
        start = time.perf_counter()
        threads = [threading.Thread(target=_write_bucket, args=(i,)) for i in range(n_shards - 1)]
        for t in threads:
            t.start()
        _write_bucket(n_shards - 1)  # last bucket on the calling thread, like Megatron
        for t in threads:
            t.join()
        upload_elapsed = time.perf_counter() - start

        # Surface any worker error (Megatron collects these via its results queue).
        errors = [r for r in shard_result if isinstance(r, Exception)]
        assert not errors, f"{len(errors)} shard upload(s) failed: {errors[0]!r}"

        # 3. Write the checkpoint metadata file (single, after the buckets).
        with msc.open(metadata_path, "wb") as stream:
            stream.write(b"MSC-BOS-CKPT-METADATA")

        # 4. Verify every shard: size on store + identifiable header prefix.
        client, _ = msc.resolve_storage_client(ckpt_dir + "/")
        for i, file_name in enumerate(shard_names):
            _, path = msc.resolve_storage_client(file_name)
            assert msc.info(file_name).content_length == shard_size, file_name
            head = client.read(path, byte_range=Range(offset=0, size=32))
            assert head == f"msc-bos-shard-{i:04d}".encode().ljust(32, b"\0"), file_name
        assert msc.info(metadata_path).content_length == len(b"MSC-BOS-CKPT-METADATA")

        # Profile: aggregate (wall-clock, concurrent) vs per-shard rates.
        per_shard_rates = [shard_mib_f / e for e in shard_result]
        total_mib = n_shards * shard_mib_f
        aggregate_rate = total_mib / upload_elapsed if upload_elapsed > 0 else float("inf")
        min_rate, max_rate = min(per_shard_rates), max(per_shard_rates)
        avg_rate = sum(per_shard_rates) / len(per_shard_rates)
        with capsys.disabled():
            print(
                f"\n[BOS bulk checkpoint upload — parallel like Megatron] {ckpt_dir}\n"
                f"  shards          : {n_shards} x {shard_mib} MiB (concurrent, thread-per-bucket)\n"
                f"  total           : {total_mib:,.1f} MiB\n"
                f"  wall-clock      : {upload_elapsed:.2f} s\n"
                f"  aggregate rate  : {aggregate_rate:,.1f} MiB/s  (total / wall-clock)\n"
                f"  per-shard rate  : avg {avg_rate:,.1f} | min {min_rate:,.1f} | "
                f"max {max_rate:,.1f} MiB/s"
            )
            for i, rate in enumerate(per_shard_rates):
                print(f"    shard {i:02d}      : {rate:,.1f} MiB/s ({shard_result[i]:.2f} s)")
    finally:
        # Clean up every object we created so the bucket isn't polluted.
        for file_name in written:
            try:
                msc.delete(file_name)
            except Exception:
                pass
