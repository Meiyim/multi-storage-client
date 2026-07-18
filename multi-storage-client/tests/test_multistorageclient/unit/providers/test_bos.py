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

from unittest.mock import MagicMock, patch

import pytest

from multistorageclient.providers.bos import PROVIDER, BaiduBosStorageProvider


def _make_provider(**kwargs) -> tuple[BaiduBosStorageProvider, MagicMock]:
    """Construct a bosfs-backed provider with bosfs.BOSFileSystem mocked out (no network).

    The backend is built lazily, so we force it inside the patch context (via
    backend_name) to capture the mocked filesystem.
    """
    kwargs.setdefault("backend", "bosfs")
    fs = MagicMock(name="BOSFileSystem")
    with patch("bosfs.BOSFileSystem", return_value=fs) as ctor:
        provider = BaiduBosStorageProvider(base_path="my-bucket", **kwargs)
        assert provider.backend_name == "bosfs"  # forces lazy backend build under the patch
    provider._ctor = ctor  # type: ignore[attr-defined]
    return provider, fs


def test_bos_defaults_to_beijing_http_endpoint():
    provider, _ = _make_provider()

    assert provider._provider_name == PROVIDER
    assert provider._region_name == "bj"
    assert provider._endpoint_url == "http://bj.bcebos.com"
    provider._ctor.assert_called_once()
    assert provider._ctor.call_args.kwargs["endpoint"] == "http://bj.bcebos.com"


def test_bos_derives_endpoint_from_region():
    provider, _ = _make_provider(region_name="gz")

    assert provider._region_name == "gz"
    assert provider._endpoint_url == "http://gz.bcebos.com"


def test_bos_respects_explicit_endpoint():
    provider, _ = _make_provider(endpoint_url="http://fwh.bcebos.com")

    assert provider._endpoint_url == "http://fwh.bcebos.com"


def test_bos_get_object_reads_via_bosfs():
    provider, fs = _make_provider()
    fs.cat_file.return_value = b"hello"

    assert provider._get_object("my-bucket/key.txt") == b"hello"
    fs.cat_file.assert_called_once_with("my-bucket/key.txt")


def test_bos_get_object_range_reads_exact_size_via_file_handle():
    from multistorageclient.types import Range

    provider, fs = _make_provider()
    handle = MagicMock()
    handle.read.return_value = b"ell"
    # Support the ``with`` context manager protocol.
    fs.open.return_value.__enter__.return_value = handle

    result = provider._get_object("my-bucket/key.txt", byte_range=Range(offset=1, size=3))

    assert result == b"ell"
    fs.open.assert_called_once_with("my-bucket/key.txt", "rb")
    handle.seek.assert_called_once_with(1)
    handle.read.assert_called_once_with(3)


def test_bos_put_object_rejects_conditional_writes():
    provider, _ = _make_provider()
    with pytest.raises(NotImplementedError):
        provider._put_object("my-bucket/key.txt", b"data", if_none_match="*")


def test_bos_symlink_not_supported():
    provider, _ = _make_provider()
    with pytest.raises(NotImplementedError):
        provider._make_symlink("my-bucket/link", "my-bucket/target")


def test_bos_list_objects_recursive_yields_files_sorted():
    provider, fs = _make_provider()
    # Recursive listing goes through bosfs's flat page accumulator (delimiter="")
    # rather than fsspec find() (delimiter="/"). See
    # test_bos_recursive_listing_uses_flat_accumulator_not_find for why.
    fs._get_object_info_list.return_value = [
        {"name": "my-bucket/b.txt", "type": "file", "size": 2},
        {"name": "my-bucket/a.txt", "type": "file", "size": 1},
        {"name": "my-bucket/sub", "type": "directory", "size": 0},
    ]

    keys = [m.key for m in provider._list_objects("my-bucket")]

    # Sorted, directories excluded when include_directories=False.
    assert keys == ["my-bucket/a.txt", "my-bucket/b.txt"]


def test_bos_list_objects_start_after_and_end_at():
    provider, fs = _make_provider()
    fs._get_object_info_list.return_value = [
        {"name": f"my-bucket/{n}", "type": "file", "size": 1} for n in ("a", "b", "c", "d")
    ]

    keys = [m.key for m in provider._list_objects("my-bucket", start_after="my-bucket/a", end_at="my-bucket/c")]

    assert keys == ["my-bucket/b", "my-bucket/c"]


def test_backend_defaults_to_go_when_available():
    """backend='auto' selects the Go backend when it constructs successfully."""
    fake_go = MagicMock(name="GoBackendInstance")
    fake_go.name = "go"
    with patch("multistorageclient.providers.bos._BosGoBackend", return_value=fake_go) as go_ctor:
        provider = BaiduBosStorageProvider(base_path="my-bucket")  # backend defaults to "auto"
        assert provider.backend_name == "go"  # lazy build under the patch
    go_ctor.assert_called_once()


def test_backend_auto_falls_back_to_bosfs_when_go_unavailable():
    """backend='auto' falls back to bosfs when the Go backend can't be built."""
    fs = MagicMock(name="BOSFileSystem")
    with (
        patch("multistorageclient.providers.bos._BosGoBackend", side_effect=ImportError("no bos_tool")),
        patch("bosfs.BOSFileSystem", return_value=fs),
    ):
        provider = BaiduBosStorageProvider(base_path="my-bucket", backend="auto")
        assert provider.backend_name == "bosfs"


def test_backend_go_forced_raises_when_unavailable():
    """backend='go' surfaces the error instead of silently falling back."""
    with patch("multistorageclient.providers.bos._BosGoBackend", side_effect=ImportError("no bos_tool")):
        with pytest.raises(ImportError):
            BaiduBosStorageProvider(base_path="my-bucket", backend="go").backend_name


def test_backend_fork_safe_parent_uses_bosfs_child_uses_go():
    """fork_safe: the main/parent pid uses bosfs; a (simulated) forked child builds Go."""
    fs = MagicMock(name="BOSFileSystem")
    fake_go = MagicMock(name="GoBackendInstance")
    fake_go.name = "go"
    with (
        patch("multistorageclient.providers.bos._BosGoBackend", return_value=fake_go) as go_ctor,
        patch("bosfs.BOSFileSystem", return_value=fs),
    ):
        provider = BaiduBosStorageProvider(base_path="my-bucket", backend="fork_safe")
        # Parent (creation pid) must NOT start the Go runtime.
        assert provider.backend_name == "bosfs"
        go_ctor.assert_not_called()

        # Simulate a forked child by flipping the recorded main pid, then rebuild.
        provider._main_pid = provider._main_pid + 1
        provider._reset_backend_cache()
        assert provider.backend_name == "go"
        go_ctor.assert_called_once()


# ---------------------------------------------------------------------------
# Regression: BOS delimiter-mode pagination drops page-boundary objects.
#
# BOS's list_objects, when issued with a delimiter ("/"), mis-paginates at the
# max_keys page boundary and silently omits an object that straddles it (the
# key at the start of the second page can be skipped while its neighbours are
# fine). fsspec find()/ls() and the listing-based info() all issue delimiter-mode
# lists, so both recursive listing and HEAD inherited the blind spot. The provider
# must instead use the flat (delimiter="") accumulator for listing and a direct
# object-metadata call for HEAD, neither of which hits the bug.
# ---------------------------------------------------------------------------


def test_bos_recursive_listing_uses_flat_accumulator_not_find():
    """Recursive listing must use the flat delimiter="" accumulator, never find()."""
    provider, fs = _make_provider()
    # Simulate the bug: delimiter-mode find() drops a boundary key; the flat
    # accumulator returns the complete set.
    fs.find.side_effect = AssertionError(
        "find() lists in delimiter mode and drops page-boundary objects; "
        "recursive listing must use _get_object_info_list(delimiter='')"
    )
    fs._get_object_info_list.return_value = [
        {"name": "my-bucket/obj-a", "type": "file", "size": 1},
        {"name": "my-bucket/obj-b", "type": "file", "size": 2},
        {"name": "my-bucket/obj-boundary", "type": "file", "size": 3},  # the boundary victim
        {"name": "my-bucket/obj-c", "type": "file", "size": 4},
    ]

    keys = [m.key for m in provider._list_objects("my-bucket")]

    # The previously-dropped boundary object is present, and find() was not used.
    assert "my-bucket/obj-boundary" in keys
    assert keys == ["my-bucket/obj-a", "my-bucket/obj-b", "my-bucket/obj-boundary", "my-bucket/obj-c"]
    fs._get_object_info_list.assert_called_once()
    assert fs._get_object_info_list.call_args.args[2] == ""  # delimiter is empty (flat)


def test_bos_head_uses_direct_metadata_not_listing_info():
    """HEAD must use a direct object-metadata request, not listing-based info()."""
    provider, fs = _make_provider()
    meta = MagicMock(content_length=123, etag="deadbeef", last_modified=None)
    fs._get_client.return_value.get_object_meta_data.return_value = MagicMock(metadata=meta)

    md = provider._get_object_metadata("my-bucket/obj-boundary")

    assert md.type == "file"
    assert md.content_length == 123
    assert md.etag == "deadbeef"
    # head() goes straight to get_object_meta_data -- it never issues a listing.
    fs._get_client.return_value.get_object_meta_data.assert_called_once_with("my-bucket", "obj-boundary")


def test_bos_head_missing_object_raises_filenotfound():
    """A genuine 404 from the metadata request maps to FileNotFoundError."""
    from baidubce.exception import BceError

    provider, fs = _make_provider()
    err = BceError("The specified key does not exist.")
    err.status_code = 404  # type: ignore[attr-defined]
    fs._get_client.return_value.get_object_meta_data.side_effect = err

    with pytest.raises(FileNotFoundError):
        provider._get_object_metadata("my-bucket/does-not-exist")


def test_bos_info_uses_direct_head_not_listing():
    """fs.info() must resolve real objects via a direct HEAD, never a listing.

    open()/get_range()/get_bytes()/download_file() all call fs.info() for the
    object size, so a listing-based info() (which drops page-boundary objects)
    breaks every read of such an object, not just head().
    """
    from multistorageclient.providers.bos import _BosfsBackend

    fs = MagicMock(name="BOSFileSystem")
    fs._strip_protocol.side_effect = lambda p: p
    fs.info.side_effect = AssertionError("listing-based info() must not be used for real objects")
    meta = MagicMock(content_length=7, etag="abc", last_modified="ts")
    fs._get_client.return_value.get_object_meta_data.return_value = MagicMock(metadata=meta)

    with patch("bosfs.BOSFileSystem", return_value=fs):
        backend = _BosfsBackend("ak", "sk", "tok", "http://bj.bcebos.com")

    info = backend._fs.info("my-bucket/obj-boundary")

    assert info == {
        "name": "my-bucket/obj-boundary",
        "size": 7,
        "type": "file",
        "LastModified": "ts",
        "ETag": "abc",
    }
    backend._fs._get_client.return_value.get_object_meta_data.assert_called_once_with(
        "my-bucket", "obj-boundary"
    )


def test_bos_info_falls_back_to_listing_for_prefix():
    """A 404 (prefix/directory, not a real object) falls back to listing-based info."""
    from baidubce.exception import BceError

    from multistorageclient.providers.bos import _BosfsBackend

    fs = MagicMock(name="BOSFileSystem")
    fs._strip_protocol.side_effect = lambda p: p
    sentinel = {"name": "my-bucket/dir", "type": "directory", "size": 0}
    fs.info.return_value = sentinel  # original listing-based info (captured at install)
    err = BceError("The specified key does not exist.")
    err.status_code = 404  # type: ignore[attr-defined]
    fs._get_client.return_value.get_object_meta_data.side_effect = err

    with patch("bosfs.BOSFileSystem", return_value=fs):
        backend = _BosfsBackend("ak", "sk", "tok", "http://bj.bcebos.com")

    assert backend._fs.info("my-bucket/dir") == sentinel
