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
    fs.find.return_value = {
        "my-bucket/b.txt": {"name": "my-bucket/b.txt", "type": "file", "size": 2},
        "my-bucket/a.txt": {"name": "my-bucket/a.txt", "type": "file", "size": 1},
        "my-bucket/sub": {"name": "my-bucket/sub", "type": "directory", "size": 0},
    }

    keys = [m.key for m in provider._list_objects("my-bucket")]

    # Sorted, directories excluded when include_directories=False.
    assert keys == ["my-bucket/a.txt", "my-bucket/b.txt"]


def test_bos_list_objects_start_after_and_end_at():
    provider, fs = _make_provider()
    fs.find.return_value = {
        f"my-bucket/{n}": {"name": f"my-bucket/{n}", "type": "file", "size": 1} for n in ("a", "b", "c", "d")
    }

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
