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

import io
import logging
import os
from collections.abc import Callable, Iterator
from datetime import timezone
from typing import IO, Any, Optional, Union

from dateutil.parser import parse as dateutil_parse

from ..telemetry import Telemetry
from ..types import (
    AWARE_DATETIME_MIN,
    CredentialsProvider,
    ObjectMetadata,
    Range,
    SymlinkHandling,
)
from ..utils import split_path
from .base import BaseStorageProvider

logger = logging.getLogger(__name__)

PROVIDER = "bos"

# Baidu Object Storage (BOS) endpoints have the form ``<region>.bcebos.com``
# (e.g. ``bj.bcebos.com``). The default region matches BOS's default (Beijing).
DEFAULT_REGION = "bj"

# Default BNS (Baidu Naming Service) group for the Go backend. Resolves to a
# load-balanced pool of Beijing BOS access nodes via ``get_instance_by_service``.
DEFAULT_BNS = "group.bos-C-nginx-bjbl.BCE.all"

# Backend selection:
#   "auto"      prefer the Go client (bos_tool), fall back to bosfs.
#   "go"        force the Go client.
#   "bosfs"     force the fsspec client.
#   "fork_safe" the Go runtime is not fork-safe. In this mode the *parent/main*
#               process uses bosfs (never starts the Go runtime), while forked
#               children (e.g. torch DataLoader workers) each lazily build their
#               own fresh Go client. This lets Energon's forked readers use the
#               fast Go/BNS path without inheriting a broken Go runtime.
_BACKEND_AUTO = "auto"
_BACKEND_GO = "go"
_BACKEND_BOSFS = "bosfs"
_BACKEND_FORK_SAFE = "fork_safe"


def _default_endpoint_url(region_name: str) -> str:
    return f"http://{region_name}.bcebos.com"


def _looks_like_not_found(message: str) -> bool:
    m = message.lower()
    return any(token in m for token in ("nosuchkey", "not found", "notfound", "404", "no such key"))


# ============================================================================
# Backends. Each exposes the same bucket/key oriented data + metadata plane so
# the provider hooks are backend-agnostic. list_page returns object keys
# *relative to the bucket* (no bucket prefix); the provider re-prepends it.
# ============================================================================


class _BosGoBackend:
    """Backend backed by the Go client (``bos_tool.GoBosClient``, via libgobos.so)."""

    name = _BACKEND_GO

    def __init__(self, access_key, secret_key, token, bns, endpoint, lib_path):
        from bos_tool import GoBosClient  # imported lazily so bosfs-only setups don't need it

        self._client = GoBosClient(
            ak=access_key or "",
            sk=secret_key or "",
            bns=bns or "",
            endpoint=endpoint,
            lib_path=lib_path,
        )

    def get_bytes(self, bucket, key):
        return self._client.get_object_as_string(bucket, key)

    def get_range(self, bucket, key, offset, size):
        return self._client.get_object_range(bucket, key, offset, size)

    def put_bytes(self, bucket, key, data):
        self._client.put_object_from_string(bucket, key, data)

    def upload_file(self, bucket, key, local_path):
        self._client.put_object_from_file(bucket, key, local_path)

    def download_file(self, bucket, key, local_path):
        self._client.get_object_to_file(bucket, key, local_path)

    def head(self, bucket, key):
        from bos_tool import GoBosError

        try:
            meta = self._client.head_object(bucket, key)
        except GoBosError as error:
            if _looks_like_not_found(str(error)):
                raise FileNotFoundError(f"bos://{bucket}/{key}") from error
            raise RuntimeError(f"HEAD failed for bos://{bucket}/{key}: {error}") from error
        return {
            "type": "file",  # GetObjectMeta only succeeds for real objects, never prefixes
            "size": int(meta.get("size") or 0),
            "last_modified": meta.get("last_modified"),
            "etag": meta.get("etag"),
        }

    def list_page(self, bucket, prefix, delimiter, marker, max_keys):
        page = self._client.list_objects(bucket, prefix=prefix, delimiter=delimiter, marker=marker, max_keys=max_keys)
        return page.get("objects", []), bool(page.get("is_truncated")), page.get("next_marker") or ""

    def delete(self, bucket, key):
        from bos_tool import GoBosError

        try:
            self._client.delete_object(bucket, key)
        except GoBosError as error:
            if _looks_like_not_found(str(error)):
                raise FileNotFoundError(f"bos://{bucket}/{key}") from error
            raise

    def delete_many(self, bucket, keys):
        if keys:
            self._client.delete_objects(bucket, keys)

    def copy(self, dst_bucket, dst_key, src_bucket, src_key):
        self._client.copy_object(dst_bucket, dst_key, src_bucket, src_key)

    def close(self):
        try:
            self._client.close()
        except Exception:
            pass


class _BosfsBackend:
    """Backend backed by ``bosfs`` (fsspec + BCE Python SDK). Fallback path."""

    name = _BACKEND_BOSFS

    def __init__(self, access_key, secret_key, token, endpoint, **kwargs):
        import bosfs

        self._fs = bosfs.BOSFileSystem(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            sts_token=token,
            **kwargs,
        )

    @staticmethod
    def _full(bucket, key):
        return f"{bucket}/{key}" if key else bucket

    def get_bytes(self, bucket, key):
        return self._fs.cat_file(self._full(bucket, key))

    def get_range(self, bucket, key, offset, size):
        # Use the file interface: bosfs cat_file mishandles start==0 and uses an
        # inclusive range, so open+seek+read is the reliable way to get exactly
        # ``size`` bytes.
        with self._fs.open(self._full(bucket, key), "rb") as fp:
            fp.seek(offset)
            return fp.read(size)

    def put_bytes(self, bucket, key, data):
        self._fs.pipe_file(self._full(bucket, key), data)

    def upload_file(self, bucket, key, local_path):
        self._fs.put_file(local_path, self._full(bucket, key))

    def download_file(self, bucket, key, local_path):
        self._fs.get_file(self._full(bucket, key), local_path)

    def head(self, bucket, key):
        info = self._fs.info(self._full(bucket, key))  # raises FileNotFoundError if missing
        return {
            # bosfs returns type="directory" for prefixes; carry it through so the
            # provider doesn't misreport a directory as a file.
            "type": "directory" if info.get("type") == "directory" else "file",
            "size": int(info.get("size") or 0),
            "last_modified": info.get("LastModified"),
            "etag": info.get("ETag"),
        }

    def list_page(self, bucket, prefix, delimiter, marker, max_keys):
        # bosfs is not marker-paginated; return everything in one page and let the
        # provider apply start_after/end_at filtering.
        full = self._full(bucket, prefix)
        try:
            if delimiter:
                infos = list(self._fs.ls(full, detail=True))
            else:
                infos = list(self._fs.find(full, detail=True).values())
        except FileNotFoundError:
            return [], False, ""

        objects = []
        bucket_prefix = f"{bucket}/"
        for info in infos:
            name = str(info["name"])
            rel = name[len(bucket_prefix) :] if name.startswith(bucket_prefix) else name.lstrip("/")
            is_dir = info.get("type") == "directory"
            objects.append(
                {
                    "key": rel,
                    "size": 0 if is_dir else int(info.get("size") or 0),
                    "last_modified": info.get("LastModified"),
                    "etag": info.get("ETag"),
                    "is_prefix": is_dir,
                }
            )
        # BOS returns keys lexicographically; bosfs find/ls does not guarantee order,
        # so sort to match the object-store listing contract.
        objects.sort(key=lambda o: o["key"])
        return objects, False, ""

    def delete(self, bucket, key):
        self._fs.rm_file(self._full(bucket, key))

    def delete_many(self, bucket, keys):
        for key in keys:
            try:
                self._fs.rm_file(self._full(bucket, key))
            except FileNotFoundError:
                continue

    def copy(self, dst_bucket, dst_key, src_bucket, src_key):
        self._fs.cp_file(self._full(src_bucket, src_key), self._full(dst_bucket, dst_key))

    def close(self):
        pass


class BaiduBosStorageProvider(BaseStorageProvider):
    """
    A concrete implementation of the :py:class:`multistorageclient.types.StorageProvider` for interacting with
    Baidu Object Storage (BOS).

    By default it uses the Go-backed client (``bos_tool``/``libgobos.so``), which talks to BOS through a
    BNS-resolved pool of access nodes for higher throughput. When the Go client is unavailable it falls back
    to the ``bosfs`` (fsspec) client.
    """

    def __init__(
        self,
        region_name: str = "",
        endpoint_url: str = "",
        base_path: str = "",
        credentials_provider: Optional[CredentialsProvider] = None,
        backend: str = _BACKEND_AUTO,
        bns: str = DEFAULT_BNS,
        go_lib_path: Optional[str] = None,
        config_dict: Optional[dict[str, Any]] = None,
        telemetry_provider: Optional[Callable[[], Telemetry]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the :py:class:`BaiduBosStorageProvider`.

        :param region_name: The BOS region code (e.g. ``bj``, ``gz``). Defaults to ``bj``.
        :param endpoint_url: The BOS endpoint. Defaults to ``http://<region_name>.bcebos.com``.
        :param base_path: The bucket (optionally with a key prefix) that scopes all operations.
        :param credentials_provider: The provider to retrieve BOS credentials (access key / secret key / token).
        :param backend: ``"auto"`` (default; prefer Go, fall back to bosfs), ``"go"`` (force Go client), or
            ``"bosfs"`` (force fsspec client).
        :param bns: BNS group for the Go client. Defaults to a Beijing BOS nginx pool.
        :param go_lib_path: Optional explicit path to ``libgobos.so`` for the Go client.
        :param config_dict: Resolved MSC config.
        :param telemetry_provider: A function that provides a telemetry instance.
        :param kwargs: Additional keyword arguments forwarded to :py:class:`bosfs.BOSFileSystem` (bosfs backend only).
        """
        super().__init__(
            base_path=base_path,
            provider_name=PROVIDER,
            config_dict=config_dict,
            telemetry_provider=telemetry_provider,
        )

        self._region_name = region_name or DEFAULT_REGION
        self._endpoint_url = endpoint_url or _default_endpoint_url(self._region_name)
        self._credentials_provider = credentials_provider
        self._bns = bns
        self._go_lib_path = go_lib_path
        self._backend_mode = (backend or _BACKEND_AUTO).lower()

        access_key = secret_key = token = None
        if credentials_provider is not None:
            credentials_provider.refresh_credentials()
            credentials = credentials_provider.get_credentials()
            access_key, secret_key, token = credentials.access_key, credentials.secret_key, credentials.token
        # Fall back to BCE environment variables when no credentials provider is
        # configured (e.g. implicit bos:// profiles). bosfs does this internally;
        # the Go client does not, so do it here for both backends.
        self._access_key = access_key or os.getenv("BCE_ACCESS_KEY_ID")
        self._secret_key = secret_key or os.getenv("BCE_SECRET_ACCESS_KEY")
        self._token = token
        self._bosfs_kwargs = kwargs

        # Backends are built lazily and cached per-PID so that a fork (e.g. torch
        # DataLoader workers) never reuses a backend built in another process.
        self._main_pid = os.getpid()
        self._cached_backend = None
        self._cached_backend_pid: Optional[int] = None
        # After a fork, force the child to rebuild its backend on next use.
        try:
            os.register_at_fork(after_in_child=self._reset_backend_cache)
        except Exception:  # noqa: BLE001 — register_at_fork is best-effort
            pass

    def _reset_backend_cache(self) -> None:
        self._cached_backend = None
        self._cached_backend_pid = None

    def _build_go_backend(self):
        return _BosGoBackend(
            self._access_key, self._secret_key, self._token, self._bns, self._endpoint_url, self._go_lib_path
        )

    def _build_bosfs_backend(self):
        return _BosfsBackend(self._access_key, self._secret_key, self._token, self._endpoint_url, **self._bosfs_kwargs)

    def _build_backend_for_pid(self, pid: int):
        mode = self._backend_mode
        if mode == _BACKEND_FORK_SAFE:
            # Parent/main process must NOT start the Go runtime (so forked children
            # don't inherit a broken one); it uses bosfs. Children use fresh Go.
            if pid == self._main_pid:
                return self._build_bosfs_backend()
            return self._build_go_backend()
        if mode in (_BACKEND_AUTO, _BACKEND_GO):
            try:
                return self._build_go_backend()
            except Exception as error:  # noqa: BLE001 — fall back unless Go was explicitly required
                if mode == _BACKEND_GO:
                    raise
                logger.warning("BOS Go backend unavailable (%s); falling back to bosfs.", error)
        return self._build_bosfs_backend()

    def _get_backend(self):
        pid = os.getpid()
        if self._cached_backend is not None and self._cached_backend_pid == pid:
            return self._cached_backend
        backend = self._build_backend_for_pid(pid)
        self._cached_backend = backend
        self._cached_backend_pid = pid
        logger.debug(
            "BOS provider pid=%s using '%s' backend (mode=%s, endpoint=%s)",
            pid,
            backend.name,
            self._backend_mode,
            self._endpoint_url,
        )
        return backend

    @property
    def backend_name(self) -> str:
        """Which backend is active for the current process: ``"go"`` or ``"bosfs"``."""
        return self._get_backend().name

    @staticmethod
    def _parse_last_modified(value: Any):
        if value is None:
            return AWARE_DATETIME_MIN
        try:
            parsed = value if hasattr(value, "tzinfo") else dateutil_parse(str(value))
        except (ValueError, OverflowError, TypeError):
            return AWARE_DATETIME_MIN
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    def _is_dir(self, path: str) -> bool:
        bucket, key = split_path(path)
        prefix = key.rstrip("/") + "/" if key else ""
        objects, _, _ = self._get_backend().list_page(bucket, prefix, "/", "", 1)
        return len(objects) > 0

    def _get_object(self, path: str, byte_range: Optional[Range] = None) -> bytes:
        bucket, key = split_path(path)
        if byte_range is not None:
            return self._get_backend().get_range(bucket, key, byte_range.offset, byte_range.size)
        return self._get_backend().get_bytes(bucket, key)

    def _put_object(
        self,
        path: str,
        body: bytes,
        if_match: Optional[str] = None,
        if_none_match: Optional[str] = None,
        attributes: Optional[dict[str, str]] = None,
    ) -> int:
        if if_match is not None or if_none_match is not None:
            raise NotImplementedError("BOS provider does not support conditional writes (if_match/if_none_match).")
        bucket, key = split_path(path)
        self._get_backend().put_bytes(bucket, key, body)
        return len(body)

    def _copy_object(self, src_path: str, dest_path: str) -> int:
        src_metadata = self._get_object_metadata(src_path)
        src_bucket, src_key = split_path(src_path)
        dst_bucket, dst_key = split_path(dest_path)
        self._get_backend().copy(dst_bucket, dst_key, src_bucket, src_key)
        return src_metadata.content_length

    def _delete_object(self, path: str, if_match: Optional[str] = None) -> None:
        if if_match is not None:
            raise NotImplementedError("BOS provider does not support conditional deletion (if_match).")
        bucket, key = split_path(path)
        self._get_backend().delete(bucket, key)

    def _delete_objects(self, paths: list[str]) -> None:
        # Group by bucket and use the backend's bulk delete. Missing objects are ignored.
        by_bucket: dict[str, list[str]] = {}
        for path in paths:
            bucket, key = split_path(path)
            by_bucket.setdefault(bucket, []).append(key)
        for bucket, keys in by_bucket.items():
            self._get_backend().delete_many(bucket, keys)

    def _make_symlink(self, path: str, target: str) -> None:
        raise NotImplementedError("BOS provider does not support symbolic links.")

    def _get_object_metadata(self, path: str, strict: bool = True) -> ObjectMetadata:
        bucket, key = split_path(path)
        try:
            meta = self._get_backend().head(bucket, key)
        except FileNotFoundError:
            if not strict and self._is_dir(path):
                return ObjectMetadata(
                    key=path.strip("/"),
                    type="directory",
                    content_length=0,
                    last_modified=AWARE_DATETIME_MIN,
                )
            raise
        if meta.get("type") == "directory":
            return ObjectMetadata(
                key=path.strip("/"),
                type="directory",
                content_length=0,
                last_modified=self._parse_last_modified(meta.get("last_modified")),
            )
        return ObjectMetadata(
            key=path.strip("/"),
            type="file",
            content_length=meta["size"],
            last_modified=self._parse_last_modified(meta.get("last_modified")),
            etag=meta.get("etag"),
        )

    def _list_objects(
        self,
        path: str,
        start_after: Optional[str] = None,
        end_at: Optional[str] = None,
        include_directories: bool = False,
        symlink_handling: SymlinkHandling = SymlinkHandling.FOLLOW,
    ) -> Iterator[ObjectMetadata]:
        bucket, prefix = split_path(path)
        # start_after / end_at arrive as full paths; compare on bucket-relative keys.
        if start_after:
            _, start_after = split_path(start_after)
        if end_at:
            _, end_at = split_path(end_at)

        delimiter = "/" if include_directories else ""
        marker = start_after or ""  # BOS marker is exclusive, matching start_after semantics

        while True:
            objects, is_truncated, next_marker = self._get_backend().list_page(bucket, prefix, delimiter, marker, 1000)
            for obj in objects:
                rel_key = str(obj["key"])
                if obj.get("is_prefix"):
                    if not include_directories:
                        continue
                    dir_key = rel_key.rstrip("/")
                    if (start_after is not None and dir_key <= start_after) or (
                        end_at is not None and dir_key > end_at
                    ):
                        if end_at is not None and dir_key > end_at:
                            return
                        continue
                    yield ObjectMetadata(
                        key=os.path.join(bucket, dir_key),
                        type="directory",
                        content_length=0,
                        last_modified=AWARE_DATETIME_MIN,
                    )
                    continue

                if start_after is not None and rel_key <= start_after:
                    continue
                if end_at is not None and rel_key > end_at:
                    return
                yield ObjectMetadata(
                    key=os.path.join(bucket, rel_key),
                    type="file",
                    content_length=int(obj.get("size") or 0),
                    last_modified=self._parse_last_modified(obj.get("last_modified")),
                    etag=obj.get("etag"),
                )

            if not is_truncated or not next_marker:
                return
            marker = next_marker

    def _upload_file(self, remote_path: str, f: Union[str, IO], attributes: Optional[dict[str, str]] = None) -> int:
        bucket, key = split_path(remote_path)
        if isinstance(f, str):
            self._get_backend().upload_file(bucket, key, f)
            return os.path.getsize(f)

        f.seek(0, io.SEEK_END)
        file_size = f.tell()
        f.seek(0)
        body = f.read()
        if isinstance(body, str):
            body = body.encode("utf-8")
        self._get_backend().put_bytes(bucket, key, body)
        return file_size

    def _download_file(self, remote_path: str, f: Union[str, IO], metadata: Optional[ObjectMetadata] = None) -> int:
        bucket, key = split_path(remote_path)
        if isinstance(f, str):
            os.makedirs(os.path.dirname(f) or ".", exist_ok=True)
            self._get_backend().download_file(bucket, key, f)
            return os.path.getsize(f)

        data = self._get_backend().get_bytes(bucket, key)
        if isinstance(f, io.StringIO):
            f.write(data.decode("utf-8"))
        else:
            f.write(data)
        return len(data)
