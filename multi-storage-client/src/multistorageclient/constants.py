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

# Threshold for loading a whole object into memory vs. streaming it via ranged reads.
# Objects smaller than this are read whole in a single GET (cheap, and avoids a storm
# of tiny range GETs for small files like .tar.idx); objects at least this large are
# read through RemoteFileReader (ranged get_object) + a read-ahead buffer, so large
# webdataset tar shards stream instead of buffering entirely in RAM. Lowered from the
# upstream 512MB because this deployment reads multi-hundred-MB / multi-GB tar shards
# over BOS, which must not be whole-buffered per DataLoader worker. Set to 5M so small
# .idx/metadata objects are still whole-loaded in one GET; a sweep found 1M vs 5M within
# noise for cook throughput (big tars always stream regardless). See outputs/mll_sweep.
MEMORY_LOAD_LIMIT = 5 * 1024 * 1024

# Default timeout values (in seconds) for storage provider connections
DEFAULT_CONNECT_TIMEOUT = 60
DEFAULT_READ_TIMEOUT = 60
DEFAULT_MAX_POOL_CONNECTIONS = 128

# Default host and port for the MSC Explorer application
DEFAULT_EXPLORER_HOST = "127.0.0.1"
DEFAULT_EXPLORER_PORT = 8888

# Default batch size for sync operations
DEFAULT_SYNC_BATCH_SIZE = 32
