# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil

from magi_compiler.config import get_compile_config


class CleanupCacheContext:
    """Context manager for cleaning cache before and after execution"""

    def __enter__(self):
        shutil.rmtree(get_compile_config().cache_root_dir, ignore_errors=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(get_compile_config().cache_root_dir, ignore_errors=True)


def enable_remote_debug():
    import os

    import debugpy

    ENABLE_MAGI_REMOTE_DEBUG = os.environ.get("ENABLE_MAGI_REMOTE_DEBUG", "false").lower()
    if ENABLE_MAGI_REMOTE_DEBUG == "false":
        return

    debug_ranks = []
    if ENABLE_MAGI_REMOTE_DEBUG == "true":
        debug_ranks = [0]
    elif ENABLE_MAGI_REMOTE_DEBUG == "all":
        debug_ranks = [i for i in range(1)]
    else:
        debug_ranks = [int(i) for i in ENABLE_MAGI_REMOTE_DEBUG.split(",")]

    rank = 0
    if rank in debug_ranks:
        debug_port = 5678 + int(rank)
        print(f"[rank {rank}] Starting remote debug on port {debug_port}")
        debugpy.listen(("0.0.0.0", debug_port))
        debugpy.wait_for_client()
        print(f"[rank {rank}] Remote debug attached")
