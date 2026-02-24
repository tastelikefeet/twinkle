from .base import Platform
from .npu import NPU, ensure_npu_backend, ensure_hccl_socket_env
from .mps import MPS, is_mps_available
from .gpu import GPU
