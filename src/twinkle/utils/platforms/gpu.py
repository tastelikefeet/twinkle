# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Platform


class GPU(Platform):

    @staticmethod
    def visible_device_env():
        return 'CUDA_VISIBLE_DEVICES'

    @staticmethod
    def device_prefix():
        return 'cuda'

    @staticmethod
    def get_local_device(idx, **kwargs) -> str:
        return f'cuda:{idx}'

    @staticmethod
    def device_backend(platform: str = None):
        return 'nccl'