# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

# import all the meta_arch, so they will be registered
from .build import REID_HEADS_REGISTRY, build_heads
from .clas_head import ClasHead
from .embedding_head import EmbeddingHead

__all__ = [
    "REID_HEADS_REGISTRY",
    "build_heads",
    "ClasHead",
    "EmbeddingHead",
]
