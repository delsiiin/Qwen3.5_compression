from ..utils import cal_similarity, compute_attention_scores

from .snapkv import SnapKV
from .snapkv_ada import SnapKV as SnapKVAda
from .snapkv_ada_online_head_cluster import SnapKVAdaOnlineHeadCluster
from .snapkv_neighbor_shared import SnapKVNeighborShared
from .snapkv_hidden_mix import SnapKVHiddenMix
from .snapkv_hidden_mix_ada import SnapKVHiddenMix as SnapKVAdaHiddenMixAda
from .snapkv_hidden_mix_layer import SnapKVHiddenMix as SnapKVHiddenMixLayer
from .snapkv_hidden_mix_neighbor import SnapKVHiddenMix as SnapKVHiddenMixNeighbor
from .snapkv_hidden_mix_no_cos import SnapKVHiddenMix as SnapKVHiddenMixNoCos
from .snapkv_hidden_mix_no_cos_head_cluster import SnapKVHiddenMix as SnapKVHiddenMixNoCosHeadCluster
from .snapkv_hidden_mix_random import SnapKVHiddenMix as SnapKVHiddenMixRandom
from .snapkv_hidden_wo_mix import SnapKVHiddenMix as SnapKVHiddenWoMix
from .streamingllm import StreamingLLM
from .h2o import H2O
from .criticalkv import CriticalKV
from .defensivekv import DefensiveKV
from .laprox import LaProx

__all__ = [
    "SnapKV",
    "SnapKVAda",
    "SnapKVAdaHeadCluster",
    "SnapKVAdaOnlineHeadCluster",
    "SnapKVNeighborShared",
    "SnapKVHiddenMix",
    "SnapKVAdaHiddenMixAda",
    "SnapKVHiddenMixLayer",
    "SnapKVHiddenMixNeighbor",
    "SnapKVHiddenMixNoCos",
    "SnapKVHiddenMixNoCosHeadCluster",
    "SnapKVHiddenMixRandom",
    "SnapKVHiddenWoMix",
    "StreamingLLM",
    "H2O",
    "CriticalKV",
    "DefensiveKV",
    "LaProx",
]
