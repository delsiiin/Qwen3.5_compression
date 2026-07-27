from transformers.models.llama import modeling_llama
from transformers.models.mistral import modeling_mistral
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3_moe import modeling_qwen3_moe
from .modeling import (
    Llama_Attention_init,
    Llama_Attention_forward,
    Llama_CausalLM_forward,
    Mistral_Attention_init,
    Mistral_Attention_forward,
    Qwen2_Attention_init,
    Qwen2_Attention_forward,
    Qwen3_Attention_init,
    Qwen3_Attention_forward,
    Qwen3_CausalLM_forward,
    Qwen3Moe_Attention_init,
    Qwen3Moe_Attention_forward,
    Qwen3Moe_CausalLM_forward,
)

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from .flash_attn.flash_attention import flash_attention_forward

def replace_llama(compression_config):
    def init_wrapper(self, config, layer_idx):
        Llama_Attention_init(self, config, layer_idx, compression_config)

    modeling_llama.LlamaAttention.__init__ = init_wrapper
    modeling_llama.LlamaAttention.forward = Llama_Attention_forward
    modeling_llama.LlamaForCausalLM.forward = (
        Llama_CausalLM_forward
    )

    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward

def replace_mistral(compression_config):
    def init_wrapper(self, config, layer_idx):
        Mistral_Attention_init(self, config, layer_idx, compression_config)

    modeling_mistral.MistralAttention.__init__ = init_wrapper
    modeling_mistral.MistralAttention.forward = Mistral_Attention_forward
    modeling_mistral.MistralForCausalLM.forward = (
        Llama_CausalLM_forward
    )

    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward

def replace_qwen2_5(compression_config):
    def init_wrapper(self, config, layer_idx):
        Qwen2_Attention_init(self, config, layer_idx, compression_config)

    modeling_qwen2.Qwen2Attention.__init__ = init_wrapper
    modeling_qwen2.Qwen2Attention.forward = Qwen2_Attention_forward
    modeling_qwen2.Qwen2ForCausalLM.forward = (
        Llama_CausalLM_forward
    )

    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward

def replace_qwen3(compression_config):
    def init_wrapper(self, config, layer_idx):
        Qwen3_Attention_init(self, config, layer_idx, compression_config)

    modeling_qwen3.Qwen3Attention.__init__ = init_wrapper
    modeling_qwen3.Qwen3Attention.forward = Qwen3_Attention_forward
    modeling_qwen3.Qwen3ForCausalLM.forward = (
        Qwen3_CausalLM_forward
    )

    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward

def replace_qwen3moe(compression_config):
    def init_wrapper(self, config, layer_idx):
        Qwen3Moe_Attention_init(self, config, layer_idx, compression_config)

    modeling_qwen3_moe.Qwen3MoeAttention.__init__ = init_wrapper
    modeling_qwen3_moe.Qwen3MoeAttention.forward = Qwen3Moe_Attention_forward
    modeling_qwen3_moe.Qwen3MoeForCausalLM.forward = (
        Qwen3Moe_CausalLM_forward
    )

    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
