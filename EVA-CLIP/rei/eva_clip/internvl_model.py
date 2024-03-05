""" huggingface model adapter

Wraps HuggingFace transformers (https://github.com/huggingface/transformers) models for use as a text tower in CLIP model.
"""

import re

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import TensorType
try:
    import transformers
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, AutoConfig, PretrainedConfig
    from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, \
        BaseModelOutputWithPoolingAndCrossAttentions
except ImportError as e:
    transformers = None


    class BaseModelOutput:
        pass


    class PretrainedConfig:
        pass

from .hf_configs import arch_dict

from transformers import LlamaConfig, LlamaForCausalLM

# utils
def _camel2snake(s):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()

# TODO: ?last - for gpt-like models
_POOLERS = {}

def register_pooler(cls):
    """Decorator registering pooler class"""
    _POOLERS[_camel2snake(cls.__name__)] = cls
    return cls


@register_pooler
class MeanPooler(nn.Module):
    """Mean pooling"""
    def forward(self, x:BaseModelOutput, attention_mask:TensorType):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)

@register_pooler
class MaxPooler(nn.Module):
    """Max pooling"""
    def forward(self, x:BaseModelOutput, attention_mask:TensorType):
        masked_output = x.last_hidden_state.masked_fill(attention_mask.unsqueeze(-1), -torch.inf)
        return masked_output.max(1).values

@register_pooler
class ClsPooler(nn.Module):
    """CLS token pooling"""
    def __init__(self, use_pooler_output=True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x:BaseModelOutput, attention_mask:TensorType):
        
        if (self.use_pooler_output and 
            isinstance(x, (BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions)) and
            (x.pooler_output is not None)
            ):
            return x.pooler_output
        
        return x.last_hidden_state[:, self.cls_token_position, :]

class InternVLLLaMA(nn.Module):
    """HuggingFace model adapter"""
    def __init__(
            self, 
            model_name_or_path: str,
            output_dim: int,
            tokenizer_name: str = None,
            config: PretrainedConfig = None,
            pooler_type: str = None,
            proj: str = None,
            pretrained: bool = True,
            masked_language_modeling: bool = False,
            context_length: int = 77,
            attn_mask: bool = True):
        super().__init__()

        """ text encoder of InternVL """
        llama_config = LlamaConfig.from_pretrained(model_name_or_path)
        model = LlamaForCausalLM(llama_config)
        self.transformer = model.model

        self.width = getattr(llama_config, arch_dict["llama"]["config_names"]["width"])
        self.text_projection = nn.Parameter(torch.empty(self.width, output_dim))

    def init_parameters(self):
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.width ** -0.5)

    def forward(self, text:TensorType) -> TensorType:
        text_key_padding_mask = text > 0
        x = self.transformer(input_ids=text, attention_mask=text_key_padding_mask).last_hidden_state
        x = x[torch.arange(x.shape[0]), text_key_padding_mask.sum(1) - 1]
        x = x @ self.text_projection
        return x

    def lock(self, unlocked_layers:int=0, freeze_layer_norm:bool=True):
        if not unlocked_layers: # full freezing
            self.text_projection.requires_grad = False
            for n, p in self.transformer.named_parameters():
                # p.requires_grad = (not freeze_layer_norm) if ("LayerNorm" in n.split(".") or "LlamaRMSNorm" in n.split(".")) else False
                p.requires_grad = False
            return

        encoder = self.transformer.encoder if hasattr(self.transformer, 'encoder') else self.transformer
        layer_list = getattr(encoder, arch_dict[self.config.model_type]["config_names"]["layer_attr"])
        print(f"Unlocking {unlocked_layers}/{len(layer_list) + 1} layers of hf model")
        embeddings = getattr(
            self.transformer, arch_dict[self.config.model_type]["config_names"]["token_embeddings_attr"])
        modules = [embeddings, *layer_list][:-unlocked_layers]
        # freeze layers
        for module in modules:
            for n, p in module.named_parameters():
                p.requires_grad = (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False


    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.gradient_checkpointing_enable()

    def get_num_layers(self):
        encoder = self.transformer.encoder if hasattr(self.transformer, 'encoder') else self.transformer
        layer_list = getattr(encoder, arch_dict["bert"]["config_names"]["layer_attr"])
        return len(layer_list)