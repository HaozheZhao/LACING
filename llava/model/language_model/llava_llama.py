#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM

                        

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from .hf_model.modeling_llama import MsPoELlamaAttention
from .hf_model.modeling_llama import  LlamaConfig, LlamaModel, LlamaForCausalLM



class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"
    # enable_mda_attention = False
    # enable_ms_poe = False
    # apply_layers = ""
    # head_type = None
    # compress_ratio_min = 1.2
    # compress_ratio_max = 1.8



class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)
        if hasattr(config,"enable_ms_poe") and config.enable_ms_poe:
            num_layers = len(self.layers)
            for layer_idx in range(num_layers):
                if layer_idx in config.apply_layers:
                    self.layers[layer_idx].self_attn = MsPoELlamaAttention(config)

class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config
        # Initialize weights and apply final processing
        self.post_init()
        if hasattr(self.config,"uncond_prob"):
            self.uncond_prob = self.config.uncond_prob
        else:
            self.uncond_prob = 0
        if self.uncond_prob > 0:
            self.token_num = 576
            self.uncond_embedding = nn.Parameter(
                torch.randn(self.token_num, self.config.hidden_size) / self.config.hidden_size ** 0.5)
    def _reset(self):
        for layer_idx in self.config.apply_layers:
            self.model.layers[layer_idx].self_attn.enable_head_metrics = True
            self.model.layers[layer_idx].self_attn.head_order = None
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        if self.uncond_prob > 0 and self.training:
            # print("start_drop out")
            image_features = self.token_drop(image_features)
        return image_features
    
    def token_drop(self, input_embeds):
        """
        Drops labels to enable classifier-free guidance.
        """
        drop_ids = torch.rand(input_embeds.shape[0], device=input_embeds.device) < self.uncond_prob

        # Replace the front part of input_embeds with self.uncond_embedding where drop_ids is True
        input_embeds = torch.where(drop_ids[:, None, None], self.uncond_embedding,
                                                          input_embeds)
        return input_embeds        
    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        images_cd: Optional[torch.FloatTensor] = None,# VCD
        cd_beta: Optional[torch.FloatTensor] = None,
        cd_alpha: Optional[torch.FloatTensor] = None,
        cfg_scale: Optional[torch.FloatTensor] = 1.0,
        cfg_interval: Optional[torch.FloatTensor] = -1,
        use_uncond_embedding: Optional[bool] = False,
        use_dd: Optional[torch.FloatTensor] = None,
        use_dd_unk: Optional[torch.FloatTensor] = None,             
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        return output



    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        cfg_scale = kwargs.get("cfg_scale") if "cfg_scale" in kwargs else 1.0
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                cfg_scale = cfg_scale,

            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)


        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
    
    def prepare_inputs_for_generation_cd( # VCD
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        
        use_dd = kwargs.get("use_dd", False)
        if use_dd:
            inputs_embeds = self.get_model().embed_tokens(input_ids).to(self.device)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images_cd", None),
            }
        )
        return model_inputs


AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
