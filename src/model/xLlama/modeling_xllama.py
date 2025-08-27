import torch
import torch.nn as nn
import re
from typing import Optional, Union
from transformers import LlamaForCausalLM, LlamaConfig


class XLlamaConfig(LlamaConfig):
    def __init__(self, projector_type: str = "mlp2x_gelu", retriever_hidden_size: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.projector_type = projector_type
        self.retriever_hidden_size = retriever_hidden_size


class Projector(nn.Module):
    def __init__(self, config: XLlamaConfig):
        super().__init__()
        projector_type = config.projector_type
        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.retriever_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            self.projector = nn.Sequential(*modules)
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")

    def forward(self, context_embedding: torch.Tensor) -> torch.Tensor:
        return self.projector(context_embedding)


class XLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: XLlamaConfig):
        super().__init__(config)
        if hasattr(config, "retriever_hidden_size") and config.retriever_hidden_size > 0:
            self.projector = Projector(config)
            self.retriever_hidden_size = config.retriever_hidden_size
        self.post_init()

    def set_xrag_token_id(self, token_id: int) -> None:
        self.xrag_token_id = token_id

    def prepare_inputs_embeds(self, input_ids: torch.Tensor, retrieval_embeds: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.model.embed_tokens(input_ids)
        retrieval_embeds = retrieval_embeds.view(-1, self.retriever_hidden_size)
        num_xrag_tokens = torch.sum(input_ids == self.xrag_token_id).item()
        num_retrieval_embeds = retrieval_embeds.shape[0]
        assert num_xrag_tokens == num_retrieval_embeds, (num_xrag_tokens, num_retrieval_embeds)
        retrieval_embeds = self.projector(retrieval_embeds.to(inputs_embeds.dtype))
        inputs_embeds[input_ids == self.xrag_token_id] = retrieval_embeds
        return inputs_embeds

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        retrieval_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        at_generation_start = False
        if inputs_embeds is not None:
            assert not self.training
            assert retrieval_embeds is None
            at_generation_start = True
        if not at_generation_start and retrieval_embeds is not None:
            inputs_embeds = self.prepare_inputs_embeds(input_ids, retrieval_embeds)
            input_ids = None
            if attention_mask is not None:
                assert inputs_embeds.shape[1] == attention_mask.shape[1]
        return super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        retrieval_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported for generate")
        inputs_embeds = None
        if retrieval_embeds is not None:
            inputs_embeds = self.prepare_inputs_embeds(input_ids, retrieval_embeds)
            input_ids = None
            if attention_mask is not None:
                assert inputs_embeds.shape[1] == attention_mask.shape[1]
            return super().generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
        return super().generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
