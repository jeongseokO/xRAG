import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class SFR(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "SFR":
        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(model)

    def get_embed_dim(self) -> int:
        return self.model.config.hidden_size

    def get_embed_length(self) -> int:
        return 1

    def get_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)
        return embeddings

    def get_doc_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.get_embedding(input_ids, attention_mask)

    def get_query_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.get_embedding(input_ids, attention_mask)
