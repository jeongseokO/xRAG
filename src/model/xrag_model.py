import torch
from transformers import AutoTokenizer
from .SFR import SFR
from .xLlama import XLlamaForCausalLM, XLlamaConfig
from src.language_modeling.utils import XRAG_TOKEN


class XRAGModel(torch.nn.Module):
    def __init__(self, model_name: str, compressor_name: str = None):
        super().__init__()
        compressor_name = compressor_name or f"{model_name}/compressor"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.compressor_tokenizer = AutoTokenizer.from_pretrained(compressor_name)
        self.compressor = SFR.from_pretrained(compressor_name)
        base_config = XLlamaConfig.from_pretrained(model_name)
        self.llm = XLlamaForCausalLM.from_pretrained(model_name, config=base_config)
        # freeze base model; only the compressor will be updated during training
        self.llm.requires_grad_(False)
        self.xrag_token_id = self.tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
        if self.xrag_token_id == self.tokenizer.unk_token_id:
            self.tokenizer.add_special_tokens({"additional_special_tokens": [XRAG_TOKEN]})
            self.llm.resize_token_embeddings(len(self.tokenizer))
            self.xrag_token_id = self.tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
        self.llm.set_xrag_token_id(self.xrag_token_id)

    def forward(self, query_ids, query_mask, doc_ids, doc_mask, labels):
        retrieval = self.compressor.get_doc_embedding(doc_ids, doc_mask)
        outputs = self.llm(
            input_ids=query_ids,
            attention_mask=query_mask,
            retrieval_embeds=retrieval,
            labels=labels,
        )
        return outputs

    @torch.no_grad()
    def generate(self, query: str, document: str, **kwargs) -> str:
        query_tokens = self.tokenizer(query + XRAG_TOKEN, return_tensors="pt")
        doc_tokens = self.compressor_tokenizer(document, return_tensors="pt", truncation=True)
        retrieval = self.compressor.get_doc_embedding(doc_tokens["input_ids"], doc_tokens["attention_mask"])
        output_ids = self.llm.generate(
            input_ids=query_tokens["input_ids"],
            attention_mask=query_tokens["attention_mask"],
            retrieval_embeds=retrieval,
            **kwargs,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
