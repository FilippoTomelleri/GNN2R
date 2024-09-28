import torch
import torch.nn as nn
from torch_scatter import scatter
from torch import LongTensor, FloatTensor
from torch.nn.utils.rnn import pack_sequence
from transformers import AutoTokenizer, AutoModel



# ----------------------------
# expression matching model

class ExpMatchModel(nn.Module):
    def __init__(self, lm_name: str, norm: int, reinit_n: int):
        super(ExpMatchModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        self.model = AutoModel.from_pretrained(lm_name)
        self.norm = norm

        for n in range(reinit_n):
            self.model.transformer.layer[-(n+1)].apply(self._init_params)

    def que_embed(self, que: str):
        que_input = self.tokenizer(que, padding=True, truncation=True, return_tensors='pt').to(torch.cuda.current_device())
        que_embed = self.model(**que_input, return_dict=True).last_hidden_state
        que_mask_expanded = que_input['attention_mask'].unsqueeze(-1).expand(que_embed.size()).float()
        que_embed = torch.sum(que_embed * que_mask_expanded, 1) / torch.clamp(que_mask_expanded.sum(1), min=1e-16)
        que_embed = nn.functional.normalize(que_embed, p=self.norm, dim=1)
        return que_embed  # size: (1, 768)

    def forward(self, subg_exps: list):
        subg_exps_input = self.tokenizer(subg_exps, padding=True, truncation=True, return_tensors='pt').to(torch.cuda.current_device())
        subg_exps_embeds = self.model(**subg_exps_input, return_dict=True).last_hidden_state
        subg_exps_mask_expanded = subg_exps_input['attention_mask'].unsqueeze(-1).expand(
            subg_exps_embeds.size()).float()
        subg_exps_embeds = torch.sum(subg_exps_embeds * subg_exps_mask_expanded, 1) / torch.clamp(
            subg_exps_mask_expanded.sum(1), min=1e-16)
        subg_exps_embeds = nn.functional.normalize(subg_exps_embeds, p=self.norm, dim=1)
        return subg_exps_embeds  # size: (num_subg_exps, 768)

    def _init_params(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()