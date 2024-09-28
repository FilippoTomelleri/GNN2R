from torch import FloatTensor, LongTensor
import torch.nn as nn

class GeneralSuperclass(nn.Module):

    def rel_enc(self):
        """
        Return the encoding of the relations
        size: (num_rels, hid_dim)
        """
        raise NotImplementedError("Method rel_enc has to be implemented in subclass")
    

    def forward(self, que_embeds: FloatTensor, r: FloatTensor, num_subg_ents: int,
            edge_index: LongTensor, edge_attr: LongTensor, loc_tops: LongTensor):
        """
        Return should be a tuple of two tensors:
        - x: the encoding of te entities in the subgraph
            size = (num_ents_in_subg, hid_dim)
        - fin_que_embed: the encoding of the question
            size = (1, hid_dim)
        """
        raise NotImplementedError("Method forward has to be implemented in subclass")