from torch import FloatTensor, LongTensor
from models.generalSuperclass import GeneralSuperclass
from torch_geometric.nn.conv import RGCNConv
import torch


class RGCNBaseline(GeneralSuperclass):

    def __init__(self, rel2embeds: dict, in_channes, out_channels, num_relations, num_bases=None, num_layers=None, **kwargs):
        super(RGCNBaseline, self).__init__()
        self.hid_dim = out_channels

        self.rel_embeds = [rel2embeds[r_id].cuda() for r_id in range(len(rel2embeds))]

        self.__model = RGCNConv(in_channels=in_channes, out_channels=out_channels, num_relations=num_relations, **kwargs)

    def rel_enc(self):
        return self.rel_embeds

    def forward(self, que_embeds: FloatTensor, r: FloatTensor, num_subg_ents: int,
            edge_index: LongTensor, edge_attr: LongTensor, loc_tops: LongTensor):
        
        x = torch.randn(num_subg_ents, self.hid_dim).cuda()
        x, self.__model(x=x, edge_index=edge_index, edge_type=edge_attr)

        return (x,  # size: (num_ents_in_subg, hid_dim)
                que_embeds)