from torch import FloatTensor, LongTensor
from models.generalSuperclass import GeneralSuperclass
from torch_geometric.nn.conv import RGCNConv
import torch


class RGCNBaseline(GeneralSuperclass):

    def __init__(self, in_channes, out_channels, num_relations, num_bases=None, num_layers=None, **kwargs):
        super(RGCNBaseline, self).__init__()
        self.hid_dim = out_channels
        self.__model = RGCNConv(in_channels=in_channes, out_channels=out_channels, num_relations=num_relations, **kwargs)


    def forward(self, que_embeds: FloatTensor, r: FloatTensor, num_subg_ents: int,
            edge_index: LongTensor, edge_attr: LongTensor, loc_tops: LongTensor):
        
        x = torch.randn(num_subg_ents, self.hid_dim).cuda()
        x, self.__model(x=x, edge_index=edge_index, edge_type=edge_attr)