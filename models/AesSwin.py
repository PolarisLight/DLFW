import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Swinv2Model


class GraphConvLayer(nn.Module):
    def __init__(self, in_feature=768,out_feature=768, bias=False, resnet=False, use_BN=False):
        super(GraphConvLayer, self).__init__()

        self.mapping1 = nn.Linear(in_feature, out_feature, bias=bias)
        self.mapping2 = nn.Linear(in_feature, out_feature, bias=bias)

        self.GCN_W = nn.Parameter(torch.FloatTensor(out_feature, out_feature))
        # self.GCN_B = nn.Parameter(torch.FloatTensor(dim_feature))
        self.relu = nn.GELU()
        self.initialize()
        self.resnet = resnet
        self.use_BN = use_BN

        self.bn = nn.BatchNorm1d(out_feature)

    def initialize(self):
        nn.init.xavier_uniform_(self.GCN_W)
        # nn.init.xavier_uniform_(self.GCN_B)

    def forward(self, x):
        x_in = x
        m1 = self.mapping1(x)
        # m2 = self.mapping2(x)
        x_out = m1

        similarity = torch.matmul(x, x.transpose(1, 2))
        A_sim = F.softmax(similarity, dim=2)

        
        A = A_sim

        x = torch.matmul(A, x_out)
        x = torch.matmul(x, self.GCN_W)
        if self.use_BN:
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)
        # x = x + self.GCN_B
        x = self.relu(x)

        if self.resnet:
            x = x + x_in

        return x

class AesSwinGCN(nn.Module):
    def __init__(self,dim_feature=768, num_classes=10):
        super(AesSwinGCN, self).__init__()#/home/cyh/nas/models--openai--backbone-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
        self.backbone = Swinv2Model.from_pretrained("/home/cyh/nas/models--microsoft--swinv2-tiny-patch4-window8-256/snapshots/40213dad8563a5a916d434a1291443c0fa6358f0")
        self.GCN_layer1 = GraphConvLayer(in_feature=dim_feature,out_feature=512, resnet=False, use_BN=True)
        self.GCN_layer2 = GraphConvLayer(in_feature=512,out_feature=256, resnet=False, use_BN=True)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 64, num_classes, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.backbone(x)
        self.feature = x.pooler_output
        x = self.GCN_layer1(x.last_hidden_state)
        x = self.GCN_layer2(x)
        return self.proj(x)
    
class AesSwin(nn.Module):
    def __init__(self,dim_feature=768, num_classes=10):
        super(AesSwin, self).__init__()#/home/cyh/nas/models--openai--backbone-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
        self.backbone = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.backbone = Swinv2Model.from_pretrained("microsoft/swinv2-base-patch4-window16-256")

        self.proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.1),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.backbone(x)
        self.feature = x.pooler_output
        return self.proj(x.pooler_output)