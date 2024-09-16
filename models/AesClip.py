import torch
import torch.nn as nn
from transformers import CLIPVisionModel
from loss.emdloss import emd_loss
from loss.supcon import SupConEMD
from metrics.metrics import dis_2_score, ACCAVA, Pearson, Spearman
from typing import Union
from utils.utils import core_module


@core_module
class AesClip(nn.Module):
    def __init__(self,args)->None:
        """
        Initialize the model
        clip_model: name of the clip model
        num_classes: number of classes
        supcon_lambda: weight of the contrastive loss
        """
        super(AesClip, self).__init__()
        self.device = args.train.device
        self.feature_extractor = CLIPVisionModel.from_pretrained(args.model.pretrained_weights,)
        self.proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=args.model.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=args.model.dropout),
            nn.Linear(256, args.model.num_classes),
            nn.Softmax(dim=1)
        )
        self.train_loss = [SupConEMD(args)]
        self.val_loss = {"emdr2": emd_loss(dist_r=2), "emdr1": emd_loss(dist_r=1)}
        self.val_metric = {'acc': ACCAVA(), 'pearson': Pearson(), 'spearman': Spearman()}

    def get_core_params(self)->dict[str,list[float]]:
        """
        Get the core parameters
        """
        return {"dropout": [0.1,1]}

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        """
        forward pass of the model

        """
        x = self.feature_extractor(x)
        self.feature = x.pooler_output
        return self.proj(x.pooler_output)
    
    def train_step(self, data:dict[str,torch.Tensor])->dict[str,torch.Tensor]:
        """
        Train the model for one step
        data: input data
        """
        x, y = data['image'].to(self.device), data['label'].to(self.device)
        pred = self(x)
        for loss_fn in self.train_loss:
            if loss_fn.__class__.__name__ == "SupConEMD":
                loss = loss_fn(self.feature,pred, y)
            else:
                loss = loss_fn(pred, y)
        return {"loss": loss['loss']}
    
    def eval_step(self, data:dict[str,torch.Tensor])->dict[str,torch.Tensor]:
        """
        Evaluate the model for one step
        data: input data
        """
        x = data['image'].to(self.device)
        pred = self(x)
        return {"pred": pred}
    
    def on_eval_end(self, pred_list, target_list)->dict[str,dict[str,float]]:
        """
        Calculate the metrics
        outputs: output of the eval step
        mode: mode of the evaluation
        """
        loss_dict = {}
        metric_dict = {}
        def check_input_type(pred_list, target_list):
            assert isinstance(pred_list[0], torch.Tensor) or isinstance(pred_list[0], list)
            assert isinstance(target_list[0], torch.Tensor) or isinstance(target_list[0], list)
            if isinstance(pred_list, list):
                pred_list = torch.cat(pred_list, dim=0)
            if isinstance(target_list, list):
                target_list = torch.cat(target_list, dim=0)

            assert pred_list.shape[0] == target_list.shape[0]
            if pred_list.device != target_list.device:
                target_list = target_list.to(pred_list.device)
            return pred_list, target_list
        pred_list, target_list = check_input_type(pred_list, target_list)
        for key,loss_fn in self.val_loss.items():
            loss = loss_fn(pred_list, target_list)
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            loss_dict[key] = loss
        pred_score = dis_2_score(pred_list)
        true_score = dis_2_score(target_list)
        for key,metric_fn in self.val_metric.items():
            
            metric = metric_fn(pred_score, true_score)
            metric_dict[key] = metric
        return {"loss": loss_dict, "metric": metric_dict}



class AesClipMP(nn.Module):
    def __init__(self):
        super(AesClipMP, self).__init__()
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.proj = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(768, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        patches = x.shape[1]
        x = x.reshape(-1, 3, 224, 224)
        x = self.clip(x)
        self.feature = x.pooler_output
        proj = self.proj(x.pooler_output)
        return proj.contiguous().view(-1, patches, 10)
