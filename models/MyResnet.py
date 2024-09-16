import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet50_Weights, resnet50
from metrics.metrics import ACC

class MyResnet(nn.Module):
    def __init__(self,args)->None:
        super(MyResnet, self).__init__()
        self.device = args.train.device
        self.feature_extractor = resnet50(pretrained=ResNet50_Weights)
        in_dim = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, args.model.num_classes),
            nn.Softmax(dim=1)
        )
        from torch.nn import CrossEntropyLoss
        self.train_loss = [CrossEntropyLoss()]
        self.val_loss = {"ce": CrossEntropyLoss()}
        self.val_metric = {'acc': ACC()}

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.proj(x)
    
    def train_step(self, data:dict[str,torch.Tensor])->dict[str,torch.Tensor]:
        """
        Train the model for one step
        data: input data
        """
        x, y = data['image'].to(self.device), data['label'].to(self.device)
        pred = self(x)

        for loss_fn in self.train_loss:
            loss = loss_fn(pred, y)
        return {"loss": loss}

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
            return pred_list, target_list
        pred_list, target_list = check_input_type(pred_list, target_list)
        for key,loss_fn in self.val_loss.items():
            loss = loss_fn(pred_list, target_list)
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            loss_dict[key] = loss
        for key,metric_fn in self.val_metric.items():
            metric = metric_fn(pred_list, target_list)
            metric_dict[key] = metric
        return {"loss": loss_dict, "metric": metric_dict}
