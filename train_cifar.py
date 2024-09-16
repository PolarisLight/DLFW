"""
This is the main file for training CIFAR dataset.
This file is used to test this framework.
"""
from argparse import Namespace, ArgumentParser
from utils.utils import set_seed
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# =============================================================================
# need to be redefined for every new task
from config.config_cifar import config
import models.MyResnet as models
import datasets.CIFAR as datasets
# =============================================================================

def get_model_and_dataset(args:Namespace)->tuple:
    """
    Get the model and dataset.
    This function need to be redefined for every new task.
    """
    model = getattr(models, args.model.name)(args)
    dataset = {'train': getattr(datasets, args.dataset.name)(args, train=True),
               'val': getattr(datasets, args.dataset.name)(args, train=False)}
    return model, dataset

def main(args:Namespace)->None:
    """
    Main function
    args: arguments
    """
    model, dataset = get_model_and_dataset(args)
    if args.train.submodule_lr:
        from Trainer.submodulelr_trainer import Submodulelr_Trainer
        trainer = Submodulelr_Trainer(args, model, dataset, args.train.device) 
    else:
        from Trainer.default_trainer import Default_Trainer
        trainer = Default_Trainer(args, model, dataset, args.train.device)
    trainer.train()
    
if __name__ == '__main__':
    opt = ArgumentParser()
    opt.add_argument('--task', type=str, default='TEST', help='task name')
    opt.add_argument('--model', type=str, default='MyResnet', help='model name')
    opt.add_argument('--dataset', type=str, default='CIFAR100Dataset', help='dataset name')
    opt.add_argument('--seed', type=int, default=2024, help='random seed')
    opt = opt.parse_args()
    set_seed(opt.seed)
    args = config(opt.task, opt.model, opt.dataset)
    main(args)