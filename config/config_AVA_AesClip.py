import torch
from datetime import datetime
from loguru import logger
from config.Arguments import Arguments
import os

## local paths
root_dir = './' # 模型位置根目录
label_dir = '/home/cyh/nas/AesthesticDataset/AVA/labels'# 标签数据目录 (加载标签数据)
data_dir = '/home/cyh/nas/AesthesticDataset/AVA/images' # 图片数据目录 (加载图片数据)
utils_dir = './' # 训练代码位置目录

## server paths
save_dir = './Log/'


def config(task:str, model:str, dataset:str, opt:Arguments=None)->Arguments:
    """
    Configure the arguments for the training
    task: task name
    model: model name
    dataset: dataset name
    opt: additional arguments
    """
    args = Arguments()
    args.task = task
    ## parameters for output
    # create two subargs for model and training
    args.model = Arguments()
    args.train = Arguments()
    args.dataset = Arguments()
    args.scheduler = Arguments()

    # training parameters
    args.train.batch_size = 128
    args.train.epochs = 20
    args.train.lr = 1e-3
    args.train.optimizer = 'Adam'
    args.train.weight_decay = 0.0001
    args.train.num_workers = 8
    args.train.save_dir = os.path.join(save_dir, task)
    args.train.save_name = f'{dataset}_{model}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
    args.train.resume = None
    args.train.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.train.use_wandb = False
    args.train.clip_grad = 5
    args.train.save_freq = 4
    args.train.log_interval = 100
    # args.train.save_epoch = [10,20,30,40,50,60,70,80,90,100]
    args.train.early_stop = 5
    args.train.core_loss = 'emdr2'
    args.train.core_metric = 'pearson'
    args.train.use_wandb = False

    args.train.submodule_lr = True
    
    if args.train.submodule_lr:
        args.submodule = Arguments()
        args.submodule.feature_extractor = 3e-6
        args.submodule.proj = 3e-3
        # ...
        # add more submodule lr here, submodule name should be the same as the submodule in the model
    
    # learning rate scheduler

    args.scheduler.name = 'linear'
    args.scheduler.warmup_epoch = 5
    args.scheduler.total_epoch = args.train.epochs
    # args.scheduler.decay_epoch = 30
    # args.scheduler.decay_milestones = [160,180]
    args.scheduler.decay_rate = 0.1

    ## parameters for dataset
    args.dataset.name = dataset
    args.dataset.label_dir = label_dir
    args.dataset.data_dir = data_dir
    args.dataset.loader = None
    args.dataset.imgsz = 224

    ## parameters for model
    args.model.name = model
    # args.model.pretrained_weights = '/home/cyh/nas/models/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
    args.model.pretrained_weights = 'openai/clip-vit-base-patch32'
    args.model.num_classes = 10
    args.model.dropout = 0.3

    # core parameters
    args.core_params = Arguments()
    args.core_params.temperature = 0.1
    args.core_params.emd_threshold = 0.1
    args.core_params.weight_supcon = 0.1

    logger.add(f'{args.train.save_dir}/{args.train.save_name}/process.log',level="INFO",filter=lambda record: not record["extra"].get("params", False))
    logger.add(f'{args.train.save_dir}/{args.train.save_name}/params.log', level="INFO", filter=lambda record: record["extra"].get("params", False))
    return args
    