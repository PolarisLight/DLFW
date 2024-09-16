import functools
from loguru import logger
import random
import numpy as np
import torch
import os

def core_module(cls):
    """
    装饰器，用于标注核心模块并输出核心超参数。
    """
    orig_init = cls.__init__

    @functools.wraps(cls.__init__)
    def new_init(self, *args):
        logger.bind(params=True).info(f"{'='*30}")
        logger.bind(params=True).info(f"Instantiating core module: {cls.__name__}")
        # 调用类中的 get_core_params 方法获取核心参数及其推荐调参范围
        if hasattr(self, 'get_core_params'):
            core_params = self.get_core_params()
            logger.bind(params=True).info("Core hyperparameters and recommended tuning ranges:")
            for param, tuning_range in core_params.items():
                # 如果核心参数在 kwargs 中，获取其值，否则使用对象默认值
                logger.bind(params=True).info(f"  {param} (Recommended range: {tuning_range})")
        logger.bind(params=True).info(f"{'='*30}")
        # 调用原始的 __init__ 方法
        orig_init(self, *args)
    
    cls.__init__ = new_init
    return cls

def set_seed(seed_value):
    # 设置 Python 内置随机性
    random.seed(seed_value)
    
    # 设置 NumPy 随机性
    np.random.seed(seed_value)
    
    # 设置 PyTorch CPU 随机性
    torch.manual_seed(seed_value)
    
    # 如果使用 GPU，则设置 GPU 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    
    # CuDNN 确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置操作系统相关的随机性
    os.environ['PYTHONHASHSEED'] = str(seed_value)