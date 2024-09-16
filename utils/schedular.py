"""
learning rate schedular
"""
import numpy as np

def get_scheduler(args)->callable:
    """
    Get the scheduler
    scheduler_name: scheduler name
    original_lr: original learning rate
    warmup_epoch: warmup epochs
    total_epoch: total epochs
    decay_epoch: decay epochs
    decay_rate: decay rate
    """
    if args.scheduler.name == 'step':
        return step_lr_scheduler(args.train.lr, args.scheduler.warmup_epoch, args.scheduler.total_epoch, args.scheduler.decay_rate)
    elif args.scheduler.name == 'cosine':
        return cosine_lr_scheduler(args.train.lr, args.scheduler.warmup_epoch, args.scheduler.total_epoch)
    elif args.scheduler.name == 'linear':
        return linear_lr_scheduler(args.train.lr, args.scheduler.warmup_epoch, args.scheduler.total_epoch)
    elif args.scheduler.name == 'exp':
        return exp_lr_scheduler(args.train.lr, args.scheduler.warmup_epoch, args.scheduler.decay_epoch, args.scheduler.decay_rate)
    elif args.scheduler.name == 'milestone':
        assert args.scheduler.decay_milestones is not None, "**decay_milestones** should be provided for milestone scheduler, not decay_epoch"
        return milestone_lr_scheduler(args.train.lr, args.scheduler.warmup_epoch, args.scheduler.total_epoch, args.scheduler.decay_milestones, args.scheduler.decay_rate)
    else:
        raise NotImplementedError(f"Unknown scheduler: {args.scheduler.name}")



# step learning rate scheduler with warmup
class step_lr_scheduler:
    def __init__(self, original_lr, warmup_epoch, decay_epoch, decay_rate):
        self.original_lr = original_lr
        self.warmup_epoch = warmup_epoch
        self.decay_epoch = decay_epoch
        self.decay_rate = decay_rate

    def __call__(self, epoch):
        if epoch < self.warmup_epoch:
            lr = self.original_lr * (epoch + 1) / self.warmup_epoch
        else:
            lr = self.original_lr * self.decay_rate ** ((epoch - self.warmup_epoch) // self.decay_epoch)
        return lr
    
    def calculate_lr(self, lr,epoch):
        if epoch < self.warmup_epoch:
            lr = lr * (epoch + 1) / self.warmup_epoch
        else:
            lr = lr * self.decay_rate ** ((epoch - self.warmup_epoch) // self.decay_epoch)
        return lr

# cosine learning rate scheduler with warmup
class cosine_lr_scheduler:
    def __init__(self, original_lr, warmup_epoch, total_epoch):
        self.original_lr = original_lr
        self.warmup_epoch = warmup_epoch
        self.total_epoch = total_epoch

    def __call__(self, epoch):
        if epoch < self.warmup_epoch:
            lr = self.original_lr * (epoch + 1) / self.warmup_epoch
        else:
            lr = 0.5 * self.original_lr * (1 + np.cos(np.pi * (epoch - self.warmup_epoch) / (self.total_epoch - self.warmup_epoch)))
        return lr
    
    def calculate_lr(self, lr,epoch):
        if epoch < self.warmup_epoch:
            lr = lr * (epoch + 1) / self.warmup_epoch
        else:
            lr = 0.5 * lr * (1 + np.cos(np.pi * (epoch - self.warmup_epoch) / (self.total_epoch - self.warmup_epoch)))
        return lr

# linear learning rate scheduler with warmup
class linear_lr_scheduler:
    def __init__(self, original_lr, warmup_epoch, total_epoch):
        self.original_lr = original_lr
        self.warmup_epoch = warmup_epoch
        self.total_epoch = total_epoch

    def __call__(self, epoch):
        if epoch < self.warmup_epoch:
            lr = self.original_lr * (epoch + 1) / self.warmup_epoch
        else:
            lr = self.original_lr * (1 - (epoch - self.warmup_epoch) / (self.total_epoch - self.warmup_epoch))
        return lr
    
    def calculate_lr(self, lr,epoch):
        if epoch < self.warmup_epoch:
            lr = lr * (epoch + 1) / self.warmup_epoch
        else:
            lr = lr * (1 - (epoch - self.warmup_epoch) / (self.total_epoch - self.warmup_epoch))
        return lr

# exponential learning rate scheduler with warmup
class exp_lr_scheduler:
    def __init__(self, original_lr, warmup_epoch, decay_epoch, decay_rate):
        self.original_lr = original_lr
        self.warmup_epoch = warmup_epoch
        self.decay_epoch = decay_epoch
        self.decay_rate = decay_rate

    def __call__(self, epoch):
        if epoch < self.warmup_epoch:
            lr = self.original_lr * (epoch + 1) / self.warmup_epoch
        else:
            lr = self.original_lr * self.decay_rate ** ((epoch - self.warmup_epoch) // self.decay_epoch)
        return lr
    
    def calculate_lr(self, lr,epoch):
        if epoch < self.warmup_epoch:
            lr = lr * (epoch + 1) / self.warmup_epoch
        else:
            lr = lr * self.decay_rate ** ((epoch - self.warmup_epoch) // self.decay_epoch)

class milestone_lr_scheduler:
    def __init__(self, original_lr, warmup_epoch, total_epoch, decay_epochs, decay_rate):
        self.original_lr = original_lr
        self.warmup_epoch = warmup_epoch
        self.total_epoch = total_epoch
        self.decay_epochs = decay_epochs
        self.decay_rate = decay_rate

    def __call__(self, epoch):
        if epoch < self.warmup_epoch:
            lr = self.original_lr * (epoch + 1) / self.warmup_epoch
        else:
            lr = self.original_lr
            for decay_epoch in self.decay_epochs:
                if epoch >= decay_epoch:
                    lr *= self.decay_rate
        return lr
    
    def calculate_lr(self, lr,epoch):
        if epoch < self.warmup_epoch:
            lr = lr * (epoch + 1) / self.warmup_epoch
        else:
            for decay_epoch in self.decay_epochs:
                if epoch >= decay_epoch:
                    lr *= self.decay_rate
        return lr