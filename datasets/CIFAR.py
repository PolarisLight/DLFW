import torch
import torchvision
from PIL import Image
from torchvision import transforms

class CIFAR10Dataset(torchvision.datasets.CIFAR10):
    def __init__(self, args,train=True):
        super(CIFAR10Dataset, self).__init__(root=args.dataset.data_dir, train=train, download=True)
        self.transform = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(self.transform)
        

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'image':img, 'label':target}

    def __len__(self):
        return len(self.data)
    
class CIFAR100Dataset(torchvision.datasets.CIFAR100):
    def __init__(self, args,train=True):
        super(CIFAR100Dataset, self).__init__(root=args.dataset.data_dir, train=train, download=True)
        self.transform = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(self.transform)
        

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'image':img, 'label':target}

    def __len__(self):
        return len(self.data)