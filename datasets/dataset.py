from .utils import transform_options, dataset_options
from torch.utils.data import DataLoader
from torchvision import transforms


class DatasetGenerator():
    def __init__(self, train_bs=128, eval_bs=256, seed=0, n_workers=4,
                 train_d_type='CIFAR10', test_d_type='CIFAR10',
                 train_path='../../datasets/', test_path='../../datasets/',
                 **kwargs):

        if train_d_type not in transform_options:
            raise('Unknown Dataset')
        elif test_d_type not in transform_options:
            raise('Unknown Dataset')

        self.train_bs = train_bs
        self.eval_bs = eval_bs
        self.n_workers = n_workers
        self.train_path = train_path
        self.test_path = test_path

        train_tf = transform_options[train_d_type]["train_transform"]
        test_tf = transform_options[test_d_type]["test_transform"]
        train_tf = transforms.Compose(train_tf)
        test_tf = transforms.Compose(test_tf)

        self.train_set = dataset_options[train_d_type](train_path, train_tf,
                                                       False, kwargs)
        self.test_set = dataset_options[test_d_type](test_path, test_tf,
                                                     True, kwargs)
        self.train_set_length = len(self.train_set)
        self.test_set_length = len(self.test_set)

    def get_loader(self, train_shuffle=True):
        train_loader = DataLoader(dataset=self.train_set, pin_memory=True,
                                  batch_size=self.train_bs, drop_last=False,
                                  num_workers=self.n_workers,
                                  shuffle=train_shuffle)

        test_loader = DataLoader(dataset=self.test_set, pin_memory=True,
                                 batch_size=self.eval_bs, drop_last=False,
                                 num_workers=self.n_workers, shuffle=False)
        return train_loader, test_loader
