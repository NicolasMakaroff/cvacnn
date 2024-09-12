import cv2 
import os 
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
from typing import Sequence, Union

class kaggleDataset(Dataset):
    """Loading brain image/tumor segmentation couples"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        size = len([name for name in os.listdir(self.root_dir ) if os.path.isfile(self.root_dir + name)])
        return size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_folder = sorted([name for name in os.listdir(self.root_dir) if os.path.isfile(self.root_dir + name)])
        mask_folder = sorted([name for name in os.listdir(self.root_dir + 'mask/') if os.path.isfile(self.root_dir + 'mask/' + name)])

        img_name = os.path.join(self.root_dir, img_folder[idx])
        mask_name = os.path.join(self.root_dir + 'mask/', mask_folder[idx])

        image = cv2.cvtColor(cv2.imread(img_name),cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        
        sample = {'image': image, 'mask': mask } # TODO rajouter /255

        if self.transform:
            sample = self.transform(**sample)

        return sample['image'], sample['mask'][:,:,np.newaxis]

class Rectangle(Dataset):
    """Loading brain image/tumor segmentation couples"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        size = len([name for name in os.listdir(self.root_dir ) if os.path.isfile(self.root_dir + name)])
        return size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_folder = sorted([name for name in os.listdir(self.root_dir) if os.path.isfile(self.root_dir + name)])
        mask_folder = sorted([name for name in os.listdir(self.root_dir + 'mask/') if os.path.isfile(self.root_dir + 'mask/' + name)])

        img_name = os.path.join(self.root_dir, img_folder[idx])
        mask_name = os.path.join(self.root_dir + 'mask/', mask_folder[idx])

        image = np.load(img_name)
        mask = np.load(mask_name)
        
        sample = {'image': image/255, 'mask': mask/255 } # TODO rajouter /255

        if self.transform:
            sample = self.transform(**sample)

        return sample['image'], sample['mask'][:,:,np.newaxis]

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def create_data_loaders(*datasets : Sequence[Dataset],
                        train : Union[bool, Sequence[bool]] = True,
                        batch_size : int = 128,
                        num_workers : int = 0,
                        seed : int = 42):
    """
    Creates data loaders used in JAX for a set of datasets.

    Args:
      datasets: Datasets for which data loaders are created.
      train: Sequence indicating which datasets are used for
        training and which not. If single bool, the same value
        is used for all datasets.
      batch_size: Batch size to use in the data loaders.
      num_workers: Number of workers for each dataset.
      seed: Seed to initialize the workers and shuffling with.
    """
    loaders = []
    # TODO divide batch by number of local device to perform pmap optimization. create an other dim for the local device count at the loaders list append operation.
    if not isinstance(train, (list, tuple)):
        train = [train for _ in datasets]
    for dataset, is_train in zip(datasets, train):
        loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=is_train,
                                 drop_last=is_train,
                                 collate_fn=numpy_collate,
                                 num_workers=num_workers,
                                 #persistent_workers=is_train,
                                 generator=torch.Generator().manual_seed(seed))
        loaders.append(loader)
    return loaders


