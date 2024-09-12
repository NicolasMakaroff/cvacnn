import torch
import numpy as np
from utils.dataset import kaggleDataset, create_data_loaders, Rectangle
from utils.transforms import transforms, test_transforms
from torch.utils.data import random_split
from hlnet import UNetTrainer
import itertools
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

#jax.config.update('jax_debug_nans', True)

data_dir = './data2/'
dataset = kaggleDataset(root_dir=data_dir + 'training_val/', transform=transforms)
#dataset = Rectangle(root_dir=data_dir, transform=False)
test_dataset = kaggleDataset(root_dir=data_dir + 'test/', transform=test_transforms)

[trainset, valset] = random_split(dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(0))

train_loader, val_loader, test_loader = create_data_loaders(trainset, valset, test_dataset, train=[True, False, False], batch_size=32)
#train_loader, val_loader = create_data_loaders(trainset, valset, train=[True, False], batch_size=32)



print('done creating dataset\n')
trainer = UNetTrainer(num_classes=1,
                        optimizer_hparams={
                            #'weight_decay': 2e-4,
                            'lr': 5e-4
                        },
                        logger_params={
                            #'base_log_dir': './checkpoint',
                            'logger_type': 'wandb',
                            'project_name': 'CVACNN'
                        },
                        exmp_input=next(iter(train_loader)),
                        check_val_every_n_epoch=10, 
                        debug=False,
                        pretrained=False)
print('done creating model\n')
print('Starting training\n')
metrics = trainer.train_model(train_loader,
                             val_loader,
                             test_loader,
                             num_epochs=1000)

