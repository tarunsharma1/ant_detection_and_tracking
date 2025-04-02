'''

Code adopted from google collab example from https://github.com/Alexandre-Delplanque/HerdNet
use conda env "myherdnet"

'''

import sys
sys.path.append('/home/tarun/Desktop/HerdNet/')

from animaloc.utils.seed import set_seed

set_seed(9292)

import albumentations as A

from animaloc.datasets import CSVDataset
from animaloc.data.transforms import MultiTransformsWrapper, DownSample, PointsToMask, FIDT
from torch.utils.data import DataLoader
from animaloc.models import HerdNet
from torch import Tensor
from animaloc.models import LossWrapper
from animaloc.train.losses import FocalLoss
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from animaloc.train import Trainer
from animaloc.eval import PointsMetrics, HerdNetStitcher, HerdNetEvaluator
from animaloc.utils.useful_funcs import mkdir

patch_size = 512
num_classes = 2
down_ratio = 2

train_dataset = CSVDataset(
    csv_file = '/home/tarun/Desktop/antcam/datasets/herdnet_ants_manual_annotation/train_patches.csv',
    root_dir = '/home/tarun/Desktop/antcam/datasets/herdnet_ants_manual_annotation/train_patches',
    albu_transforms = [
        A.VerticalFlip(p=0.5), 
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        A.Blur(blur_limit=15, p=0.2),
        A.Normalize(p=1.0)
        ],
    end_transforms = [MultiTransformsWrapper([
        FIDT(num_classes=num_classes, down_ratio=down_ratio),
        PointsToMask(radius=2, num_classes=num_classes, squeeze=True, down_ratio=int(patch_size//16))
        ])]
    )

val_dataset = CSVDataset(
    csv_file = '/home/tarun/Desktop/antcam/datasets/herdnet_ants_manual_annotation/val.csv',
    root_dir = '/home/tarun/Desktop/antcam/datasets/herdnet_ants_manual_annotation/val',
    albu_transforms = [A.Normalize(p=1.0)],
    end_transforms = [DownSample(down_ratio=down_ratio, anno_type='point')]
    )

test_dataset = CSVDataset(
    csv_file = '/home/tarun/Desktop/antcam/datasets/herdnet_ants_manual_annotation/test.csv',
    root_dir = '/home/tarun/Desktop/antcam/datasets/herdnet_ants_manual_annotation/test',
    albu_transforms = [A.Normalize(p=1.0)],
    end_transforms = [DownSample(down_ratio=down_ratio, anno_type='point')]
    )

train_dataloader = DataLoader(dataset = train_dataset, batch_size = 4, shuffle = True)

val_dataloader = DataLoader(dataset = val_dataset, batch_size = 1, shuffle = False)

test_dataloader = DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False)

herdnet = HerdNet(num_classes=num_classes, down_ratio=down_ratio).cuda()

weight = Tensor([1.0, 1.0]).cuda()

losses = [
    {'loss': FocalLoss(reduction='mean'), 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'focal_loss'},
    {'loss': CrossEntropyLoss(reduction='mean', weight=weight), 'idx': 1, 'idy': 1, 'lambda': 1.0, 'name': 'ce_loss'}
    ]

herdnet = LossWrapper(herdnet, losses=losses)


work_dir = '/home/tarun/Desktop/HerdNet/output'
mkdir(work_dir)

lr = 1e-4
weight_decay = 1e-3
epochs = 100

optimizer = Adam(params=herdnet.parameters(), lr=lr, weight_decay=weight_decay)

metrics = PointsMetrics(radius=20, num_classes=num_classes)

stitcher = HerdNetStitcher(
    model=herdnet, 
    size=(patch_size,patch_size), 
    overlap=160, 
    down_ratio=down_ratio, 
    reduction='mean'
    )

evaluator = HerdNetEvaluator(
    model=herdnet, 
    dataloader=val_dataloader, 
    metrics=metrics, 
    stitcher=stitcher, 
    work_dir=work_dir, 
    header='validation'
    )

trainer = Trainer(
    model=herdnet,
    train_dataloader=train_dataloader,
    optimizer=optimizer,
    num_epochs=epochs,
    evaluator=evaluator,
    work_dir=work_dir
    )

trainer.start(warmup_iters=100, checkpoints='best', select='max', validate_on='f1_score')