import os
import torch
import albumentations as A
import numpy as np

from collections import Counter
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from albumentations.pytorch import ToTensorV2

ROOT_DIR = '..\input'
VALID_SPLIT = 0.1
RESIZE_TO = 224
BATCH_SIZE = 128
NUM_WORKERS = 4

# Training transforms.
class TrainTransforms:
    def __init__(self, resize_to):
        # removed random transform as for now
        self.transforms = A.Compose([
            A.Resize(resize_to, resize_to), 
            # A.RandomBrightnessContrast(),
            # A.RandomFog(),
            # A.RandomRain(),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                ),
            ToTensorV2()
        ])
    
    def __call__(self, img):
        return self.transforms(image=np.array(img))['image']


# Validation transforms.
class ValidTransforms:
    def __init__(self, resize_to):
        self.transforms = A.Compose([
            A.Resize(resize_to, resize_to),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                ),
            ToTensorV2()
        ])
    
    def __call__(self, img):
        return self.transforms(image=np.array(img))['image']



# def GTSRBDataLoader():
#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.Resize((32, 32)),
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.3403, 0.3121, 0.3214),
#                                     (0.2724, 0.2608, 0.2669))
#         ]),
#         'val': transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.3403, 0.3121, 0.3214),
#                                     (0.2724, 0.2608, 0.2669))
#         ])
#     }

#     # image datasets in png format
#     image_datasets = {}
#     grand_datasets = datasets.ImageFolder(ROOT_DIR, transform= data_transforms)
#     #     # Calculate the validation dataset size. (9:1 split)
#     #     valid_size = int(VALID_SPLIT*dataset_size)
#     #     # Radomize the data indices.
#     #     indices = torch.randperm(len(dataset)).tolist()
#     #     # Training and validation sets.

#     #     #  Train: 0->valid_size  |  Test: valid_size->last
#     #     dataset_train = Subset(dataset, indices[:-valid_size])
#     #     dataset_valid = Subset(dataset_test, indices[-valid_size:])
#     image_datasets['train'] = datasets.ImageFolder(root=os.path.join(ROOT_DIR, 'Train'), transform=data_transforms['train'])
#     image_datasets['val'] = datasets.ImageFolder(root=os.path.join(ROOT_DIR, 'Test'), transform=data_transforms['val'])

#     dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True if x == 'train' else False,
#                     num_workers=NUM_WORKERS, pin_memory=True) for x in ['train', 'val']}
    
#     return dataloders
    
def get_datasets():
    dataset = datasets.GTSRB(root=ROOT_DIR, split='train', transform=ValidTransforms(), download=True)
    dataset_test = datasets.GTSRB(root=ROOT_DIR, split='test', transform=ValidTransforms(), download=True)

    # dataset = datasets.ImageFolder(
    #     ROOT_DIR, 
    #     transform=TrainTransforms()
    # )
    # dataset_test = datasets.ImageFolder(
    #     ROOT_DIR, 
    #     transform=ValidTransforms()
    # )
    dataset_size = len(dataset) 

    # Calculate the validation dataset size. (9:1 split)
    valid_size = int(VALID_SPLIT*dataset_size)
    # Radomize the data indices.
    indices = torch.randperm(len(dataset)).tolist()
    # Training and validation sets.

    #  Train: 0->valid_size  |  Test: valid_size->last
    dataset_train = Subset(dataset, indices[:-valid_size])
    dataset_valid = Subset(dataset_test, indices[-valid_size:])

    return dataset_train, dataset_valid, dataset.classes

def get_data_loaders(dataset_train, dataset_valid):
    """
    Prepares the training and validation data loaders.

    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.

    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS
    )
    return train_loader, valid_loader 