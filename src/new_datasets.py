import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms, datasets
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import skimage.io as io

TRAIN_DIR = '..\input\gtsrb\GTSRB\Final_training'
TEST_DIR = '..\input\gtsrb\GTSRB\Final_Test\PNG'

# train/val -> download True -> augmentation -> convert to png -> dataloader

#########################################################################################################################

    # image_datasets = {}
    # image_datasets['train'] = datasets.GTSRB(root=ROOT_DIR, split='train', transform=TrainTransform(), download=False)
    # image_datasets['test'] = datasets.GTSRB(root=ROOT_DIR, split='test', transform=ValidTransform(), download=False)

    # print(image_datasets)
    
    # train_dir = '..\input\gtsrb\GTSRB\Training'
    # train_root_folders = os.listdir(train_dir)
    # save_train_path = '..\input\gtsrb\GTSRB\Final_Training'
   
    # convert Train to png 

    # for file in train_root_folders:
    #     # class folder -> 00001, 00002, 00003
    #     # print(file) #00001
    #     image_path = os.path.join(train_dir, file) 
    #     # print(image_path) #..\input\gtsrb\GTSRB\Training\00001
    #     image_files = os.listdir(image_path) #..\input\gtsrb\GTSRB\Training\00001\00000_00001.ppm
    #     save_class_path = os.path.join(save_train_path, file)
    #     # print(save_class_path) #..\input\gtsrb\GTSRB\Final_Training\00042
    #     if not os.path.exists(save_class_path):
    #         os.mkdir(save_class_path)
    #     for images in image_files:
    #         image_name, file_type = os.path.splitext(images)
    #         print(image_name)
    #         if file_type == '.ppm':
    #             file_path = os.path.join(image_path, images)
    #             save_file_path = os.path.join(save_class_path, image_name + '.png')
    #             image = Image.open(file_path)
    #             # print(save_file_path)
    #             image.save(save_file_path)

    # convert Test to png 

    # test_dir = '..\input\gtsrb\GTSRB\Final_Test\Images'
    # test_root_folders = os.listdir(test_dir)
    # save_test_path = '..\input\gtsrb\GTSRB\Final_Test\PNG'
    # for images in test_root_folders:
    #     # print(images) #00001
    #     image_name, file_type = os.path.splitext(images)

    #     if file_type == '.ppm':
    #         file_path = os.path.join(test_dir, images)
    #         save_file_path = os.path.join(save_test_path, image_name + '.png')
    #         image = Image.open(file_path)
    #         # print(save_file_path)
    #         image.save(save_file_path)

#########################################################################################################################

# define transform
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.3403, 0.3121, 0.3214),
                        (0.2724, 0.2608, 0.2669))
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.3403, 0.3121, 0.3214),
                        (0.2724, 0.2608, 0.2669))
])

# dataset loader for test data
class TrafficSignDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_frame = pd.read_csv(csv_file, sep=';')
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.labels_frame)
    
    def __getitem__(self,index):
        img = self.labels_frame.iloc[index]

        image = Image.open(os.path.join(self.root_dir, img['filename']))
        label = img['ClassId']

        if self.transform:
            image = self.transform(image)
        
        sample = {'image': image, 'class': label}

        return sample

# dataset loader for train, valid
def get_train_valid_loader(data_dir,
                            train_batch_size,
                            val_batch_size,
                            train_transform,
                            valid_transform,
                            random_seed,
                            valid_size=0.1,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=False):
    """
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """

    # load the dataset
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

    classes = train_dataset.classes
    valid_dataset = datasets.ImageFolder(root=data_dir, transform=valid_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_data_size = len(train_sampler)
    valid_data_size = len(valid_sampler)

    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=val_batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader, train_data_size, valid_data_size, classes

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def check():
    labels_frame = pd.read_csv('..\input\gtsrb\GTSRB\Final_Test\PNG\GT-final_test.csv', sep=";")
    img_name = labels_frame.iloc[10]
    print(img_name)
    print(img_name['Filename'])
    # print(labels_frame.iloc[10,:])
    image = io.imread(os.path.join('..\input\gtsrb\GTSRB\Final_Test\PNG',img_name['Filename']))
    plt.imshow(image)
    plt.show()

    # GTSRB/Online-Test/
    root_dir = '..\input\gtsrb\GTSRB\Final_Test\PNG'
    traffic_dataset = TrafficSignDataset(csv_file = os.path.join(root_dir, 'GT-final_test.csv'), 
                                         root_dir = root_dir,
                                         transform = test_transform)

    test_loader = torch.utils.data.DataLoader(traffic_dataset, batch_size=8)                           
    data_iter = iter(test_loader)
    sample = data_iter.next()
    images, labels = sample['image'], sample['class']
    # Make a grid from batch
    out = make_grid(images)

    imshow(out, title=[x for x in labels])

if __name__ == '__main__':
    check()