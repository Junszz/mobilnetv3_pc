import os
from tkinter.tix import IMAGE
from xml.etree.ElementInclude import default_loader
import torch
import numpy as np

from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

TRAIN_DIR = '..\input\gtsrb\GTSRB\Final_training'
TEST_DIR = '..\input\gtsrb\GTSRB\Final_Test\PNG'


# train/val -> download True -> augmentation -> convert to png -> dataloader
def GTSRBDataLoader():

    def TrainTransform():
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                            (0.2724, 0.2608, 0.2669))

    def ValidTransform():
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                            (0.2724, 0.2608, 0.2669))

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

    # create training CSV

    # pass to dataloader

    class loadDataset():
        def __init__(self, root_dir, transform=None, loader=default_loader):
            imgs = []
            for images in root_dir:
                image_path = os.path.join(root_dir, images)
                
                self.loader = loader
                self.imgs = imgs
                self.transform = transform
                self.root_dir = root_dir
        
        def __len__(self):
            return len(self.imgs)
        
        def __getitem__(self,index):
            images, label = self.imgs[index]
            image = self.loader(images)
            image = self.transform(image)
            return image, label

    dataset = datasets.ImageFolder(
        TRAIN_DIR, 
        transform=TrainTransform()
    )

    dataset_test = datasets.ImageFolder(
        TEST_DIR, 
        transform=ValidTransform()
    )

    print(f'Training size: {len(dataset)}')
    print(f'Test size: {len(dataset_test)}')


if __name__ == '__main__':
    test = GTSRBDataLoader()