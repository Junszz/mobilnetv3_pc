import os
from PIL import Image

ROOT_DIR = '../input'
image_datasets = {}
# image_datasets['train'] = datasets.GTSRB(root=ROOT_DIR, split='train', transform=TrainTransform(), download=False)
# image_datasets['test'] = datasets.GTSRB(root=ROOT_DIR, split='test', transform=ValidTransform(), download=False)

# print(image_datasets)

train_dir = '..\input\gtsrb\GTSRB\Training'
train_root_folders = os.listdir(train_dir)
save_train_path = '..\input\gtsrb\GTSRB\Final_Training'
   
# convert Train to png 

for file in train_root_folders:
    # class folder -> 00001, 00002, 00003
    # print(file) #00001
    image_path = os.path.join(train_dir, file) 
    # print(image_path) #..\input\gtsrb\GTSRB\Training\00001
    image_files = os.listdir(image_path) #..\input\gtsrb\GTSRB\Training\00001\00000_00001.ppm
    save_class_path = os.path.join(save_train_path, file)
    # print(save_class_path) #..\input\gtsrb\GTSRB\Final_Training\00042
    if not os.path.exists(save_class_path):
        os.mkdir(save_class_path)
    for images in image_files:
        image_name, file_type = os.path.splitext(images)
        print(image_name)
        if file_type == '.ppm':
            file_path = os.path.join(image_path, images)
            save_file_path = os.path.join(save_class_path, image_name + '.png')
            image = Image.open(file_path)
            # print(save_file_path)
            image.save(save_file_path)

# convert Test to png 

test_dir = '..\input\gtsrb\GTSRB\Final_Test\Images'
test_root_folders = os.listdir(test_dir)
save_test_path = '..\input\gtsrb\GTSRB\Final_Test\PNG'
for images in test_root_folders:
    # print(images) #00001
    image_name, file_type = os.path.splitext(images)

    if file_type == '.ppm':
        file_path = os.path.join(test_dir, images)
        save_file_path = os.path.join(save_test_path, image_name + '.png')
        image = Image.open(file_path)
        # print(save_file_path)
        image.save(save_file_path)