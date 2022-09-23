import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time

from mobilenetv3 import MobileNetV3
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from model import build_model
from new_datasets import TrafficSignDataset, get_train_valid_loader
from utils import save_model, save_plots


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Training function.
# def train(model, trainloader, optimizer, criterion, scheduler, epoch):
#     model.train()
#     print('Training')
#     train_running_loss = 0.0
#     train_running_correct = 0
#     counter = 0
#     iters = len(trainloader)
#     for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
#         counter += 1
#         image, labels = data
#         image = image.to(device)
#         labels = labels.to(device)
#         optimizer.zero_grad()
#         # Forward pass.
#         outputs = model(image)
#         # Calculate the loss.
#         loss = criterion(outputs, labels)
#         train_running_loss += loss.item()
#         # Calculate the accuracy.
#         _, preds = torch.max(outputs.data, 1)
#         train_running_correct += (preds == labels).sum().item()
#         # Backpropagation.
#         loss.backward()
#         # Update the weights.
#         optimizer.step()

#         if scheduler is not None:
#             scheduler.step(epoch + i / iters)
    
#     # Loss and accuracy for the complete epoch.
#     epoch_loss = train_running_loss / counter
#     epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
#     return epoch_loss, epoch_acc

# # Validation function.
# def validate(model, testloader, criterion, class_names):
#     model.eval()
#     print('Validation')
#     valid_running_loss = 0.0
#     valid_running_correct = 0
#     counter = 0

#     # We need two lists to keep track of class-wise accuracy.
#     class_correct = list(0. for i in range(len(class_names)))
#     class_total = list(0. for i in range(len(class_names)))

#     with torch.no_grad():
#         for i, data in tqdm(enumerate(testloader), total=len(testloader)):
#             counter += 1
            
#             image, labels = data
#             image = image.to(device)
#             labels = labels.to(device)
#             # Forward pass.
#             outputs = model(image)
#             # Calculate the loss.
#             loss = criterion(outputs, labels)
#             valid_running_loss += loss.item()
#             # Calculate the accuracy.
#             _, preds = torch.max(outputs.data, 1)
#             valid_running_correct += (preds == labels).sum().item()

#             # Calculate the accuracy for each class.
#             correct  = (preds == labels).squeeze()
#             for i in range(len(preds)):
#                 label = labels[i]
#                 class_correct[label] += correct[i].item()
#                 class_total[label] += 1
        
#     # Loss and accuracy for the complete epoch.
#     epoch_loss = valid_running_loss / counter
#     epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))

#     # Print the accuracy for each class after every epoch.
#     print('\n')
#     for i in range(len(class_names)):
#         print(f"Accuracy of class {class_names[i]}: {100*class_correct[i]/class_total[i]}")
#     print('\n')
#     return epoch_loss, epoch_acc

if __name__ == '__main__':
    # Construct the argument parser.
    parser = argparse.ArgumentParser(description='PyTorch implementation of MobileNetV3')
    # Root catalog of images
    parser.add_argument('--data-dir', type=str, default='/media/data2/chenjiarong/ImageData')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4)
    #parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--print-freq', type=int, default=1000)
    parser.add_argument('--save-epoch-freq', type=int, default=1)
    parser.add_argument('--save-path', type=str, default='/media/data2/chenjiarong/saved-model/MobileNetV3')
    parser.add_argument('-save', default=False, action='store_true', help='save model or not')
    parser.add_argument('--resume', type=str, default='', help='For training from one checkpoint')
    parser.add_argument('--start-epoch', type=int, default=0, help='Corresponding to the epoch of resume')
    parser.add_argument('--ema-decay', type=float, default=0.9999, help='The decay of exponential moving average ')
    parser.add_argument('--dataset', type=str, default='ImageNet', help='The dataset to be trained')
    parser.add_argument('-dali', default=False, action='store_true', help='Using DALI or not')
    parser.add_argument('--mode', type=str, default='large', help='large or small MobileNetV3')
    # parser.add_argument('--num-class', type=int, default=1000)
    parser.add_argument('--width-multiplier', type=float, default=1.0, help='width multiplier')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--lr-decay', type=str, default='step', help='learning rate decay method, step, cos or sgdr')
    parser.add_argument('--step-size', type=int, default=3, help='step size in stepLR()')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma in stepLR()')
    parser.add_argument('--lr-min', type=float, default=0, help='minium lr using in CosineWarmupLR')
    parser.add_argument('--warmup-epochs', type=int, default=0, help='warmup epochs using in CosineWarmupLR')
    parser.add_argument('--T-0', type=int, default=10, help='T_0 in CosineAnnealingWarmRestarts')
    parser.add_argument('--T-mult', type=int, default=2, help='T_mult in CosineAnnealingWarmRestarts')
    parser.add_argument('--decay-rate', type=float, default=1, help='decay rate in CosineAnnealingWarmRestarts')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--bn-momentum', type=float, default=0.1, help='momentum in BatchNorm2d')
    parser.add_argument('-use-seed', default=False, action='store_true', help='using fixed random seed or not')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('-deterministic', default=False, action='store_true', help='torch.backends.cudnn.deterministic')
    parser.add_argument('-nbd', default=False, action='store_true', help='no bias decay')
    parser.add_argument('-zero-gamma', default=False, action='store_true', help='zero gamma in BatchNorm2d when init')
    parser.add_argument('-mixup', default=False, action='store_true', help='mixup or not')
    parser.add_argument('--mixup-alpha', type=float, default=0.2, help='alpha used in mixup')
    args = parser.parse_args()

    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))

    # set random seed
    if args.use_seed:
        print('Using fixed random seed')
        torch.manual_seed(args.seed)
    else:
        print('do not use fixed random seed')
    if use_gpu:
        if args.use_seed:
            torch.cuda.manual_seed(args.seed)
            if torch.cuda.device_count() > 1:
                torch.cuda.manual_seed_all(args.seed)
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        print('torch.backends.cudnn.deterministic:' + str(args.deterministic))
    
    # Load the training and validation datasets.
    train_data_dir = '..\input\gtsrb\GTSRB\Final_Training'
    test_data_dir = '..\input\gtsrb\GTSRB\Final_Test\PNG'
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
    # train dataset
    train_loader, valid_loader, train_size, valid_size,\
    classes = get_train_valid_loader(train_data_dir, train_batch_size=64,
                                    val_batch_size = 16,
                                    train_transform = train_transform,
                                    valid_transform = test_transform,
                                    random_seed = 42,
                                    valid_size = 0.1,
                                    shuffle = True,
                                    num_workers = 4,
                                    pin_memory = True)
    # test dataset
    test_data_set = TrafficSignDataset(csv_file = os.path.join(test_data_dir,'GT-final_test.csv'),
                                        root_dir = test_data_dir,
                                        transform = test_transform)
    # change batch_size according to need
    test_loader = DataLoader(test_data_set, batch_size = 16)
    print(f"[INFO]: Number of training images: {train_size}")
    print(f"[INFO]: Number of validation images: {valid_size}")
    print(f"[INFO]: Class names: {classes}\n")

    # Learning_parameters. 
    # lr = args['learning_rate']
    # epochs = args['epochs']
    # device = ('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Computation device: {device}")
    # print(f"Learning rate: {lr}")
    # print(f"Epochs to train for: {epochs}\n")

    # # Load the model.
    # # model = build_model(
    # #     pretrained=args['pretrained'],
    # #     fine_tune=args['fine_tune'], 
    # #     num_classes=len(dataset_classes)
    # # ).to(device)
    # model = MobileNetV3(mode=args['mode'], classes_num=len(dataset_classes), input_size=32, 
    #                 width_multiplier=args['width_multiplier'], dropout=args['dropout'], 
    #                 BN_momentum=args['bn_momentum'], zero_gamma=['args.zero_gamma'])
    
    # # Total parameters and trainable parameters.
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"{total_params:,} total parameters.")
    # total_trainable_params = sum(
    #     p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"{total_trainable_params:,} training parameters.")

    # # Optimizer.
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # # Loss function.
    # criterion = nn.CrossEntropyLoss()

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, 
    #     T_0=10, 
    #     T_mult=1,
    #     verbose=True
    # )

    # # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=, gamma=args.gamma)

    # # Lists to keep track of losses and accuracies.
    # train_loss, valid_loss = [], []
    # train_acc, valid_acc = [], []
    # lrate = []
    # # Start the training.
    # for epoch in range(epochs):
    #     print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    #     train_epoch_loss, train_epoch_acc = train(
    #         model, train_loader, 
    #         optimizer, criterion,
    #         scheduler=scheduler, epoch=epoch
    #     )
    #     valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
    #                                                 criterion, dataset_classes)
    #     current_lr = scheduler.get_last_lr()
    #     train_loss.append(train_epoch_loss)
    #     valid_loss.append(valid_epoch_loss)
    #     train_acc.append(train_epoch_acc)
    #     valid_acc.append(valid_epoch_acc)
    #     lrate.append(current_lr)
    #     print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    #     print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    #     print('-'*50)
    #     time.sleep(5)
        
    # # Save the trained model weights.
    # save_model(epochs, model, optimizer, criterion)
    # # Save the loss and accuracy plots.
    # save_plots(train_acc, valid_acc, train_loss, valid_loss, lrate)
    # print('TRAINING COMPLETE')