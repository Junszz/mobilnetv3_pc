import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time

from mobilenetv3 import MobileNetV3
from tqdm.auto import tqdm

from model import build_model
from datasets import GTSRBDataLoader
from utils import save_model, save_plots


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs', type=int, default=5,
    help='Number of epochs to train our network for'
)
parser.add_argument(
    '-lr', '--learning-rate', type=float,
    dest='learning_rate', default=1e-1,
    help='Learning rate for training the model'
)
parser.add_argument(
    '-pw', '--pretrained', action='store_true', 
    help='whether to use pretrained weihgts or not'
)
parser.add_argument(
    '-ft', '--fine-tune', dest='fine_tune', action='store_true',
    help='whether to train all layers or not'
)
parser.add_argument(
    '--mode', type=str, default='large', help='large or small MobileNetV3'
)
parser.add_argument(
    '--step-size', type=int, default=3, help='step size in stepLR()'
)
parser.add_argument(
    '--width-multiplier', type=float, default=1.0, help='width multiplier'
)
parser.add_argument(
    '--dropout', type=float, default=0.2, help='dropout rate'
)
parser.add_argument(
    '--bn-momentum', type=float, default=0.1, help='momentum in BatchNorm2d'
)
parser.add_argument(
    '-zero-gamma', default=False, action='store_true', help='zero gamma in BatchNorm2d when init'
)
args = vars(parser.parse_args())

# Training function.
def train(model, trainloader, optimizer, criterion, scheduler, epoch):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    iters = len(trainloader)
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation.
        loss.backward()
        # Update the weights.
        optimizer.step()

        if scheduler is not None:
            scheduler.step(epoch + i / iters)
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

# Validation function.
def validate(model, testloader, criterion, class_names):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    # We need two lists to keep track of class-wise accuracy.
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

            # Calculate the accuracy for each class.
            correct  = (preds == labels).squeeze()
            for i in range(len(preds)):
                label = labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))

    # Print the accuracy for each class after every epoch.
    print('\n')
    for i in range(len(class_names)):
        print(f"Accuracy of class {class_names[i]}: {100*class_correct[i]/class_total[i]}")
    print('\n')
    return epoch_loss, epoch_acc

if __name__ == '__main__':
    # Load the training and validation datasets.
    loaders = GTSRBDataLoader()
    train_loader = loaders['train']
    train_loader_len = len(train_loader)
    val_loader = loaders['val']
    val_loader_len = len(val_loader)
    dataloaders = {'train' : train_loader, 'val' : val_loader}
    loaders_len = {'train': train_loader_len, 'val' : val_loader_len}

    print(f'{dataloaders},{loaders_len}')
    # dataset_train, dataset_valid, dataset_classes = get_datasets()
    # print(f"[INFO]: Number of training images: {len(dataset_train)}")
    # print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    # print(f"[INFO]: Class names: {dataset_classes}\n")
    # # Load the training and validation data loaders.
    # train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid)

    # Learning_parameters. 
    lr = args['learning_rate']
    epochs = args['epochs']
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")

    # Load the model.
    # model = build_model(
    #     pretrained=args['pretrained'],
    #     fine_tune=args['fine_tune'], 
    #     num_classes=len(dataset_classes)
    # ).to(device)
    model = MobileNetV3(mode=args['mode'], classes_num=len(dataset_classes), input_size=32, 
                    width_multiplier=args['width_multiplier'], dropout=args['dropout'], 
                    BN_momentum=args['bn_momentum'], zero_gamma=['args.zero_gamma'])
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Loss function.
    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10, 
        T_mult=1,
        verbose=True
    )

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=, gamma=args.gamma)

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    lrate = []
    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model, train_loader, 
            optimizer, criterion,
            scheduler=scheduler, epoch=epoch
        )
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
                                                    criterion, dataset_classes)
        current_lr = scheduler.get_last_lr()
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        lrate.append(current_lr)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)
        time.sleep(5)
        
    # Save the trained model weights.
    save_model(epochs, model, optimizer, criterion)
    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, lrate)
    print('TRAINING COMPLETE')