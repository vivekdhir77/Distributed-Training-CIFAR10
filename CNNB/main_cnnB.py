import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import logging
import time
from datetime import datetime, timedelta
import os
import glob
import argparse


## for distributed training
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', type=str, default='./weights',help='Directory to save model checkpoints')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 10 
batch_size = 16 
learning_rate = 0.001 
start = 0

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels))
        
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion))
    
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(identity)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global Average Pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        
        # First block with potential downsampling
        layers.append(block(self.in_channels, out_channels, stride))
        
        # Update in_channels for subsequent blocks
        self.in_channels = out_channels * block.expansion
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # ResNet backbone
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification layer
        x = self.fc(x)
        
        return x


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def load_data(batch_size):
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=False, sampler=DistributedSampler(train_dataset, shuffle=True))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True) 
    return train_loader, test_loader

def get_latest_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'latest.pth'))
    if not checkpoint_files:
        return None
    return max(checkpoint_files, key=os.path.getctime)

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, n_total_steps, initial_train_time=0):
    # For tracking metrics
    train_accuracies = []
    test_accuracies = []
    best_accuracy = 0.0
    total_train_time = initial_train_time  # start with previously accumulated time
    
    for epoch in range(start, num_epochs):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Set epoch for the DistributedSampler
        if distributed_training and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
            print(f"[INFO] Set epoch {epoch} for distributed sampler")
            
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # fwd pass
            if isinstance(model, DistributedDataParallel):
                outputs = model.module(images)
            else:
                outputs = model(images)
                
            loss = criterion(outputs, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

            if (i+1) % 512 == 0: 
                current_loss = loss.item()
                logging.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {current_loss:.4f}')
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {current_loss:.4f}')
        
        if scheduler is not None:
            scheduler.step()
            
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)
        

        test_accuracy = evaluate_accuracy(model, test_loader)
        test_accuracies.append(test_accuracy)
        
        epoch_time = time.time() - epoch_start_time
        total_train_time += epoch_time
        
        logging.info(f'Epoch {epoch+1} completed in {epoch_time:.2f} seconds')
        logging.info(f'Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
        logging.info(f'Total training time so far: {total_train_time:.2f} seconds')
        print(f'Epoch {epoch+1} completed in {epoch_time:.2f} seconds')
        print(f'Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
        print(f'Total training time so far: {total_train_time:.2f} seconds')
        
        # Update best accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            # Save best model
            if not distributed_training or (distributed_training and global_rank == 0):
                print(f"New best accuracy: {best_accuracy:.2f}%")
                PATH = os.path.join(args.model_folder, 'best_model.pth')
                # Save the correct state dict
                if isinstance(model, DistributedDataParallel):
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                    
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'best_accuracy': best_accuracy,
                    'total_train_time': total_train_time
                }, PATH)
        
        # Save checkpoint after each epoch
        PATH = os.path.join(args.model_folder, 'latest.pth')
        # Save the correct state dict
        if isinstance(model, DistributedDataParallel):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'train_accuracy': train_accuracies[-1] if train_accuracies else 0,
            'test_accuracy': test_accuracies[-1] if test_accuracies else 0,
            'best_accuracy': best_accuracy,
            'total_train_time': total_train_time
        }, PATH)
        
    return train_accuracies, test_accuracies, best_accuracy, total_train_time

def evaluate_accuracy(model, test_loader):
    """Calculate accuracy on a dataset"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            if isinstance(model, DistributedDataParallel):
                outputs = model.module(images)
            else:
                outputs = model(images)
                
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def evaluate_model(model, test_loader):
    confusion_matrix = torch.zeros(10, 10, dtype=torch.int64)
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        if isinstance(model, DistributedDataParallel):
            outputs = model.module(images)
        else:
            outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for t, p in zip(labels.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

    return confusion_matrix

def log_metrics(confusion_matrix):
    print('\nMetrics for each class:')
    print('------------------------')
    for i in range(10):
        TP = confusion_matrix[i, i].item()
        FP = (confusion_matrix[:, i].sum() - confusion_matrix[i, i]).item()
        FN = (confusion_matrix[i, :].sum() - confusion_matrix[i, i]).item()
        TN = (confusion_matrix.sum() - confusion_matrix[i, :].sum() - confusion_matrix[:, i].sum() + confusion_matrix[i, i]).item()

        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

        print(f'\nClass: {classes[i]}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-score: {f1:.4f}')
        print(f'Specificity: {specificity:.4f}')

        log_message = f"""
Class: {classes[i]}
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-score: {f1:.4f}
Specificity: {specificity:.4f}"""
        logging.info(log_message)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("This code requires GPU support. No GPU found!")

    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    global_rank = int(os.environ.get('RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1))
    
    if global_rank <= 0:  # Only download on the first process
        print(" Checking/Downloading CIFAR10 dataset...")
        torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
        torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        print(" Dataset is ready!")
    
    distributed_training = local_rank != -1 and global_rank != -1
    
    if distributed_training:
        print(f" Starting distributed training with:")
        print(f"       - Global rank: {global_rank}")
        print(f"       - Local rank: {local_rank}")
        print(f"       - World size: {world_size}")
        
        try:
            init_process_group(
                backend='nccl',
                init_method='env://',  # Use environment variables set by torchrun
                timeout=timedelta(seconds=60)  # Adding timeout to prevent indefinite hanging
            )
            print(f"[INFO] Process group initialized successfully!")
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
        except Exception as e:
            print(f"[ERROR] Failed to initialize distributed training: {str(e)}")
            raise
    else:
        print("[INFO] Running in single-GPU mode")
        device = torch.device('cuda:0')

    try:
        os.makedirs(args.model_folder, exist_ok=True)
    except PermissionError:
        print(f"[WARNING] No permission to create directory: {args.model_folder}")
        print("[WARNING] Falling back to local directory './weights'")
        args.model_folder = "./weights"
        os.makedirs(args.model_folder, exist_ok=True)

    if distributed_training:
        torch.distributed.barrier()

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)

    if distributed_training:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=False, sampler=train_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)

    model = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    checkpoint_dir = args.model_folder
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    
    train_accuracies = []
    test_accuracies = []
    best_accuracy = 0.0
    total_train_time = 0.0
    
    if latest_checkpoint:
        try:
            if distributed_training:
                state = torch.load(latest_checkpoint, map_location=f'cuda:{local_rank}')
            else:
                state = torch.load(latest_checkpoint, map_location='cuda:0')
            
            model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
            start = state['epoch'] + 1 
            
            if 'train_accuracies' in state:
                train_accuracies = state['train_accuracies']
                print(f"Loaded {len(train_accuracies)} previous train accuracy records")
            
            if 'test_accuracies' in state:
                test_accuracies = state['test_accuracies']
                print(f"Loaded {len(test_accuracies)} previous test accuracy records")
                
            if 'best_accuracy' in state:
                best_accuracy = state['best_accuracy']
                print(f"Previous best accuracy: {best_accuracy:.4f}%")
                
            if 'total_train_time' in state:
                total_train_time = state['total_train_time']
                print(f"Previous total training time: {total_train_time:.2f} seconds")
                
            print(f"Loaded model from {latest_checkpoint}")
            print(f"Continuing training from epoch {start}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")
            start = 0
    else:
        print("No checkpoint found. Starting training from scratch")
        start = 0

    if distributed_training:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    logging.basicConfig(
        filename=f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

    n_total_steps = len(train_loader)

    start_time = time.time()
    
    print(f"Starting training with:")
    print(f" Number of epochs: {num_epochs}")
    print(f" Batch size: {batch_size}")
    print(f" Total steps per epoch: {n_total_steps}")
    print(f" Learning rate: {learning_rate}")
    print(f" Starting from epoch: {start}")
    print(f" Device: {device}")
    if distributed_training:
        print(f" Distributed training enabled with {world_size} processes")
        
    train_accuracies, test_accuracies, best_accuracy, total_train_time = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, n_total_steps, total_train_time)

    total_time = time.time() - start_time

    if distributed_training:
        destroy_process_group()
    logging.info(f'Training completed in {total_time:.2f} seconds')
    print('Finished Training')
    PATH = os.path.join(args.model_folder, f'cnn_{batch_size}_{num_epochs}.pth')
    if isinstance(model, DistributedDataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
        
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'train_accuracy': train_accuracies[-1] if train_accuracies else 0,
        'test_accuracy': test_accuracies[-1] if test_accuracies else 0,
        'best_accuracy': best_accuracy,
        'total_train_time': total_train_time
    }, PATH)

    confusion_matrix = evaluate_model(model, test_loader)
    
    if not distributed_training or (distributed_training and global_rank == 0):
        log_metrics(confusion_matrix)

        if train_accuracies and test_accuracies:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            epochs = range(1, len(train_accuracies) + 1)
            plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
            plt.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
            plt.title(f'Training and Test Accuracy\nTotal Train Time: {total_train_time:.2f}s')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.model_folder, 'accuracy_plot.png'))
            plt.close()
            
            print(f"Training plots saved to {os.path.join(args.model_folder, 'accuracy_plot.png')}")
            print(f"Best test accuracy: {best_accuracy:.2f}%")
            print(f"Total training time: {total_train_time:.2f} seconds")
            
        print('\nConfusion Matrix:')
        print('----------------')
        print('Predicted →')
        print('Actual ↓')
        print('      ' + ''.join([f'{classes[i]:<7}' for i in range(10)]))
        for i in range(10):
            print(f'{classes[i]:<6}' + ''.join([f'{confusion_matrix[i, j].item():7d}' for j in range(10)]))

        logging.info('\nConfusion Matrix:')
        logging.info('----------------')
        logging.info('Predicted ->')
        logging.info('Actual ↓')
        logging.info('      ' + ''.join([f'{classes[i]:<7}' for i in range(10)]))
        for i in range(10):
            logging.info(f'{classes[i]:<6}' + ''.join([f'{confusion_matrix[i, j].item():7d}' for j in range(10)]))
