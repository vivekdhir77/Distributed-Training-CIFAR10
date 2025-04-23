import warnings
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
import os
from datetime import datetime, timedelta
import time
from pathlib import Path
import glob
import logging
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', type=str, default='weights', help='Directory to save model checkpoints')
args = parser.parse_args()
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
def get_default_config():
    batch_size = 16
    num_epochs = 10
    learning_rate = 3e-4
    
    image_size = 32
    patch_size = 4
    embed_dim = 192
    num_heads = 3
    num_layers = 6
    num_classes = 10
    forward_expansion = 4
    dropout = 0.1
    stochastic_depth = 0.1
    
    model_folder = args.model_folder
    model_basename = "tmodel_{0:02d}.pt"
    preload = "latest"
    tokenizer_file = "tokenizer_{0}.json"
    
    local_rank = -1
    global_rank = -1

    return {
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'image_size': image_size,
        'patch_size': patch_size,
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'num_classes': num_classes,
        'forward_expansion': forward_expansion,
        'dropout': dropout,
        'stochastic_depth': stochastic_depth,
        'model_folder': model_folder,
        'model_basename': model_basename,
        'preload': preload,
        'tokenizer_file': tokenizer_file,
        'local_rank': local_rank,
        'global_rank': global_rank
    }

def get_weights_file_path(model_folder, model_basename, epoch):
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    filename = model_basename.format(epoch)
    full_path = os.path.join(model_folder, filename)
    
    return full_path

def get_latest_weights_file_path(model_folder):
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    latest_path = os.path.join(model_folder, 'latest.pt')
    if os.path.exists(latest_path):
        return latest_path
    
    all_files = os.listdir(model_folder)
    model_files = [f for f in all_files if f.startswith('tmodel_') and f.endswith('.pt')]
    
    if not model_files:
        return None
    epochs = []
    for f in model_files:
        try:
            epoch = int(f.split('_')[1].split('.')[0])
            epochs.append(epoch)
        except:
            continue
    
    if not epochs:
        return None
    
    latest_epoch = max(epochs)
    return os.path.join(model_folder, f'tmodel_{latest_epoch:02d}.pt')

# Vision Transformer Model
class EmbedLayer(nn.Module):
    def __init__(self, n_channels, embed_dim, image_size, patch_size, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = n_channels * patch_size * patch_size

        self.proj = nn.Linear(self.patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, self.num_patches, -1)
        x = self.proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        return self.dropout(self.norm(x))

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_probs = attn_scores.softmax(dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion, dropout=0.1, stochastic_depth=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.drop_path1 = nn.Dropout(stochastic_depth)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, forward_expansion * embed_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(forward_expansion * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.drop_path2 = nn.Dropout(stochastic_depth)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.dropout(self.fc2(self.activation(self.fc1(self.norm2(x))))))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, n_channels, embed_dim, num_layers, num_heads, forward_expansion, image_size, patch_size, num_classes, dropout=0.1, stochastic_depth=0.1):
        super().__init__()
        self.embedding = EmbedLayer(n_channels, embed_dim, image_size, patch_size, dropout)
        self.encoder = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, forward_expansion, dropout, stochastic_depth)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.classifier(x[:, 0])  # Use of CLS token
        return x


def get_data_loaders(batch_size, distributed_training=False, local_rank=-1):

    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomRotation(15), 
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)), 
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # normalization for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    if distributed_training:
        train_sampler = DistributedSampler(trainset, shuffle=True)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=False,
            sampler=train_sampler, num_workers=4, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return trainloader, testloader


def evaluate_model(model, test_loader, device, num_classes=10):
    model.eval()
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    with torch.no_grad():
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

def log_metrics(confusion_matrix, classes):
    """Calculate and log metrics for each class based on confusion matrix"""
    print('\nMetrics for each class:')
    print('------------------------')
    
    for i in range(len(classes)):
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
    total_correct = confusion_matrix.diag().sum().item()
    total_samples = confusion_matrix.sum().item()
    overall_accuracy = total_correct / total_samples
    
    print(f'\nOverall Accuracy: {overall_accuracy:.4f}')
    logging.info(f'Overall Accuracy: {overall_accuracy:.4f}')
    
    return overall_accuracy

def train_model(batch_size, num_epochs, learning_rate, model_folder, model_basename, 
                preload, tokenizer_file, image_size, patch_size, embed_dim, num_heads, 
                num_layers, num_classes, forward_expansion, dropout, stochastic_depth,
                local_rank=-1, global_rank=-1, distributed_training=False):

    if not torch.cuda.is_available():
        print("Error: CUDA is not available!")
        return
    
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    if distributed_training:
        device = torch.device(f'cuda:{local_rank}')
        print(f"Using GPU {local_rank} in distributed mode")
    else:
        device = torch.device('cuda:0')
        print("Using single GPU mode")
    
    trainloader, testloader = get_data_loaders(batch_size, distributed_training, local_rank)
    

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    log_file = f'vit_training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    

    model = VisionTransformer(
        n_channels=3, 
        embed_dim=embed_dim, 
        num_layers=num_layers,
        num_heads=num_heads, 
        forward_expansion=forward_expansion,
        image_size=image_size, 
        patch_size=patch_size,
        num_classes=num_classes, 
        dropout=dropout, 
        stochastic_depth=stochastic_depth
    )
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    start_epoch = 0
    global_step = 0
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    best_accuracy = 0.0
    total_train_time = 0
    
    if preload:
        if preload == 'latest':
            checkpoint_path = get_latest_weights_file_path(model_folder)
        else:
            checkpoint_path = get_weights_file_path(model_folder, model_basename, int(preload))
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            try:
                if distributed_training:
                    checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{local_rank}')
                else:
                    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
                
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                global_step = checkpoint.get('global_step', 0)
                
                if 'train_accuracies_history' in checkpoint:
                    train_accuracies = checkpoint['train_accuracies_history']
                if 'test_accuracies_history' in checkpoint:
                    val_accuracies = checkpoint['test_accuracies_history']
                if 'best_accuracy' in checkpoint:
                    best_accuracy = checkpoint['best_accuracy']
                if 'total_train_time' in checkpoint:
                    total_train_time = checkpoint['total_train_time']
                
                print(f"Resuming training from epoch {start_epoch}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting fresh training")
        else:
            print("No checkpoint found, starting fresh training")
    
    if distributed_training:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        if distributed_training and hasattr(trainloader, 'sampler'):
            trainloader.sampler.set_epoch(epoch)
        
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                if distributed_training:
                    outputs = model(images)
                else:
                    outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], 'f'Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%')
        
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, 'f'Accuracy: {epoch_acc:.2f}%, Time: {epoch_time:.2f}s')
        
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, 'f'Accuracy: {epoch_acc:.2f}%, Time: {epoch_time:.2f}s')
        
        if not distributed_training or (distributed_training and global_rank == 0):
            if distributed_training:
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'train_accuracy': epoch_acc,
                'train_accuracies_history': train_accuracies,
                'global_step': global_step,
                'total_train_time': total_train_time
            }
            
            checkpoint_path = get_weights_file_path(model_folder, model_basename, epoch)
            torch.save(checkpoint, checkpoint_path)
            
            latest_path = os.path.join(model_folder, 'latest.pt')
            torch.save(checkpoint, latest_path)
            
            print(f'Saved checkpoint to {checkpoint_path}')
    
    if not distributed_training or (distributed_training and global_rank == 0):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies)
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_folder, 'training_metrics.png'))
        plt.close()
        
        print(f'Training completed in {total_train_time:.2f} seconds')
        
        # Log total training time
        logging.info(f'Training completed in {total_train_time:.2f} seconds')
        
        print('Evaluating on test set...')
        confusion_matrix = evaluate_model(model, testloader, device, num_classes)
        val_accuracy = log_metrics(confusion_matrix, cifar10_classes)
        
        print('\nConfusion Matrix:')
        print('----------------')
        print('Predicted →')
        print('Actual ↓')
        print('      ' + ''.join([f'{classes[i]:<7}' for i in range(num_classes)]))
        for i in range(num_classes):
            print(f'{classes[i]:<6}' + ''.join([f'{confusion_matrix[i, j].item():7d}' for j in range(num_classes)]))
    
    total_time = time.time() - start_time
    print(f'Total training time: {total_time:.2f} seconds')
    print(f'Training time (excluding setup): {total_train_time:.2f} seconds')
    
    if not distributed_training or (distributed_training and global_rank == 0):
        logging.info(f'Total training time: {total_time:.2f} seconds')
        logging.info(f'Training time (excluding setup): {total_train_time:.2f} seconds')
    
    return total_train_time

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    config = get_default_config()
    
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    global_rank = int(os.environ.get('RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1))
    

    distributed_training = local_rank != -1 and global_rank != -1
    
    if global_rank <= 0:  # Only download on the first process
        print(" Checking/Downloading CIFAR10 dataset...")
        torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transforms.ToTensor())
        print("Dataset is ready!")
    
    if distributed_training:
        print(f'Starting distributed training with:')
        print(f' Global rank: {global_rank}')
        print(f' Local rank: {local_rank}')
        print(f' World size: {world_size}')
        
        distributed_init_success = False
        try:
            init_process_group(
                backend='nccl',
                init_method='env://',
                timeout=timedelta(seconds=60))
            distributed_init_success = True
            print(f' Process group initialized successfully!')
            torch.cuda.set_device(local_rank)
        except Exception as e:
            print(f'Failed to initialize distributed training: {str(e)}')
            distributed_training = False  # Fall back to single GPU mode
            local_rank = 0
            global_rank = 0
            torch.cuda.set_device(0)
            print('Falling back to single GPU mode')
    else:
        print('Running in single GPU mode')
        local_rank = 0
        global_rank = 0
        torch.cuda.set_device(0)
    
    try:
        if not os.path.exists(config['model_folder']):
            os.makedirs(config['model_folder'])
    except PermissionError:
        print(f"No permission to create directory: {config['model_folder']}")
        print('Falling back to local directory "./weights"')
        config['model_folder'] = "./weights"
        os.makedirs(config['model_folder'], exist_ok=True)
    
    # Wait for all processes
    if distributed_training:
        torch.distributed.barrier()
        
    start_time = time.time()
    
    print(f'Starting Vision Transformer training with:')
    print(f" Number of epochs: {config['num_epochs']}")
    print(f" Batch size: {config['batch_size']}")
    print(f" Learning rate: {config['learning_rate']}")
    print(f" Model architecture: ViT-{config['embed_dim']}")
    print(f" Image size: {config['image_size']}")
    print(f" Patch size: {config['patch_size']}")
    print(f" Num layers: {config['num_layers']}")
    print(f" Num heads: {config['num_heads']}")
    print(f" Device: cuda:{local_rank}" if distributed_training else " Device: cuda:0")
    if distributed_training:
        print(f" Distributed training enabled with {world_size} processes")
        
    total_train_time = train_model(
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        model_folder=config['model_folder'],
        model_basename=config['model_basename'],
        preload=config['preload'],
        tokenizer_file=config['tokenizer_file'],
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        forward_expansion=config['forward_expansion'],
        dropout=config['dropout'],
        stochastic_depth=config['stochastic_depth'],
        local_rank=local_rank,
        global_rank=global_rank,
        distributed_training=distributed_training)
    total_time = time.time() - start_time
    print(f'Total training time: {total_time:.2f} seconds')
    print(f'Training time (excluding setup): {total_train_time:.2f} seconds')
    
    if distributed_training:
        try:
            # Check if the process group exists before destroying it
            if torch.distributed.is_initialized():
                destroy_process_group()
                print('Cleaned up distributed training')
            else:
                print('Process group was not initialized, skipping cleanup')
        except Exception as e:
            print(f'Error during distributed cleanup: {e}') 