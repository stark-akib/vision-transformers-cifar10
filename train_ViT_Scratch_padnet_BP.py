# -*- coding: utf-8 -*-
'''
Train CIFAR10/CIFAR100 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47
modified to support CIFAR100
'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import random

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time
import json
import pickle
import cv2
from art.attacks.evasion import ProjectedGradientDescent, CarliniLInfMethod
from art.estimators.classification import PyTorchClassifier
from sklearn.metrics import accuracy_score

from models import *
from utils import progress_bar
from randomaug import RandAugment
from models.vit import ViT
from models.convmixer import ConvMixer
from models.mobilevit import mobilevit_xxs


# Add after the imports
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        # Initialize state for all parameters and ensure rho is in param_groups
        for group in self.param_groups:
            group['rho'] = rho
            for p in group["params"]:
                self.state[p] = dict()

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        if grad_norm == 0:
            # Initialize state even if we return early
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    self.state[p]["old_p"] = p.data.clone()
            return
            
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                # Avoid in-place operation
                p.data = p.data + e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                if "old_p" in self.state[p]:  # Check if state exists
                    p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and p.grad.norm(p=2) > 0:
                    grads.append(p.grad.norm(p=2).to(shared_device))
        if not grads:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(grads), p=2)
    

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')  # Increased from 1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_false', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to use (cifar10, cifar100, or imagenet100)')
parser.add_argument('--train_dir', type=str, required=True, help='path to ImageNet-100 training data')
parser.add_argument('--val_dir', type=str, required=True, help='path to ImageNet-100 validation data')
parser.add_argument('--rho', default=0.05, type=float, help='SAM optimizer rho parameter')  # Back to 0.05
parser.add_argument('--alpha', default=0.00001, type=float, help='Gradient regularization weight')  # Reduced from 10.0
parser.add_argument('--beta', default=1.0, type=float, help='Adversarial example weight')  # Reduced from 1.0
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='directory to save model checkpoints')

args = parser.parse_args()

# take in args
usewandb = ~args.nowandb
if usewandb:
    import wandb
    watermark = "{}_lr{}_{}".format(args.net, args.lr, args.dataset)
    wandb.init(project="cifar-challenge",
            name=watermark)
    wandb.config.update(args)

bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
aug = args.noaug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.net=="vit_timm":
    size = 384
elif args.dataset == 'imagenet100':
    size = 224  # ImageNet standard size
else:
    size = imsize

# Set up normalization based on the dataset
if args.dataset == 'cifar10':
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    num_classes = 11  # 10 original classes + 1 padding class
    dataset_class = torchvision.datasets.CIFAR10
elif args.dataset == 'cifar100':
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    num_classes = 101  # 100 original classes + 1 padding class
    dataset_class = torchvision.datasets.CIFAR100
elif args.dataset == 'imagenet100':
    mean = (0.485, 0.456, 0.406)  # ImageNet mean
    std = (0.229, 0.224, 0.225)   # ImageNet std
    num_classes = 101  # 100 original classes + 1 padding class
    # We'll use ImageFolder for ImageNet-100
    dataset_class = torchvision.datasets.ImageFolder
else:
    raise ValueError("Dataset must be either 'cifar10', 'cifar100', or 'imagenet100'")

# Modify transforms for ImageNet-100
if args.dataset == 'imagenet100':
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),  # ImageNet standard size
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
else:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

# Add RandAugment with N, M(hyperparameter)
if aug:  
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))

# Prepare dataset
if args.dataset == 'imagenet100':
    # For ImageNet-100, we expect the data to be in a specific directory structure
    # train_dir and val_dir should be provided as arguments
    trainset = dataset_class(root=args.train_dir, transform=transform_train)
    testset = dataset_class(root=args.val_dir, transform=transform_test)
else:
    trainset = dataset_class(root='./data', train=True, download=True, transform=transform_train)
    testset = dataset_class(root='./data', train=False, download=True, transform=transform_test)

# Adjust batch size for ImageNet-100
if args.dataset == 'imagenet100':
    bs = min(bs, 256)  # Limit batch size for ImageNet-100 due to larger images

trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

# Set up class names based on the dataset
if args.dataset == 'cifar10':
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
elif args.dataset == 'cifar100':
    # CIFAR100 has 100 classes, so we don't list them all here
    classes = None
elif args.dataset == 'imagenet100':
    # Load ImageNet-100 class labels from JSON
    with open('Labels.json', 'r') as f:
        imagenet_labels = json.load(f)
    # Convert to list of class names, sorted by class ID
    classes = [imagenet_labels[class_id] for class_id in sorted(imagenet_labels.keys())]
    print("ImageNet-100 Classes:")
    for i, class_name in enumerate(classes):
        print(f"{i}: {class_name}")

# Model factory..
print('==> Building model..')
# net = VGG('VGG19')
if args.net=='res18':
    net = ResNet18(num_classes=num_classes)
elif args.net=='vgg':
    net = VGG('VGG19', num_classes=num_classes)
elif args.net=='res34':
    net = ResNet34(num_classes=num_classes)
elif args.net=='res50':
    net = ResNet50(num_classes=num_classes)
elif args.net=='res101':
    net = ResNet101(num_classes=num_classes)
elif args.net=="convmixer":
    # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
    net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=num_classes)
elif args.net=="mlpmixer":
    from models.mlpmixer import MLPMixer
    net = MLPMixer(
    image_size = 32,
    channels = 3,
    patch_size = args.patch,
    dim = 512,
    depth = 6,
    num_classes = num_classes
)
elif args.net=="vit_small":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = num_classes,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_tiny":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = num_classes,
    dim = int(args.dimhead),
    depth = 4,
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="simplevit":
    from models.simplevit import SimpleViT
    net = SimpleViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = num_classes,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512
)
elif args.net=="vit":
    # ViT for cifar10/100/imagenet100
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = num_classes,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_timm":
    import timm
    net = timm.create_model("vit_base_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, num_classes)
elif args.net=="cait":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = num_classes,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="cait_small":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = num_classes,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="swin":
    from models.swin import swin_t
    net = swin_t(window_size=args.patch,
                num_classes=num_classes,
                downscaling_factors=(2,2,2,1))
elif args.net=="mobilevit":
    net = mobilevit_xxs(size, num_classes)
else:
    raise ValueError(f"'{args.net}' is not a valid model")

# For Multi-GPU
if 'cuda' in device:
    print(device)
    if args.dp:
        print("using data parallel")
        net = torch.nn.DataParallel(net) # make parallel
        cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.checkpoint_dir), f'Error: checkpoint directory {args.checkpoint_dir} not found!'
    # Find the latest checkpoint file
    checkpoint_files = [f for f in os.listdir(args.checkpoint_dir) if f.startswith(f'{args.net}-{args.dataset}-{args.patch}')]
    if not checkpoint_files:
        raise FileNotFoundError(f'No checkpoint files found in {args.checkpoint_dir}')
    # Sort by accuracy in filename and get the latest
    checkpoint_files.sort(key=lambda x: float(x.split('acc')[-1].split('.t7')[0]), reverse=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_files[0])
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()

base_optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)  # Added weight decay
optimizer = SAM(net.parameters(), base_optimizer, rho=args.rho)

# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# Add the padding sample generation function
def gen_padding_images(X_batch, y_batch, batch_size):
    """
    Generate padding samples by interpolating between pairs of images and adding noise
    Args:
        X_batch: Input images tensor
        y_batch: Input labels tensor
        batch_size: Size of the batch
    Returns:
        X_pad: Padded images
        y_pad: Labels for padded images (all set to num_classes-1)
    """
    # Convert to numpy for easier manipulation
    X_np = X_batch.cpu().numpy()
    y_np = y_batch.cpu().numpy()
    
    # Initialize arrays for padded samples
    X_pad = [X_np[0]]  # Start with first image
    y_pad = [y_np[0]]  # Start with first label
    
    # Generate padding samples
    for i in range(len(X_np) - 1):
        # Get two consecutive images
        s = X_np[i]
        t = X_np[i + 1]
        
        # Calculate mean
        m = np.mean([s, t], axis=0)
        
        # Generate weighted average padding sample
        alpha = 0.2
        wa1 = alpha * s + (1 - alpha) * t
        
        # Add noise to weighted average
        var = random.uniform(0.01, 0.1)
        rand = np.random.normal(0, var, wa1.shape)
        wa1 = np.add(rand, wa1)
        wa1 = np.clip(wa1, 0, 1)
        
        # Add weighted average sample
        X_pad.append(wa1)
        y_pad.append(num_classes - 1)  # Label as padding class
        
        # Generate mean-based noise sample
        var = random.uniform(0.01, 0.1)
        rand = np.random.normal(0, var, m.shape)
        rand_pert = np.add(rand, m)
        rand_pert = np.clip(rand_pert, 0, 1)
        
        # Add mean-based noise sample
        X_pad.append(rand_pert)
        y_pad.append(num_classes - 1)  # Label as padding class
    
    # Add one more benign sample to match size
    X_pad.append(X_np[1])
    y_pad.append(y_np[1])
    
    # Convert to numpy arrays
    X_pad = np.array(X_pad)
    y_pad = np.array(y_pad)
    
    # Convert back to tensors
    X_pad = torch.FloatTensor(X_pad).to(X_batch.device)
    y_pad = torch.LongTensor(y_pad).to(y_batch.device)
    
    return X_pad, y_pad

# Add these functions before the train() function
def random_batch(X, y, batch_size):
    idx = np.random.randint(X.shape[0], size=batch_size)
    return X[idx], y[idx]


def early_stopping(model, X_train, y_train, X_test, y_test, name):
    num_test_samples = 30
    
    # Create ART classifier
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=101,  # 100 classes + 1 padding class
        clip_values=(0, 1)
    )
    
    # Generate adversarial examples
    attack = CarliniLInfMethod(classifier=classifier, confidence=0, max_iter=10, targeted=False)
    x_test_adv = attack.generate(x=X_test[:num_test_samples])
    
    # Get predictions
    predictions = classifier.predict(x_test_adv)
    y_pred = np.argmax(predictions, axis=1)
    
    # Calculate success rate
    success = 0.0
    for i in range(y_test[:num_test_samples].shape[0]):
        if y_pred[i] != y_test[i] and y_pred[i] != 100:  # 100 is padding class
            success += 1.0
    sr = success/num_test_samples
    
    # Calculate accuracies
    model.eval()
    with torch.no_grad():
        train_pred = model(torch.FloatTensor(X_train).to(device))
        train_acc = accuracy_score(y_train, train_pred.argmax(dim=1).cpu().numpy())
        
        test_pred = model(torch.FloatTensor(X_test).to(device))
        test_acc = accuracy_score(y_test, test_pred.argmax(dim=1).cpu().numpy())
    
    # Log results
    print(f"Attack success: {sr}")
    print(f"Train Accuracy: {train_acc}")
    print(f"Test Accuracy: {test_acc}")
    
    with open(f"{name}.txt", "a") as f:
        f.write(f"Attack success: {sr}\n")
        f.write(f"Train Accuracy: {train_acc}\n")
        f.write(f"Test Accuracy: {test_acc}\n")
    
    return sr, test_acc


# Modify the training function to use standard loss
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    # Hyperparameters
    count = 0          # For learning rate scheduling
    best_acc = 0       # For early stopping
    best_sr = 1.0      # For attack success rate
    
    # Warmup learning rate for first few epochs
    if epoch < 5:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * (epoch + 1) / 5
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Generate padding samples
        X_pad, y_pad = gen_padding_images(inputs, targets, inputs.size(0))
        
        # Combine original and padding samples
        combined_inputs = torch.cat([inputs, X_pad], dim=0)
        combined_targets = torch.cat([targets, y_pad], dim=0)
        
        # Training step with SAM
        def closure():
            # Zero out existing gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = net(combined_inputs)
            loss = criterion(outputs, combined_targets)
            
            # Compute gradients
            loss.backward()
            return loss
        
        # SAM optimizer step
        optimizer.step(closure)
        
        # Update statistics
        with torch.no_grad():
            outputs = net(combined_inputs)
            loss = criterion(outputs, combined_targets)
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += combined_targets.size(0)
            correct += predicted.eq(combined_targets).sum().item()
        
        # Print progress
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        # Early stopping and learning rate scheduling
        if batch_idx % 150 == 0:
            # Evaluate on validation set
            val_loss, val_acc = test(epoch)
            
            if val_acc > best_acc:
                best_acc = val_acc
                count = 0
                # Save best model
                state = {
                    'net': net.state_dict(),
                    'acc': val_acc,
                    'epoch': epoch,
                }
                acc_str = f"{val_acc:.2f}".replace('.', '_')
                checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.net}-{args.dataset}-{args.patch}-acc{acc_str}.t7')
                torch.save(state, checkpoint_path)
            else:
                count += 1
            
            # Learning rate scheduling
            if count == 10:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                count = 0
    
    return train_loss/(batch_idx+1)

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        # Format accuracy to 2 decimal places for filename
        acc_str = f"{acc:.2f}".replace('.', '_')
        checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.net}-{args.dataset}-{args.patch}-acc{acc_str}.t7')
        torch.save(state, checkpoint_path)
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    log_file = f'log/log_{args.net}_{args.dataset}_patch{args.patch}.txt'
    with open(log_file, 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []

if usewandb:
    wandb.watch(net)
    
net.cuda()
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    
    scheduler.step(epoch-1) # step cosine scheduling
    
    list_loss.append(val_loss)
    list_acc.append(acc)
    
    # Log training..
    if usewandb:
        wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
        "epoch_time": time.time()-start})

    # Write out csv..
    csv_file = f'log/log_{args.net}_{args.dataset}_patch{args.patch}.csv'
    with open(csv_file, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    print(list_loss)

# writeout wandb
if usewandb:
    wandb.save("wandb_{}_{}.h5".format(args.net, args.dataset))



