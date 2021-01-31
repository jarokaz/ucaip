
# Copyright 2020 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import hypertune
import numpy as np
import time
import os
import copy
import matplotlib.pyplot as plt
import zipfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, models, transforms


DEFAULT_ROOT = '/tmp'

def get_catsanddogs(root):
    """
    Creates training and validation Datasets based on images
    of cats and dogs from 
    https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip.
    """
    
    # Download and extract the images
    source_url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    local_filename = source_url.split('/')[-1]
    datasets.utils.download_url(source_url, root, )
    path_to_zip = os.path.join(root, local_filename)
    with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
        zip_ref.extractall(root)
    
    
    # Create datasets
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    train_dataset = datasets.ImageFolder(
        root=os.path.join(path_to_zip[:-4], 'train'),
        transform=train_transforms)
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(path_to_zip[:-4], 'validation'),
        transform=val_transforms
    )
    
    return train_dataset, val_dataset
    


def get_model(num_layers, dropout_ratio, num_classes):
    """
    Creates a convolution net using ResNet50 trunk and
    a custom head.
    """

    # Create the ResNet50 trunk
    model = models.resnet18(pretrained=True)

    # Get the number of input features to the default head
    num_features = model.fc.in_features

    # Freeze trunk weights
    for param in model.parameters():
        param.requires_grad = False

    # Define the new head
    head = nn.Sequential(nn.Linear(num_features, num_layers),
                         nn.ReLU(),
                         nn.Dropout(dropout_ratio),
                         nn.Linear(num_layers, num_classes))

    # Replace the head
    model.fc = head

    return model


def train_eval(device, model, train_dataloader, valid_dataloader,
               criterion, optimizer, scheduler, num_epochs, writer=None):
    """
    Trains and evaluates a model.
    """
    since = time.time()

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    hpt = hypertune.HyperTune()

    for epoch in range(1, num_epochs+1):

        # Training phase
        model.train()
        num_train_examples = 0
        train_loss = 0.0

        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            num_train_examples += inputs.size(0)
            train_loss += loss.item() * inputs.size(0)
        scheduler.step()

        # Validation phase
        model.eval()
        num_val_examples = 0
        val_loss = 0
        val_corrects = 0

        for inputs, labels in valid_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            num_val_examples += inputs.size(0)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(torch.eq(torch.max(outputs, 1)
                                               [1], labels))

        # Log epoch metrics
        train_loss = train_loss / num_train_examples
        val_loss = val_loss / num_val_examples
        val_acc = val_corrects.double() / num_val_examples

        print('Epoch: {}/{}, Training loss: {:.3f}, Validation loss: {:.3f}, Validation accuracy: {:.3f}'.format(
              epoch, num_epochs, train_loss, val_loss, val_acc))

        # Write to Tensorboard
        if writer:
            writer.add_scalars(
                'Loss', {'training': train_loss, 'validation': val_loss}, epoch)
            writer.add_scalar('Validation accuracy', val_acc, epoch)
            writer.flush()
            
        # Report to HyperTune
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='accuracy',
            metric_value=val_acc,
            global_step=epoch
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc


def get_args():
    """
    Returns parsed command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=20,
        help='number of times to go through the data, default=20')
    parser.add_argument(
        '--batch-size',
        default=32,
        type=int,
        help='number of records to read during each training step, default=32')
    parser.add_argument(
        '--num-layers',
        default=64,
        type=int,
        help='number of hidden layers in the classification head , default=64')
    parser.add_argument(
        '--dropout-ratio',
        default=0.5,
        type=float,
        help='dropout ration in the classification head , default=128')
    parser.add_argument(
        '--step-size',
        default=7,
        type=int,
        help='step size of LR scheduler')
    parser.add_argument(
        '--log-dir',
        type=str,
        default='/tmp',
        help='directory for TensorBoard logs')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    
    # Parse command line arguments
    args = get_args()
    
    # Create train and validation dataloaders
    train_dataset, val_dataset = get_catsanddogs(DEFAULT_ROOT)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    class_names = train_dataset.classes
    
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('-' * 10)
    print(f'Training on device: {device}')

    # Configure training
    model = get_model(args.num_layers, args.dropout_ratio, len(class_names))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=0.1)

    # Set location for the TensorBoard logs
    if 'AIP_TENSORBOARD_LOG_DIR' in os.environ:
        log_dir = os.environ['AIP_TENSORBOARD_LOG_DIR']
    else:
        log_dir = args.log_dir

    with SummaryWriter(log_dir) as writer:
        # Add sample normalized images to Tensorboard
        images, _ = iter(train_dataloader).next()
        img_grid = torchvision.utils.make_grid(images)
        writer.add_image('Example images', img_grid)
        # Add graph to Tensorboard
        writer.add_graph(model, images)
        trained_model, accuracy = train_eval(device, model, train_dataloader, val_dataloader,
                                             criterion, optimizer, scheduler, args.num_epochs, writer)

        # Add final results and hyperparams to Tensorboard
        writer.add_hparams({
            'batch_size': args.batch_size,
            'hidden_layers': args.num_layers,
            'dropout_ratio': args.dropout_ratio
        },
            {
            'hparam/accuracy': accuracy
        })
