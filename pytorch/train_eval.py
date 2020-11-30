
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
import numpy as np
import time
import os
import copy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, models, transforms


def get_data(data_dir, batch_size):
    """Creates training and validation splits."""

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data = {
        'train':
            datasets.ImageFolder(root=os.path.join(data_dir, 'train'),
                                 transform=data_transforms['train']),
        'val':
            datasets.ImageFolder(root=os.path.join(data_dir, 'val'),
                                 transform=data_transforms['val'])
    }

    datasizes = {
        'train': len(data['train']),
        'val': len(data['val'])
    }

    dataloaders = {
        'train':
            DataLoader(data['train'], batch_size=batch_size, shuffle=True),
        'val':
            DataLoader(data['val'], batch_size=batch_size, shuffle=True)
    }

    class_names = data['train'].classes

    return dataloaders, datasizes, class_names


def get_model(num_layers, dropout_ratio, num_classes):
    """
    Creates a convolution net using ResNet50 trunk and
    a custom head.
    """

    # Create the ResNet50 trunk
    model = models.resnet50(pretrained=True)

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


def train_eval(device, train_dataloader, valid_dataloader, 
               model, criterion, optimizer, num_epochs, log_dir='/tmp'):
    """
    Trains and evaluates a model.
    """

    since = time.time()
    writer = SummaryWriter(log_dir)

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Training phase
        model.train()  
        num_examples = 0
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate loss and accuracy
            num_examples += inputs.shape(0)
            running_loss += loss.item() * inputs.size(0) 
            corrects = torch.sum(torch.eq(torch.max(outputs, dim=1)[1], labels))
            running_corrects += corrects.item() 

        epoch_training_loss = running_loss / num_examples 
        epoch_training_acc = running_corrects.double() / num_examples

        # Validation phase
        model.eval()   
        num_examples = 0
        running_loss = 0.0
        running_corrects = 0

        for inputes, labels in valid_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Accumulate loss and accuracy
            num_examples += inputs.shape(0)
            running_loss += loss.item() * inputs.size(0)
            corrects = torch.sum(torch.eq(torch.max(outputs, dim=1)[1], labels)) 
            running_corrects += corrects.item()
 
        epoch_valid_loss = running_loss / num_examples 
        epoch_valid_acc = running_corrects.double() / num_examples

        if epoch_valid_acc > best_acc:
            best_acc = epoch_valid_acc
            best_model_wts = copy.deepcopy(model.state_dict)
           
        # Log epoch metrics
        print('Training Loss: {:.4f} Acc: {:.4f}'.format(
              epoch_training_loss, epoch_training_acc))
        print('Validation Loss: {:.4f} Acc: {:.4f}'.format(
              epoch_valid_loss, epoch_valid))

        # Write loss and accuracy to TensorBoard
        writer.add_scalar('Loss/training', epoch_training_loss, epoch)
        writer.add_scalar('Acc/training', epoch_training_acc, epoch)
        writer.add_scalar('Loss/validation', epoch_validation_loss, epoch)
        writer.add_scalar('Acc/validation', epoch_validation_acc, epoch)
        writer.flush()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    writer.close()
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def get_args():
    """
    Returns parsed command line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=20,
        help='number of times to go through the data, default=20')
    parser.add_argument(
        '--batch-size',
        default=128,
        type=int,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--num_layers',
        default=32,
        type=int,
        help='number of hidden layers in the classification head , default=128')
    parser.add_argument(
        '--droput_ratio',
        default=0.5,
        type=float,
        help='dropout ration in the classification head , default=128')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(args)