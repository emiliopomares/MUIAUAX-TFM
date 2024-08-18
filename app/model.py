import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.quantization import fuse_modules
import torch.quantization
import torchvision.transforms as transforms
import torch.nn.functional as F

import os

import cv2

import numpy as np

import random

import time

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

from data_loader import CustomDataLoader

checkpoint_dir = "../checkpoints"
checkpoint_fn = "model_checkpoint_running_mVoxelNet_v4_2_correct_rmisty-meadow-32_tl0.009472012519836426_tm9287_vm9287_256167.16s.pth" # Your checkpoint here
TRAIN_DATASET_PATH = "/media/emilio/2TBDrive/robovision_train" # Point to your test data
TEST_DATASET_PATH = "/media/emilio/2TBDrive/robovision_test" # Point to your test data

IMG_SIZE = 256
N_CHANNELS = 6
BATCH_SIZE = 32 # Let's stick to the classics

aug_params = {
    'contrast': 0.2,
    'brightness': 0.2,
    'saturation': 0.2,
    'hue': 0.2,
    'noise': 0.02
}

# We need the model definition here
def copy_inflate(input_tensor):
    """
    Inflates towards the 2nd axis by
    producing multiple copies of the
    0th-1st axis slice
    """
    # Get the shape of the input tensor
    batch_size, C, _, N = input_tensor.shape

    # Reshape the input tensor to add a singleton dimension at the end
    inflated_tensor = input_tensor.unsqueeze(-1)

    # Repeat the singleton dimension N times along the last axis
    inflated_tensor = inflated_tensor.expand(-1, -1, -1, -1, N)

    return inflated_tensor


class DoubleConv2D(nn.Module):
    """
    Double Convolution 2D
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv1(x)     
        x = self.bn1(x)     
        x = self.relu(x)       
        x = self.conv2(x)      
        x = self.bn2(x)     
        x = self.relu(x)       
        return x

        
class DoubleConv3D(nn.Module):
    """
    Double Convolution 3D
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class VoxelNet_v4(nn.Module):
    """
    VoxelNet, U-Net inspired network which will
    map a 6 channel, stereo RGB image into a
    3d 64x64x64 occupation probability map
    """
    def __init__(self, in_channels, out_channels, steps=5):
        super(VoxelNet_v4, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.residual_connections = nn.ModuleList()
        self.steps = steps
        self.relu = nn.ReLU()
        self.mode = "occupancy"

        features = [2 ** (i+4) for i in range(steps)]

        # Encoder
        for feature in features:
            #print(f" addinf out feature {feature}")
            self.encoder.append(
                    DoubleConv2D(in_channels, feature),
            )
            in_channels = feature

        out_ch = in_channels
        #print("We start with out_channels: ", out_channels)
        
        # Decoder
        for i in range(1,6):
            # Let's try the last layer trick
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose3d(out_ch*2, out_ch, kernel_size=2 if i<5 else 3, stride=2 if i<5 else 3),
                    DoubleConv3D(out_ch, out_ch//2)
                )
            )
            out_ch = out_ch//2
        
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.final_conv_1 = nn.Conv3d(64, 32, kernel_size=1)
        self.final_conv_2 = nn.Conv3d(32, 16, kernel_size=1)
        self.final_conv_3 = nn.Conv3d(16, 4, kernel_size=1)
        self.final_conv_4 = nn.Conv3d(4, out_channels, kernel_size=1)

        self.pose_dense_1 = nn.Linear(2 ** (self.steps+4-1), 256)
        self.pose_dense_2 = nn.Linear(256, 64)
        self.pose_dense_3 = nn.Linear(64, 16)
        self.pose_dense_4 = nn.Linear(16, 7)

    def set_mode(mode):
        freeze_pose = True
        if mode=='occupancy':
            freeze_pose = True
        elif mode=='pose':
            freeze_pose = False
        else:
            raise ValueException("Unknown mode")
        # Freeze layer1 and layer2
        for param in self.encoder.parameters():
            param.requires_grad = freeze_pose
        for param in self.decoder.parameters():
            param.requires_grad = freeze_pose
        for param in self.residual_connections.parameters():
            param.requires_grad = freeze_pose
        for param in self.final_conv_1.parameters():
            param.requires_grad = freeze_pose
        for param in self.final_conv_2.parameters():
            param.requires_grad = freeze_pose
        for param in self.final_conv_3.parameters():
            param.requires_grad = freeze_pose
        for param in self.final_conv_4.parameters():
            param.requires_grad = freeze_pose
        for param in self.pose_dense_1.parameters():
            param.requires_grad = not freeze_pose
        for param in self.pose_dense_2.parameters():
            param.requires_grad = not freeze_pose
        for param in self.pose_dense_3.parameters():
            param.requires_grad = not freeze_pose
        for param in self.pose_dense_4.parameters():
            param.requires_grad = not freeze_pose

    def forward(self, x):
        
        encoder_outputs = []

        # Encoder
        for module in self.encoder:
            x = module(x)
            x = self.pool(x)
            encoder_outputs.append(x)
        
        if self.mode == "pose":
            x_pose = x[:,:,0,0]
            x_pose = self.pose_dense_1(x_pose)
            x = self.relu(x_pose)
            x_pose = self.pose_dense_2(x_pose)
            x = self.relu(x_pose)
            x_pose = self.pose_dense_3(x_pose)
            x = self.relu(x_pose)
            x_pose = self.pose_dense_4(x_pose)
            return x_pose
        
        x = x.unsqueeze(-1)
        
        # Decoder
        for i in range(1,6):
            residual_connection = encoder_outputs[-i]
            inflated_connection = copy_inflate(residual_connection)
            x = torch.cat([x, inflated_connection], dim=1)
            x = self.decoder[i-1](x)
        
        x = self.final_conv_1(x)
        x = self.relu(x)
        x = self.final_conv_2(x)
        x = self.relu(x)
        x = self.final_conv_3(x)
        x = self.relu(x)
        x = self.final_conv_4(x).squeeze(dim=1)

        return x
    
def get_model():
    """Gets the final inference model"""
    model = VoxelNet_v4(in_channels=6, out_channels=1, steps=8).to("cpu")
    checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_fn))
    # Load the model's state_dict from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

test_dataloader = CustomDataLoader(TEST_DATASET_PATH, additional_param=aug_params)
test_loader = DataLoader(test_dataloader, batch_size=BATCH_SIZE)

def get_test_batch():
    test_batch = None
    for batch in test_loader:
        test_batch = batch
        break
    return test_batch