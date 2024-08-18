from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import cv2
import random
import numpy as np

import os

from tensor_permutation import permute_target_tensor, expand_output
from model_input import make_model_input

import struct

IMG_SIZE = 256
N_CHANNELS = 6
BATCH_SIZE = 32 # Let's stick to the classics

def load_target_datapoint(file_path, dataset_path=""):
    # Define the format string for reading the binary data
    format_string = "<3f4f"  # 3 floats (position), 4 floats (quaternion)
    # Calculate the size of the bytes for occupation data
    occupation_size = 37 * 25 * 18
    format_string += str(occupation_size) + "s"  # Occupation data

    with open(os.path.join(dataset_path, file_path), "rb") as file:
        # Read the binary data
        data = file.read(struct.calcsize(format_string))
        # Unpack the binary data according to the format string
        unpacked_data = struct.unpack(format_string, data)

        # Extract position, rotation, and occupation data
        position = unpacked_data[:3]
        rotation = unpacked_data[3:7]
        # Convert occupation data to array of numbers
        occupation = struct.unpack(str(occupation_size) + "B", unpacked_data[7])

        return position, rotation, occupation
    
def load_stereo_image(index=0, 
                      dataset_path="./", 
                      model_size=(IMG_SIZE, IMG_SIZE), 
                      l_path=None, 
                      r_path=None,
                      plot=False
                     ):
    dx = -110
    dy = -6
    l_file = l_path if l_path is not None else os.path.join(dataset_path, f"{index}L.png")
    r_file = r_path if r_path is not None else os.path.join(dataset_path, f"{index}R.png")
    l_img = cv2.cvtColor(cv2.imread(l_file), cv2.COLOR_BGR2RGB)
    l_img = cv2.warpAffine(l_img, np.float32([[1, 0, dx], [0, 1, dy]]), (l_img.shape[1], l_img.shape[0]))
    r_img = cv2.cvtColor(cv2.imread(r_file), cv2.COLOR_BGR2RGB)
    l_img = cv2.resize(l_img[0:714, 0:1170], model_size, interpolation=cv2.INTER_AREA)
    r_img = cv2.resize(r_img[0:714, 0:1170], model_size, interpolation=cv2.INTER_AREA)
    blended_image = cv2.addWeighted(l_img, 0.5, r_img, 0.5, 0)
    
    if plot:
        # Plot the blended image
        plt.title(f"sample {index} ({model_size[0]}x{model_size[0]} 3+3 channel, aspect corrected)")
        plt.imshow(blended_image, aspect=1/1.4)
        
    return torch.tensor(l_img/255.0), torch.tensor(r_img/255.0)

class CustomDataLoader(Dataset):
    """Class to load data from files in disk and convert
    them to tensors on the fly"""
    def __init__(self, data_dir, transform=None, additional_param=None):
        self.data_dir = data_dir
        self.transform = transform
        self.l_img_list = []
        self.r_img_list = []
        self.gt_list = []
        self.additional_param = additional_param
        for file in sorted(os.listdir(data_dir)):
            if file.endswith("L.png"):
                self.l_img_list.append(os.path.join(data_dir, file))
            elif file.endswith("R.png"):
                self.r_img_list.append(os.path.join(data_dir, file))
            elif file.endswith("T.bin"):
                self.gt_list.append(os.path.join(data_dir, file))   
        
    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):        
        l, r = load_stereo_image(l_path=self.l_img_list[idx], r_path=self.r_img_list[idx]);
        t, q, o = load_target_datapoint(self.gt_list[idx])

        # Apply transformations if specified
        if self.transform:
            l = self.transform(l)
            r = self.transform(r)

        l = l.permute(2, 0, 1)
        r = r.permute(2, 0, 1)

        augment_contrast=self.additional_param['contrast'] if (self.additional_param is not None and 'contrast' in self.additional_param) else 0
        augment_saturation=self.additional_param['saturation'] if (self.additional_param is not None and 'saturation' in self.additional_param) else 0
        augment_brightness=self.additional_param['brightness'] if (self.additional_param is not None and 'brightness' in self.additional_param) else 0
        augment_hue=self.additional_param['hue'] if (self.additional_param is not None and 'hue' in self.additional_param) else 0
        augment_noise=self.additional_param['noise'] if (self.additional_param is not None and 'noise' in self.additional_param) else 0

        # Augment each image separately, as l and r cameras are independent

        l = transforms.functional.adjust_brightness(l, 1 + (random.random() - 0.5) * 2 * augment_brightness)
        l = transforms.functional.adjust_contrast(l, 1 + (random.random() - 0.5) * 2 * augment_contrast)
        l = transforms.functional.adjust_saturation(l, 1 + (random.random() - 0.5) * 2 * augment_saturation)
        l = transforms.functional.adjust_hue(l, (random.random() - 0.5) * 2 * augment_hue)
        noise = np.random.normal(0, augment_noise, l.shape).astype(np.float32)
        l = l + noise
        l = torch.clip(l, 0, 1)

        r = transforms.functional.adjust_brightness(r, 1 + (random.random() - 0.5) * 2 * augment_brightness)
        r = transforms.functional.adjust_contrast(r, 1 + (random.random() - 0.5) * 2 * augment_contrast)
        r = transforms.functional.adjust_saturation(r, 1 + (random.random() - 0.5) * 2 * augment_saturation)
        r = transforms.functional.adjust_hue(r, (random.random() - 0.5) * 2 * augment_hue)
        noise = np.random.normal(0, augment_noise, r.shape).astype(np.float32)
        r = r + noise
        r = torch.clip(r, 0, 1)
            
        X = make_model_input(l, r, permute=False)
            
        occupation = torch.tensor(np.array(o, dtype='float32').reshape(37,25,18))
        occupation = expand_output(occupation)
        y = permute_target_tensor(occupation)

        return X, y