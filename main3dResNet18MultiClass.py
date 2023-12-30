
#%%




#%%
import gc
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU ID
#%%
import numpy as np
import pandas as pd
import torch
from fastai.vision.augment import aug_transforms
from torch.utils.data import Dataset
from fastai.vision import *
from tqdm import tqdm

import sys
sys.path.append('/scratch/pterway/slivit/SLIViT')
from CKAGoogle import feature_space_linear_cka, cka, gram_linear, gram_rbf, cca
from vit3dclassification import ViViT
from auxiliaries import *

from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, \
    BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
import torchvision.transforms as transforms_resize
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
)
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from CKA import linear_CKA, kernel_CKA
#%%
class pil_contrast_strech(object):
    def __init__(self, low=2, high=98):
        self.low, self.high = low, high
    def __call__(self, img):
        # Contrast stretching
        img = np.array(img)
        plow, phigh = np.percentile(img, (self.low, self.high))
        return PIL.Image.fromarray(exposure.rescale_intensity(img, in_range=(plow, phigh)))
#%%    
transform_new = tf.Compose(
    [
        ToTensor(),
        tf.ToPILImage(),
        tf.Resize((112, 112)),
        pil_contrast_strech(),
        #RandomResizedCrop((256,256)),
        #RandomHorizontalFlip(),
        ToTensor(),
        #normalize,
        gray2rgb
    ])


#%%

from medmnist import NoduleMNIST3D
# from medmnist import OCTMNIST
from medmnist import OrganMNIST3D


# class NoduleMNISTDataset(Dataset):
#     def __init__(self, dataset= NoduleMNIST3D(root='/scratch/pterway/slivit/datasets',
#                                                split='train', download=True),
#                                                 transform=transform_new  
#                                             ):
#         super().__init__()
#         self.dataset = dataset 
#         self.transform = transform
#         self.resize_transform = transforms_resize.Resize((224, 224))


#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         image, label = self.dataset[idx]
#         image= image[0]
#         total_frames = image.shape[-1]
#         middle_frame = int(total_frames/2)
#         # imgs = [image[i] for i in range(image.shape[-1]) if i==middle_frame]
#         imgs = [image[i] for i in range(image.shape[-1]) if i<middle_frame]
#         # imgs = [image[i] for i in range(image.shape[-1])]
#         t_imgs = torch.cat([torch.FloatTensor(transform_new(torch.tensor(im))) for im in imgs], dim=1)
#         return t_imgs, label

#%%
class NoduleMNISTDataset(Dataset):
    def __init__(self, dataset= NoduleMNIST3D(root='/scratch/pterway/slivit/datasets',
                                               split='train', download=True),
                                                transform=transform_new  ):
        super()
        self.dataset=dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        #x=5

       # self.dataset[idx][0]

        image, label = self.dataset[idx]
        # # transform
        # # concatenate the image
        #imgs = [image[0, i] for i in range(image.shape[1])]
        # t_imgs = torch.cat([torch.FloatTensor(transform_new(im)) for im in imgs], dim=1)
        t_image = torch.zeros((28,3,224, 224))
        for j in range(28):
            
            t_image[j,:,:,:]=transform_new(image[0,j,:,:])

       #x=5
    
        #transformed_image = transform_new(t_imgs)
        #
        # return t_imgs, label
        #t_imgs = torch.cat([self.transform(im) for im in imgs], dim=1)

        
        # return torch.FloatTensor(t_image), torch.FloatTensor(label)

        return torch.FloatTensor(t_image), torch.squeeze(torch.LongTensor(label))

#%%
from torch.utils.data import DataLoader
# check if the dataloader works by sampling a batch
dataset = NoduleMNISTDataset()

dataset_validation = NoduleMNISTDataset(dataset= NoduleMNIST3D(root='/scratch/pterway/slivit/datasets',
                                               split='val', download=True),
                                                transform=transform_new  )
dataset_test = NoduleMNISTDataset(dataset= NoduleMNIST3D(root='/scratch/pterway/slivit/datasets',
                                               split='test', download=True),
                                                transform=transform_new  )

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

dataloader_validation = DataLoader(dataset_validation, batch_size=4, shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size=4, shuffle=False)

#%%
# load the first batch of the data


# Load the first batch of data
first_batch = next(iter(dataloader))

# Access the input data and labels
inputs, labels = first_batch

# Print the shape of the input data and labels
print("Input shape:", inputs.shape)
print("Labels shape:", labels.shape)
#%%
import torchvision.models as models

# Define the 3D ResNet model
class ResNet3D(nn.Module):
    def __init__(self, num_classes):
        super(ResNet3D, self).__init__()
        self.resnet = models.video.r3d_18(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
#%%    

# model = ViViT(224, 16, 1, 28).cuda()
num_classes = 2
model = ResNet3D(num_classes)
#%%    
model = model.cuda()
# model = ViViT(224, 16, 2, 28).cuda()

forward = model(inputs.cuda())

# criteria = nn.BCEWithLogitsLoss()
criteria = nn.CrossEntropyLoss()
loss = criteria(forward, labels.cuda())
loss.backward()

#%%
#ViT
# Use pretrained version
# Vary dim of feedforward
# Vary number layers in transformer

#%%
# ResNEt
# ResNet3D 18, ResNet3D 50, Pretrained Kinetics
#

#%%
# Create a grid search for dim [] and depth
dim_head = 64
depth = 4
model =  ViViT(224, 16, 1, 28, depth=depth, dim_head=dim_head).cuda()

#%%
# Define the loss function and optimizer
import torch.optim as optim
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
#%%
from tqdm import tqdm
#%%
# Initialize variables to track the best validation accuracy and corresponding model weights
best_accuracy = 0.0
best_model_weights = None

# Specify the number of epochs
num_epochs = 10

# Training loop
# Initialize lists to store training and validation losses
train_losses = []
val_losses = []

# Training loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Training") as pbar:
        for inputs, labels in pbar:
            inputs = inputs.cuda()
            labels = labels.float().cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({"Train Loss": train_loss / len(dataloader.dataset)})
        train_losses.append(train_loss / len(dataloader.dataset))

    # Validation phase
    model.eval()
    val_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        with tqdm(dataloader_validation, desc=f"Epoch {epoch+1}/{num_epochs} - Validation") as pbar:
            for inputs, labels in pbar:
                inputs = inputs.cuda()
                labels = labels.float().cuda()

                outputs = model(inputs)
                # predicted = torch.round(torch.sigmoid(outputs))
                predicted = torch.argmax(outputs, dim=1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                this_val_loss = criterion(outputs, labels)
                val_loss += this_val_loss.item() * inputs.size(0)
                pbar.set_postfix({"Val Loss": val_loss / len(dataloader_validation.dataset)})

            val_losses.append(val_loss / len(dataloader_validation.dataset))

    # Calculate validation accuracy
    validation_accuracy = total_correct / total_samples

    # Check if the current model has the best validation accuracy
    if validation_accuracy > best_accuracy:
        best_accuracy = validation_accuracy
        best_model_weights = model.state_dict()
# Save the best model
torch.save(best_model_weights, 'best_model.pth')
