#%%
import gc
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # GPU ID
#%%
import numpy as np
import pandas as pd
import torch
from fastai.vision.augment import aug_transforms
from torch.utils.data import Dataset
from fastai.vision import *
import sys
sys.path.append('/scratch/pterway/slivit/SLIViT')
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
        tf.ToPILImage(),
        tf.Resize((256, 256)),
        pil_contrast_strech(),
        #RandomResizedCrop((256,256)),
        #RandomHorizontalFlip(),
        ToTensor(),
        #normalize,
        gray2rgb
    ])
#%%
class AmishDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.dataset[idx][0]), self.dataset[idx][1].astype(np.float32)
#%%
# from torch.nn.functional import F
# def resize_frames(frames, size):
#   """
#   Resizes all frames of an image tensor to a given size.

#   Args:
#     frames: A torch tensor of shape (T, H, W, C) where T is the number of frames and H, W, C are the height, width and channel dimensions of each frame.
#     size: A tuple of two integers representing the desired output size (H, W).

#   Returns:
#     A torch tensor of shape (T, resized_H, resized_W, C) with all frames resized to the specified size.
#   """
#   resized_frames = []
#   for i in range(frames.shape[-1]):
#     this_frame = frames[:,:,i]
#     frame_tensor = torch.tensor(this_frame)
#     resized_frame = torch.nn.functional.interpolate(frame_tensor.unsqueeze(0), size, mode="bilinear").squeeze(0)
#     # resized_frames = torch.nn.functional.interpolate(frames, size=size, mode='bilinear', align_corners=False)

#     resized_frames.append(resized_frame)
#   return torch.stack(resized_frames)
#%%
# from torchvision import transforms
# def resize_images(images, new_height, new_width):
#     """
#     Resize a batch of images.

#     Args:
#     - images (torch.Tensor): Input images of shape (#samples, height, width, channels).
#     - new_height (int): The new height for resizing.
#     - new_width (int): The new width for resizing.

#     Returns:
#     - torch.Tensor: Resized images of shape (#samples, new_height, new_width, channels).
#     """

#     # Convert the images to PIL Image format
#     pil_images = [transforms.ToPILImage()(image) for image in images]

#     # Define the transformation
#     resize_transform = transforms.Resize((new_height, new_width))

#     # Apply the transformation to each image
#     resized_images = [resize_transform(image) for image in pil_images]

#     # Convert the resized images back to a PyTorch tensor
#     resized_images_tensor = torch.stack([transforms.ToTensor()(image) for image in resized_images])

#     return resized_images_tensor    
#%%    
# Define the custom dataset
from medmnist import NoduleMNIST3D

class NoduleMNISTDataset(Dataset):
    def __init__(self, dataset= NoduleMNIST3D(root='/scratch/pterway/slivit/datasets',
                                               split='train', download=True),
                                                transform=transform_new  
                                            ):
        super().__init__()
        self.dataset = dataset 
        self.transform = transform
        self.resize_transform = transforms_resize.Resize((224, 224))


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image= image[0]
  
        imgs = [image[i] for i in range(image.shape[-1]) if i<2]
        t_imgs = torch.cat([torch.FloatTensor(transform_new(torch.tensor(im))) for im in imgs], dim=1)
        return t_imgs, label
        # return transforms.ToTensor()(image), label
        # return torch.FloatTensor(self.dataset[idx][0]), self.dataset[idx][1].astype(np.float32)

#%%
from medmnist import NoduleMNIST3D
# dataset = NoduleMNIST3D(root='/scratch/pterway/slivit/datasets', split='train', download=True)    
# %%
from torch.utils.data import DataLoader
# check if the dataloader works by sampling a batch
dataset = NoduleMNISTDataset()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# Iterate over the dataloader to get one image per folder
# for images, labels in dataloader:
#     # Process the images as needed
#     print(images.shape)  # Shape will be (batch_size, channels, height, width)
#     print(labels)
#%%

# %%
def load_backbone(path):
    model_tmp = AutoModelForImageClassification.from_pretrained("facebook/convnext-tiny-224", return_dict=False,num_labels=4, ignore_mismatched_sizes=True)
    model = ConvNext(model_tmp)
    model.load_state_dict(torch.load(path, map_location=torch.device("cuda")))
    model = torch.nn.Sequential(*[list(list(model_tmp.children())[0].children())[0], list(list(model_tmp.children())[0].children())[1]])
    return model

# %%
backbone_path = '/scratch/pterway/slivit/backbone/SLIViT_Backbones/Kermany_combined_backbone.pth' 

# %%
backbone = load_backbone(backbone_path)
backbone.to(device)

#%%
# Activation function for the hook
from collections import defaultdict
activation_backbone = defaultdict(list)
# activation_backbone = {}
def get_activation(name):
    def hook(model, input, outputs):
        activation_backbone[name].append(outputs.detach())
    return hook
# %%
# Register the hooks
# backbone[1].stages[0].layers[0].drop_path.register_forward_hook(get_activation('stage0_layer0_drop_path'))
# backbone[1].stages[0].layers[1].drop_path.register_forward_hook(get_activation('stage0_layer1_drop_path'))
# backbone[1].stages[0].layers[2].drop_path.register_forward_hook(get_activation('stage0_layer2_drop_path'))

backbone[1].stages[1].layers[2].drop_path.register_forward_hook(get_activation('drop_path'))
backbone[1].stages[0].layers[0].drop_path.register_forward_hook(get_activation('stage0_layer0_drop_path'))
backbone[1].stages[0].layers[1].drop_path.register_forward_hook(get_activation('stage0_layer1_drop_path'))
backbone[1].stages[0].layers[2].drop_path.register_forward_hook(get_activation('stage0_layer2_drop_path'))
with torch.no_grad():
    for i, data in enumerate(dataloader):
        if i<3:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = backbone(images)


# In[]

#%%
images, labels = next(iter(dataloader))
#%%
with torch.no_grad():
    outputs = backbone(images)
    # activation_model1 = get_activation(backbone, images, lname)[layer1]


#%%
#%%

# %%
def get_activation(model, images, layer_name):
    activation = {}

    def hook(model, input, output):
        activation[layer_name] = output.detach()

    layer = model._modules[layer_name]
    layer.register_forward_hook(hook)

    model(images)

    return activation


# %%
# load the first batch of images from the data loader
images, labels = next(iter(dataloader))

#%%
layer_name_for_activation = ['']
# %%
# push images to GPU
images = images.cuda()
# push model to GPU
backbone = backbone.cuda()
#%%
# Get layer names
layer_name = None
allnames = []
allmodules = []
for name, module in backbone.named_modules():
    allnames.append(name)
    allmodules.append(module)
        
# %%
lname = '1.stages.0.layers.0.pwconv1'
with torch.no_grad():
    outputs = backbone(images)
    activation_model1 = get_activation(backbone, images, lname)[layer1]
# %%
[k for k in backbone.named_parameters()]
backbone._modules.keys()

# %%
#%%
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# %%
# backbone[1].layernorm.register_forward_hook(get_activation('layernorm'))
backbone[1].stages[1].layers[2].drop_path.register_forward_hook(get_activation('drop_path'))

output = backbone(images)
act = activation['drop_path']
# %%
