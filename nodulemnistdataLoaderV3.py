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
  
        imgs = [image[i] for i in range(image.shape[-1]) if i<1]
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
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
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
backbone_path_2 = '/scratch/pterway/slivit/backbone/SLIViT_Backbones/MRI_combined_backbone.pth' 
# backbone_path_2 = '/scratch/pterway/slivit/backbone/SLIViT_Backbones/Kermany_combined_backbone.pth' 
# backbone/SLIViT_Backbones/MRI_combined_backbone.pth
# %%
backbone = load_backbone(backbone_path)
backbone.to(device)

backbone_2 = load_backbone(backbone_path_2)
backbone_2.to(device)
#%%
# Activation function for the hook
from collections import defaultdict
# activation_backbone = defaultdict(list)
# # activation_backbone = {}
# def get_activation(name):
#     def hook(model, input, outputs):
#         activation_backbone[name].append(outputs.detach())
#     return hook
# # %%
# # Register forward hooks for stage 0
# for i, layer in enumerate(backbone_2[1].stages[0].layers):
#     layer.drop_path.register_forward_hook(get_activation(f'stage0_layer{i}_drop_path'))

# # Register forward hooks for stage 1
# for i, layer in enumerate(backbone_2[1].stages[1].layers):
#     layer.drop_path.register_forward_hook(get_activation(f'stage1_layer{i}_drop_path'))

# images, labels = next(iter(dataloader))
# images = images.to(device)

# outputs = backbone_2(images)
# %%
def get_activation_model(model, images):
    activation = {}

    def hook(layer_name):
        def fn(model, input, output):
            activation[layer_name] = output.detach()
        return fn

    for i, layer in enumerate(model[1].stages[0].layers):
        layer.pwconv2.register_forward_hook(hook(f'stage0_layer{i}_drop_path'))

    # Register forward hooks for stage 1
    for i, layer in enumerate(model[1].stages[1].layers):
        layer.pwconv2.register_forward_hook(hook(f'stage1_layer{i}_drop_path'))

    model(images)

    return activation
#%%
images, labels = next(iter(dataloader))
images = images.to(device)
outputs = backbone(images)
activate = get_activation_model(backbone, images)
keys = list(activate.keys())
#%%
cka_score = defaultdict(list)
#%%
with torch.no_grad():
    for k, data in enumerate(dataloader):
        if k<3:
            images, labels = data[0].to(device), data[1].to(device)
            activation_model1_all = get_activation_model(backbone, images)
            outputs_1 = backbone(images)
            activation_model2_all = get_activation_model(backbone_2, images)
            outputs_2 = backbone_2(images)
            for i, layer1 in enumerate(keys):
                for j, layer2 in enumerate(keys):
                    print(layer1, layer2)
                    activation_model1 = activation_model1_all[layer1]
                    activation_model2 = activation_model2_all[layer2]


                    # activation_model1_flatten = activation_model1.reshape(activation_model1.size(0), -1)
                    activation_model1_flatten_np = activation_model1.cpu().numpy()
                    activation_model1_flatten_np = np.mean(activation_model1_flatten_np, axis=(1,2))


                    # activation_model2_flatten = activation_model2.reshape(activation_model2.size(0), -1)
                    activation_model2_flatten_np = activation_model2.cpu().numpy()
                    activation_model2_flatten_np = np.mean(activation_model2_flatten_np, axis=(1,2))
                    # this_score = linear_CKA(activation_model1_flatten_np,
                    #                                             activation_model2_flatten_np)
                    this_score = kernel_CKA(activation_model1_flatten_np,
                                                                activation_model2_flatten_np)
                    print(this_score)
                    # avg_acts1 = np.mean(activation_model1, axis=(1,2))
                    # avg_acts2 = np.mean(activation_model2, axis=(1,2))
                    cka_score[(layer1,layer2)].append(this_score)



# In[]
# make a heat map of the activations
import numpy as np
cka_score_mean = {}
cka_score_std = {}
for key in cka_score.keys():
    cka_list = cka_score[key]
    mean_value = np.mean(cka_list)
    std_value = np.std(cka_list)
    cka_score_mean[key] = mean_value
    cka_score_std[key] = std_value

# %%
import matplotlib.pyplot as plt
data = dict(cka_score_mean)
layer_names = set()

for key in data.keys():
    layer_names.update(key)
n_layers = len(layer_names)

matrix = np.zeros((n_layers, n_layers))
layer_names_list = list(layer_names)
for (layer1, layer2), score in data.items():
    matrix[layer_names_list.index(layer1)][layer_names_list.index(layer2)] = score
x_labels = keys.copy()
y_labels = x_labels.copy()
plt.figure(figsize=(8, 6))
ax = plt.matshow(matrix, cmap='coolwarm')

# Set ticks and labels
plt.xticks(range(len(x_labels)), x_labels, rotation=45)
plt.yticks(range(len(y_labels)), y_labels)

# Add colorbar
plt.colorbar()

# Title and labels
plt.title("CKA Score Heatmap for the Two ML Models")
plt.xlabel("Model 1 Layer")
plt.ylabel("Model 2 Layer")

# Show the plot
plt.tight_layout()
plt.show()                    
#%%
print('done')
# images, labels = next(iter(dataloader))
#%%
# with torch.no_grad():
#     outputs = backbone(images)
#     # activation_model1 = get_activation(backbone, images, lname)[layer1]


# #%%
# #%%

# # %%
# def get_activation(model, images, layer_name):
#     activation = {}

#     def hook(model, input, output):
#         activation[layer_name] = output.detach()

#     layer = model._modules[layer_name]
#     layer.register_forward_hook(hook)

#     model(images)

#     return activation


# # %%
# # load the first batch of images from the data loader
# images, labels = next(iter(dataloader))

# #%%
# layer_name_for_activation = ['']
# # %%
# # push images to GPU
# images = images.cuda()
# # push model to GPU
# backbone = backbone.cuda()
# #%%
# # Get layer names
# layer_name = None
# allnames = []
# allmodules = []
# for name, module in backbone.named_modules():
#     allnames.append(name)
#     allmodules.append(module)
        
# # %%
# lname = '1.stages.0.layers.0.pwconv1'
# with torch.no_grad():
#     outputs = backbone(images)
#     activation_model1 = get_activation(backbone, images, lname)[layer1]
# # %%
# [k for k in backbone.named_parameters()]
# backbone._modules.keys()

# # %%
# #%%
# activation = {}
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook

# # %%
# # backbone[1].layernorm.register_forward_hook(get_activation('layernorm'))
# backbone[1].stages[1].layers[2].drop_path.register_forward_hook(get_activation('drop_path'))

# output = backbone(images)
# act = activation['drop_path']
# # %%
