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
# from medmnist import OCTMNIST
from medmnist import OrganMNIST3D


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
        total_frames = image.shape[-1]
        middle_frame = int(total_frames/2)
        imgs = [image[i] for i in range(image.shape[-1]) if i==middle_frame]
        # imgs = [image[i] for i in range(image.shape[-1]) if i<1]
        # imgs = [image[i] for i in range(image.shape[-1])]
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
dataloader = DataLoader(dataset, batch_size=38, shuffle=True)
# Iterate over the dataloader to get one image per folder
# for images, labels in dataloader:
#     # Process the images as needed
#     print(images.shape)  # Shape will be (batch_size, channels, height, width)
#     print(labels)
#%%

# %%
def load_backbone(path, num_labels=4):
    model_tmp = AutoModelForImageClassification.from_pretrained("facebook/convnext-tiny-224", return_dict=False,num_labels=num_labels, ignore_mismatched_sizes=True)
    model = ConvNext(model_tmp)
    model.load_state_dict(torch.load(path, map_location=torch.device("cuda")))
    model = torch.nn.Sequential(*[list(list(model_tmp.children())[0].children())[0], list(list(model_tmp.children())[0].children())[1]])
    return model

# %%
backbone_path = '/scratch/pterway/slivit/backbone/SLIViT_Backbones/Kermany_combined_backbone.pth' 
# backbone_path_2 = '/scratch/pterway/slivit/backbone/SLIViT_Backbones/MRI_combined_backbone.pth' 
# backbone/SLIViT_Backbones/ssCombined_backbone.pt
# backbone_path_2 = '/scratch/pterway/slivit/backbone/SLIViT_Backbones/ssCombined_backbone.pt' 

# backbone/SLIViT_Backbones/Xray_combined_backbone.pth
# backbone_path_2 = '/scratch/pterway/slivit/backbone/SLIViT_Backbones/Xray_backbone.pth' 
backbone_path_2 = '/scratch/pterway/slivit/backbone/SLIViT_Backbones/Xray_combined_backbone.pth' 

#%%
class ConvNextM(nn.Module):
    def __init__(self, model):
        super(ConvNextM, self).__init__()
        self.model = model  # model here is original convnext

    def forward(self, x):
        x = self.model(x)
        x=x.last_hidden_state # Output size : batch x 768 x 8 x n_slices * 8  
        x = x.reshape((x.shape[0],x.shape[1] * int(x.shape[3] / 8), 8, 8))  # Output size : batch x 768 * n_slices x 8 x 8
        return x
#%%

# new_model = ConvNextM(backbone).cuda()
# backbone/SLIViT_Backbones/MRI_combined_backbone.pth
    # MRI 4, Xray 14 -> number of labels
# %%
backbone = load_backbone(backbone_path, num_labels=4)
# backbone = ConvNextM(backbone)
backbone.to(device)

backbone_2 = load_backbone(backbone_path_2, num_labels=14)
# backbone_2 = ConvNextM(backbone_2)
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
        layer.dwconv.register_forward_hook(hook(f'stage0_layer{i}_dwconv'))
        if i>1:
            break

    for i, layer in enumerate(model[1].stages[0].layers):
        if i<2:
            layer.drop_path.register_forward_hook(hook(f'stage0_layer{i}_drop_path'))

    # Register forward hooks for stage 1
    # for i, layer in enumerate(model[1].stages[1].layers):
    #     layer.drop_path.register_forward_hook(hook(f'stage1_layer{i}_drop_path'))

    model(images)

    return activation
#%%
images, labels = next(iter(dataloader))
images = images.to(device)
with torch.no_grad():
    outputs = backbone(images)
activate = get_activation_model(backbone, images)
keys = list(activate.keys())
#%%
cka_score = defaultdict(list)
layer_activations = defaultdict(list)
cka_score_combined_later = defaultdict(list)
#%%
from itertools import combinations
from itertools import combinations_with_replacement

layers_names = keys.copy()
layers_names.append('output')

combinations_2 = list(combinations_with_replacement(layers_names, 2))
#%%
scores = []
combinations_2 = [('output', 'output')]
#%%
with torch.no_grad():
    for k, data in tqdm(enumerate(dataloader)):
        if k<10:
            images, labels = data[0].to(device), data[1].to(device)
            activation_model1_all = get_activation_model(backbone, images)
            outputs_1 = backbone(images)
            outputs_1 = outputs_1.last_hidden_state
            activation_model1_all['output'] = outputs_1
            activation_model2_all = get_activation_model(backbone_2, images)
            outputs_2 = backbone_2(images)
            outputs_2 = outputs_2.last_hidden_state
            activation_model2_all['output'] = outputs_2

            for ii, (layer1,layer2) in enumerate(combinations_2):
                # print('='*50)
            # for j, layer2 in enumerate(keys):
                # print(layer1, layer2)
                activation_model1 = activation_model1_all[layer1]
                activation_model2 = activation_model2_all[layer2]


                # activation_model1_flatten = activation_model1.reshape(activation_model1.size(0), -1).cpu().numpy()
                activation_model1_flatten = activation_model1.cpu().numpy()
                activation_model1_flatten_np = np.mean(activation_model1_flatten, axis=(2,3))

                #TODO use a for loop before linear cka
                # activation_model2_flatten = activation_model2.reshape(activation_model2.size(0), -1).cpu().numpy()
                activation_model2_flatten = activation_model2.cpu().numpy()
                # activation_model2_flatten_np = np.mean(activation_model2_flatten, axis=(2,3))
                 #TODO use a for loop before linear cka
                # scores = []
                dimension_array = activation_model1_flatten.shape[1]
                cross_Score = np.zeros((dimension_array, dimension_array))
                batch_scores = []
                for i in range(0,activation_model2_flatten.shape[1]):
                    for j in range(0,activation_model1_flatten.shape[1]):
                        map_backbone_1 = activation_model1_flatten[:,i,:,:]
                        map_backbone_1_reshaped = map_backbone_1.reshape(map_backbone_1.shape[0], -1)

                        maps_backbone_2 = activation_model2_flatten[:,j,:,:]
                        maps_backbone_2_reshaped = maps_backbone_2.reshape(maps_backbone_2.shape[0], -1)
                    
                        this_score = linear_CKA(map_backbone_1_reshaped,
                                                                maps_backbone_2_reshaped)
                        cross_Score[i][j] = this_score

                scores.append(cross_Score)
                    # this_score = np.mean(np.array(batch_scores), axis=0)
                    # scores.append(batch_scores)
                
                
                # for i in range(0,activation_model2_flatten.shape[1]):
                #     map_backbone_1 = activation_model1_flatten[:,i,:,:]
                #     map_backbone_1_reshaped = map_backbone_1.reshape(map_backbone_1.shape[0], -1)

                #     maps_backbone_2 = activation_model2_flatten[:,i,:,:]
                #     maps_backbone_2_reshaped = maps_backbone_2.reshape(maps_backbone_2.shape[0], -1)
                
                #     this_score = linear_CKA(map_backbone_1_reshaped,
                #                                             maps_backbone_2_reshaped)
                    
                #     batch_scores.append(this_score)

                # # this_score = np.mean(np.array(batch_scores), axis=0)
                # scores.append(batch_scores)

                # this_score = kernel_CKA(activation_model1_flatten_np,
                #                                             activation_model2_flatten_np)
                # print(this_score)
                # avg_acts1 = np.mean(activation_model1, axis=(1,2))
                # avg_acts2 = np.mean(activation_model2, axis=(1,2))
                # layer_activations[layer1].append(activation_model1_flatten_np)
                # layer_activations[layer2].append(activation_model2_flatten_np)
                # layer_activations[(layer1,layer2)].append((activation_model1_flatten_np, activation_model2_flatten_np))

                # cka_score[(layer1,layer2)].append(this_score)
                # if layer1 != layer2:
                #     cka_score[(layer2,layer1)].append(this_score)

            # for i, layer1 in enumerate(keys):
            #     print('='*50)
            #     for j, layer2 in enumerate(keys):
            #         print(layer1, layer2)
            #         activation_model1 = activation_model1_all[layer1]
            #         activation_model2 = activation_model2_all[layer2]


            #         # activation_model1_flatten = activation_model1.reshape(activation_model1.size(0), -1)
            #         activation_model1_flatten_np = activation_model1.cpu().numpy()
            #         activation_model1_flatten_np = np.mean(activation_model1_flatten_np, axis=(1,2))


            #         # activation_model2_flatten = activation_model2.reshape(activation_model2.size(0), -1)
            #         activation_model2_flatten_np = activation_model2.cpu().numpy()
            #         activation_model2_flatten_np = np.mean(activation_model2_flatten_np, axis=(1,2))
            #         # this_score = linear_CKA(activation_model1_flatten_np,
            #         #                                             activation_model2_flatten_np)
            #         this_score = kernel_CKA(activation_model1_flatten_np,
            #                                                     activation_model2_flatten_np)
            #         print(this_score)
            #         # avg_acts1 = np.mean(activation_model1, axis=(1,2))
            #         # avg_acts2 = np.mean(activation_model2, axis=(1,2))
            #         cka_score[(layer1,layer2)].append(this_score)
#%%
# Convert the list of 2D arrays to a 3D array
scores_3d = np.stack(scores)         
#%%    
mean_matrix = np.mean(scores_3d, axis=0)
std_matrix = np.std(scores_3d, axis=0)
np.savez('/scratch/pterway/slivit/SLIViT/analysisStores/matrix_data_middle_layer.npz', mean_matrix=mean_matrix, std_matrix=std_matrix)
#%%
# make a heatmaps of the mean and std
import matplotlib.pyplot as plt
import seaborn as sns
# Create a heatmap using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(mean_matrix[:10,:10], annot=True, fmt=".2f", cmap="YlGnBu")
plt.xlabel('Column')
plt.ylabel('Row')
plt.title('Heatmap of Mean Matrix')
plt.show()


#%%
scores_array = np.array(scores)
# compute the mean and std
scores_mean = np.mean(scores_array, axis=0)
scores_std = np.std(scores_array, axis=0)  
# make a bar plot of the mean and std
#%%
import matplotlib.pyplot as plt

# Calculate the number of features per subplot
n_features = len(scores_mean)
features_per_subplot = n_features // 4

# Create a figure with four subplots
fig, axs = plt.subplots(2, 2, figsize=(10*2, 8))

# Iterate over the subplots and plot the corresponding features
for i, ax in enumerate(axs.flat):
    start_idx = i * features_per_subplot
    end_idx = start_idx + features_per_subplot

    ax.bar(np.arange(start_idx, end_idx), scores_mean[start_idx:end_idx], yerr=scores_std[start_idx:end_idx])
    ax.set_ylabel('CKA Score')
    ax.set_xlabel('Feature map')
    ax.set_title(f'CKA Score across feature maps (Plot {i+1})')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()
#%%
plt.figure(figsize=(10, 8))
plt.bar(np.arange(len(scores_std)), scores_std, yerr=scores_std)
# plt.xticks(np.arange(len(scores_std)), keys, rotation=90)
plt.ylabel('CKA Score std. dev.')
plt.xlabel('Feature map')
plt.title('CKA Score std. dev. across feature maps')
plt.show()

#%%
for ii, (layer1,layer2) in enumerate(combinations_2):    
    this_layer_activations =  layer_activations[(layer1,layer2)]    
    model_1_activation = np.concatenate([this_layer_activations[i][0] for i in range(len(this_layer_activations))])
    model_2_activation = np.concatenate([this_layer_activations[i][1] for i in range(len(this_layer_activations))]) 
    this_score = linear_CKA(model_1_activation, model_2_activation)
    if layer1 != layer2:
        cka_score_combined_later[(layer2,layer1)].append(this_score)
        cka_score_combined_later[(layer1,layer2)].append(this_score)   
    else:
        cka_score_combined_later[(layer1,layer2)].append(this_score)        

# In[]
# make a heat map of the activations
import numpy as np
cka_score_mean = {}
cka_score_std = {}
cka_score_combined_later_mean = {}
for key in cka_score.keys():
    cka_list = cka_score[key]
    mean_value = np.mean(cka_list)
    std_value = np.std(cka_list)
    cka_score_mean[key] = mean_value
    cka_score_std[key] = std_value
    cka_score_combined_later_mean[key] = cka_score_combined_later[key][0]


#%%
import matplotlib.pyplot as plt
import seaborn as sns
data = dict(cka_score_mean)
# Extract unique keys for X and Y axes
x_keys = sorted(set(key[0] for key in data.keys()))
y_keys = sorted(set(key[1] for key in data.keys()))

# Create a matrix to store the values
matrix = np.zeros((len(y_keys), len(x_keys)))

# Fill the matrix with the corresponding values from the dictionary
for i, y_key in enumerate(y_keys):
    for j, x_key in enumerate(x_keys):
        matrix[i, j] = data.get((x_key, y_key), 0.0)
# Specify the color bar range (vmin and vmax)
vmin, vmax = 0, 1
# Create a heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=x_keys, yticklabels=y_keys, vmin=vmin, vmax=vmax)
# sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=x_keys, yticklabels=y_keys)
plt.xlabel('Model 1')
plt.ylabel('Model 2')
plt.title('Mean')
plt.show()
#%%
data = dict(cka_score_std)
# Extract unique keys for X and Y axes
x_keys = sorted(set(key[0] for key in data.keys()))
y_keys = sorted(set(key[1] for key in data.keys()))

# Create a matrix to store the values
matrix = np.zeros((len(y_keys), len(x_keys)))

# Fill the matrix with the corresponding values from the dictionary
for i, y_key in enumerate(y_keys):
    for j, x_key in enumerate(x_keys):
        matrix[i, j] = data.get((x_key, y_key), 0.0)
vmin, vmax = 0, .4
# Create a heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=x_keys, yticklabels=y_keys, vmin=vmin, vmax=vmax)

# sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=x_keys, yticklabels=y_keys)
plt.xlabel('Model 1')
plt.ylabel('Model 2')
plt.title('Standard deviation')
plt.show()
#%%
data = dict(cka_score_combined_later_mean)
# Extract unique keys for X and Y axes
x_keys = sorted(set(key[0] for key in data.keys()))
y_keys = sorted(set(key[1] for key in data.keys()))

# Create a matrix to store the values
matrix = np.zeros((len(y_keys), len(x_keys)))

# Fill the matrix with the corresponding values from the dictionary
for i, y_key in enumerate(y_keys):
    for j, x_key in enumerate(x_keys):
        matrix[i, j] = data.get((x_key, y_key), 0.0)
# Specify the color bar range (vmin and vmax)
vmin, vmax = 0, 1
# Create a heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=x_keys, yticklabels=y_keys, vmin=vmin, vmax=vmax)
# sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=x_keys, yticklabels=y_keys)
plt.xlabel('Model 1')
plt.ylabel('Model 2')
plt.title('Mean')
plt.show()
# %%
# import matplotlib.pyplot as plt
# data = dict(cka_score_mean)
# layer_names = set()

# for key in data.keys():
#     layer_names.update(key)
# n_layers = len(layer_names)

# matrix = np.zeros((n_layers, n_layers))
# layer_names_list = list(layer_names)
# for (layer1, layer2), score in data.items():
#     matrix[layer_names_list.index(layer1)][layer_names_list.index(layer2)] = score
# x_labels = layer_names.copy()
# y_labels = x_labels.copy()
# plt.figure(figsize=(8, 6))
# ax = plt.matshow(matrix, cmap='coolwarm')

# # Set ticks and labels
# plt.xticks(range(len(x_labels)), x_labels, rotation=90)
# plt.yticks(range(len(y_labels)), y_labels)

# # Add colorbar
# plt.colorbar()
# # Set scale of the colorbar to range from 0 to 1
# plt.clim(0, 1)
# # Title and labels
# plt.title("CKA Score Heatmap for the Two ML Models")
# plt.xlabel("Model 1 Layer")
# plt.ylabel("Model 2 Layer")

# # Show the plot
# plt.tight_layout()
# plt.show() 
# #%%
# # %%
# data = dict(cka_score_std )
# layer_names = set()

# for key in data.keys():
#     layer_names.update(key)
# n_layers = len(layer_names)

# matrix = np.zeros((n_layers, n_layers))
# layer_names_list = list(layer_names)
# for (layer1, layer2), score in data.items():
#     matrix[layer_names_list.index(layer1)][layer_names_list.index(layer2)] = score

# plt.figure(figsize=(8, 6))
# ax = plt.matshow(matrix, cmap='coolwarm')

# # Set ticks and labels
# plt.xticks(range(len(x_labels)), x_labels, rotation=45)
# plt.yticks(range(len(y_labels)), y_labels)

# # Add colorbar
# plt.colorbar()
# plt.clim(0, .3)
# # Title and labels
# plt.title("CKA Score Heatmap std. dev. for the Two ML Models")
# plt.xlabel("Model 1 Layer")
# plt.ylabel("Model 2 Layer")

# # Show the plot
# plt.tight_layout()
# plt.show()
# # %%                   
# #%%
# print('done')
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
