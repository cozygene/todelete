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
import torch.nn.init as init
from auxiliaries import *
from transformers import ConvNextConfig, ConvNextModel
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
from skimage.metrics import structural_similarity
# from skim import ssim
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from CKA import linear_CKA, kernel_CKA
from CKAGoogle import feature_space_linear_cka, cka, gram_linear, gram_rbf, cca
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
                                               split='test', download=True),
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
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
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

#%%
# def load_backbone_scratch(num_labels=4, num_channels = 3):
#     print('scratch backbone Loading')
#     configuration = ConvNextConfig(num_labels=num_labels,num_channels=num_channels)
#     model2 = ConvNextModel(configuration)
#     model = ConvNext(model2)
#     model_tmp=list(model.children())[0]
#     model_tmp = list(model_tmp.children())
#     model= torch.nn.Sequential(
#         model_tmp[0], model_tmp[1],model_tmp[2])
#     return model

def load_backbone_scratch(path, num_labels=4):
    model_tmp = AutoModelForImageClassification.from_pretrained("facebook/convnext-tiny-224", return_dict=False, num_labels=num_labels, ignore_mismatched_sizes=True)
    model = ConvNext(model_tmp)

    # Load the state dict, but don't overwrite weights
    model_dict = model.state_dict()
    state_dict = torch.load(path, map_location=torch.device("cuda"))
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == state_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # # Initialize weights with Xavier initialization, handling 1D tensors appropriately
    # for param in model.parameters():
    #     if param.requires_grad:
    #         if len(param.shape) >= 2:  # Apply Xavier initialization for tensors with 2 or more dimensions
    #             init.xavier_uniform_(param)  # Use xavier_uniform_ for activations like tanh or sigmoid
    #         else:  # For 1D tensors (e.g., biases), use a different initialization strategy
    #             init.normal_(param, mean=0.0, std=0.01)  # Example: Normal initialization

    model = torch.nn.Sequential(*[list(list(model_tmp.children())[0].children())[0], list(list(model_tmp.children())[0].children())[1]])
    # Initialize weights with Xavier initialization, handling 1D tensors appropriately
    for param in model.parameters():
        if param.requires_grad:
            # if len(param.shape) >= 2:  # Apply Xavier initialization for tensors with 2 or more dimensions
            #     init.xavier_uniform_(param)  # Use xavier_uniform_ for activations like tanh or sigmoid
            # else:  # For 1D tensors (e.g., biases), use a different initialization strategy
            #     init.normal_(param, mean=0.0, std=0.01)  # Example: Normal initialization
             init.normal_(param, mean = 0.0, std = 0.01)
    return model

# def load_backbone_scratch(path, num_labels=4):
#     from transformers import ConvNextConfig, ConvNextModel
#     configuration = ConvNextConfig(num_labels=1, num_channels=3)
#     model2 = ConvNextModel(configuration)
#     model = ConvNext(model2)
#     model_tmp = list(model.children())[0]
#     model_tmp = list(model_tmp.children())
#     model = torch.nn.Sequential(
#         model_tmp[0], model_tmp[1], model_tmp[2])
#     return model
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

backbone_2 = load_backbone_scratch(backbone_path_2, num_labels=14)
# backbone_2 = ConvNextM(backbone_2)
#%%

# backbone_2 = load_backbone_scratch(backbone_path, num_labels=4)

# backbone_2 = load_backbone_random(backbone_path_2, num_labels=14)

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
all_features = []
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
                activation_model1_flatten_pooled = torch.max(activation_model1, dim=1)[0]
                activation_model1_flatten_pooled_np = activation_model1_flatten_pooled.cpu().numpy()


                #TODO use a for loop before linear cka
                # activation_model2_flatten = activation_model2.reshape(activation_model2.size(0), -1).cpu().numpy()
                activation_model2_flatten = activation_model2.cpu().numpy()
                activation_model2_flatten_np = np.mean(activation_model2_flatten, axis=(2,3))
                activation_model2_flatten_pooled = torch.max(activation_model2, dim=1)[0]
                activation_model2_flatten_pooled_np = activation_model2_flatten_pooled.cpu().numpy()

                all_features.append((activation_model1_flatten_pooled_np, activation_model2_flatten_pooled_np))

                 #TODO use a for loop before linear cka
                # scores = []
                dimension_array = activation_model1_flatten.shape[1]
                # dimension_array = 10
                cross_Score = np.zeros((dimension_array, dimension_array))
                batch_scores = []
                # for i in range(0,activation_model2_flatten.shape[1]):
                #     for j in range(0,activation_model1_flatten.shape[1]):
                #         map_backbone_1 = activation_model1_flatten[:,i,:,:]
                #         map_backbone_1_reshaped = map_backbone_1.reshape(map_backbone_1.shape[0], -1)

                #         maps_backbone_2 = activation_model2_flatten[:,j,:,:]
                #         # maps_backbone_2 = np.random.random(maps_backbone_2.shape)
                #         maps_backbone_2 = maps_backbone_2*np.random.random()
                #         maps_backbone_2_reshaped = maps_backbone_2.reshape(maps_backbone_2.shape[0], -1)
                    
                #         this_score = linear_CKA(map_backbone_1_reshaped,
                #                                                 maps_backbone_2_reshaped)
                #         score2 = cka(gram_linear(map_backbone_1_reshaped), gram_linear(maps_backbone_2_reshaped))
                #         score3 = feature_space_linear_cka(map_backbone_1_reshaped, maps_backbone_2_reshaped)
                #         score_4 = cka(gram_linear(map_backbone_1_reshaped), gram_linear(maps_backbone_2_reshaped), debiased=True)
                #         # score_5 = cka(gram_linear(map_backbone_1_reshaped), gram_linear(maps_backbone_2_reshaped), debiased=True)
                #         score_5 = feature_space_linear_cka(map_backbone_1_reshaped, maps_backbone_2_reshaped, debiased=True)
                #         score6 = cka(gram_rbf(map_backbone_1_reshaped, 0.4), gram_rbf(maps_backbone_2_reshaped, 0.4))
                #         score7 = cka(gram_rbf(map_backbone_1_reshaped, 0.4), gram_rbf(maps_backbone_2_reshaped, 0.4), debiased=True)
                #         score8 = cca(map_backbone_1_reshaped, maps_backbone_2_reshaped)
                #         cross_Score[i][j] = this_score

                # scores.append(cross_Score)
#%%
feature_backbone_1 = np.concatenate([i[0] for i in all_features], axis=0)
feature_backbone_2 = np.concatenate([i[1] for i in all_features], axis=0)      
map_backbone_1_reshaped = feature_backbone_1.reshape(feature_backbone_1.shape[0], -1)
maps_backbone_2_reshaped = feature_backbone_2.reshape(feature_backbone_2.shape[0], -1)
#%%
this_score = linear_CKA(map_backbone_1_reshaped,
                                        maps_backbone_2_reshaped)
score2 = cka(gram_linear(map_backbone_1_reshaped), gram_linear(maps_backbone_2_reshaped))
score3 = feature_space_linear_cka(map_backbone_1_reshaped, maps_backbone_2_reshaped)
score_4 = cka(gram_linear(map_backbone_1_reshaped), gram_linear(maps_backbone_2_reshaped), debiased=True)
# score_5 = cka(gram_linear(map_backbone_1_reshaped), gram_linear(maps_backbone_2_reshaped), debiased=True)
score_5 = feature_space_linear_cka(map_backbone_1_reshaped, maps_backbone_2_reshaped, debiased=True)
score6 = cka(gram_rbf(map_backbone_1_reshaped, 0.4), gram_rbf(maps_backbone_2_reshaped, 0.4))
score7 = cka(gram_rbf(map_backbone_1_reshaped, 0.4), gram_rbf(maps_backbone_2_reshaped, 0.4), debiased=True)
score8 = cca(map_backbone_1_reshaped, maps_backbone_2_reshaped)   
#%%
ss_scores = []
for ll in range(feature_backbone_1.shape[0]):
    image1 = feature_backbone_1[ll]
    image1_scaled = (image1 - np.min(image1)) / (np.max(image1) - np.min(image1))
    image2 = feature_backbone_2[ll]
    image2_scaled = (image2 - np.min(image2)) / (np.max(image2) - np.min(image2))
    ss_scores.append(structural_similarity(image1, image2, data_range=image1.max() - image1.min()))
ss_scores_array = np.array(ss_scores)
print(np.mean(ss_scores_array))
#%%
#%%
# Convert the list of 2D arrays to a 3D array
scores_3d = np.stack(scores)         
#%%    
loading = False
if loading==False:
    mean_matrix = np.mean(scores_3d, axis=0)
    std_matrix = np.std(scores_3d, axis=0)
    np.savez('/scratch/pterway/slivit/SLIViT/analysisStores/matrix_data_middle_layer_random_v3.npz', mean_matrix=mean_matrix, std_matrix=std_matrix)
else:
    data = np.load('/scratch/pterway/slivit/SLIViT/analysisStores/matrix_data_middle_layer_random.npz')
    mean_matrix = data['mean_matrix']
    std_matrix = data['std_matrix']
#%%
# fing the 10 highest indices across the diagonals of mean_matrix
# Get the diagonal values of the matrix
diagonal_values = np.diag(mean_matrix)

# Sort the diagonal values in descending order and get the indices
sorted_indices = np.argsort(diagonal_values)[::-1]

# Get the top 10 indices
top_10_indices = sorted_indices[:10]
#%%
# subset_mean = mean_matrix[top_10_indices,top_10_indices]
subset_mean = mean_matrix[top_10_indices][:, top_10_indices]
#%%
# make a heatmaps of the mean and std
import matplotlib.pyplot as plt
import seaborn as sns
# Create a heatmap using Seaborn
plt.figure(figsize=(10, 8))
vmin, vmax = 0, 1
# sns.heatmap(mean_matrix[:10,:10], annot=True, fmt=".2f", cmap="YlGnBu",  vmin=vmin, vmax=vmax)
sns.heatmap(subset_mean, annot=True, fmt=".2f", cmap="YlGnBu",  vmin=vmin, vmax=vmax)

# Set the x and y ticks using top_10_indices
plt.xticks(range(len(top_10_indices)), top_10_indices)
plt.yticks(range(len(top_10_indices)), top_10_indices)

plt.xlabel('Feature number')
plt.ylabel('Feature number')
plt.title('Heatmap of Mean Matrix')
plt.show()
#%%
# Create a heatmap using Seaborn
subset_std = std_matrix[top_10_indices][:, top_10_indices]

plt.figure(figsize=(10, 8))
vmin, vmax = 0, 1
# sns.heatmap(mean_matrix[:10,:10], annot=True, fmt=".2f", cmap="YlGnBu",  vmin=vmin, vmax=vmax)
sns.heatmap(subset_std, annot=True, fmt=".2f", cmap="YlGnBu",  vmin=vmin, vmax=vmax)

# Set the x and y ticks using top_10_indices
plt.xticks(range(len(top_10_indices)), top_10_indices)
plt.yticks(range(len(top_10_indices)), top_10_indices)

plt.xlabel('Feature number')
plt.ylabel('Feature number')
plt.title('Heatmap of Mean Matrix')
plt.show()


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
