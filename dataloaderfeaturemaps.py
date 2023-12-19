#%%
import pandas as pd
from torch.utils.data import Dataset
from auxiliaries import *
#%%
default_transform_gray = tf.Compose([
    tf.ToPILImage(),
    ##aug_transforms(),
    tf.Resize((256, 256)),
    tf.CenterCrop(256),
    pil_contrast_strech(),
    # ,
    tf.ToTensor(),
    gray2rgb
])
#%%
class TempDataset(Dataset):
    def __init__(self, metafile_path, annotations_path, pathologies, transform=default_transform_gray,
                 data_format='tiff'):
        """
        Initializes a TempDataset object.

        Args:
            metafile_path (str): The path to the metadata file.
            annotations_path (str): The path to the annotations file.
            pathologies (list): A list of pathologies to predict.
            transform (function, optional): The transformation function to apply to the data. Defaults to default_transform_gray.
            data_format (str, optional): The format of the data. Defaults to 'tiff'.
        """
        self.metadata = pd.read_csv(metafile_path)
        self.annotations = pd.read_csv(annotations_path)
        self.pathologies = pathologies
        self.samples = get_samples(self.metadata, self.annotations, pathologies)

        self.t = default_transform_gray
        logger.info(f'{data_format.upper()} dataset loaded')
        self.data_reader = dict(
            tiff=load_tiff
        )[data_format]
        logger.info(f'Predicting {pathologies}')
        self.label_reader = get_labels
        self.labels = [(self.label_reader(self.samples[i], self.annotations, self.pathologies))[0][0]
                       for i in range(len(self.samples))]
        self.labels = torch.FloatTensor(self.labels)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the transformed images and labels.
        """
        sample = self.samples[idx]
        # torch tensor (image) or list of tensors (volume)
        imgs = self.data_reader(sample)
        labels = self.label_reader(sample, self.annotations, self.pathologies)  # unwrap two-dimensional array
        labels = torch.FloatTensor(labels)
        t_imgs = torch.cat([self.t(im) for im in imgs], dim=1)
        return t_imgs, labels.squeeze() #TODO ADD EHR info

#%%
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, root_dir, data_format='tiff', transform=default_transform_gray):


        self.root_dir = root_dir
        self.folder_names = os.listdir(root_dir)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.data_reader = dict(
            tiff=load_tiff
                        )[data_format]
        self.t = transform

    def __len__(self):
        return len(self.folder_names)

    def __getitem__(self, idx):
        folder_name = self.folder_names[idx]
        folder_path = os.path.join(self.root_dir, folder_name)
        image_paths = os.listdir(folder_path)

        sample = self.samples[idx]
        imgs = self.data_reader(sample)
        # Load and tile the images
        tiled_images = []
        for image_name in image_paths:
            image_path = os.path.join(folder_path, image_name)
            image = Image.open(image_path)
            # Tile the image here (implement your own tiling logic)
            tiled_images.append(image)

        # Concatenate the tiled images
        concatenated_image = torch.cat(tiled_images, dim=1)

        # Apply transformations
        transformed_image = self.transform(concatenated_image)

        return transformed_image
#%%
# Example usage
root_dir = '/home/avram/scratch/Houston/tiff_without_laterality/'  # Replace with the actual root directory of the image folders
dataset = ImageDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Iterate over the dataloader to get one image per folder
for images in dataloader:
    # Process the images as needed
    print(images.shape)  # Shape will be (batch_size, channels, height, width)
# %%
from medmnist import NoduleMNIST3D
dataset = NoduleMNIST3D(root='/scratch/pterway/slivit/datasets', split='train', download=True)
# %%
