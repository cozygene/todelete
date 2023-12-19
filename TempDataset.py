import pandas as pd
from torch.utils.data import Dataset
from auxiliaries import *

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

class TempDataset(Dataset):
    def __init__(self, metafile_path, annotations_path, pathologies, transform=default_transform_gray,
                 data_format='tiff'):
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
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # torch tensor (image) or list of tensors (volume)
        imgs = self.data_reader(sample)
        labels = self.label_reader(sample, self.annotations, self.pathologies)  # unwrap two-dimensional array
        labels = torch.FloatTensor(labels)
        t_imgs = torch.cat([self.t(im) for im in imgs], dim=1)
        return t_imgs, labels.squeeze() #TODO ADD EHR info

