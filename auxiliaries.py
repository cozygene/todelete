
# %%
import logging
import os
import PIL
import numpy as np
from PIL import Image
from skimage import exposure
from torchvision import transforms as tf
import torch
from torch import nn
from transformers import AutoModelForImageClassification
import torchvision.models as tmodels

logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

gray2rgb = tf.Lambda(lambda x: x.expand(3, -1, -1))
totensor = tf.Compose([
    tf.ToTensor(),
])

def get_labels(sample, labels, pathologies):
    filename = sample.split('/')[-1]

    pat1, pat2, _, _, exam_date, _, laterality = filename.split('_')

    patient_id = pat1 + '_' + pat2
    if exam_date == 'NA':
        exam_date = np.nan
    label = labels[(labels['PAT-ID'] == patient_id)&
                   (labels.DATE == exam_date)&
                   (labels.LT == laterality)
    ][pathologies].astype(int).values
    if label.shape[0] == 0:
        label=np.array([[np.nan]])
    if label.shape[0] > 1:
        label = label[:1]
    return label


def get_samples(metadata, labels, pathologies):
    samples = []
    label_to_count = {p: {} for p in pathologies}
    i=0
    for sample in metadata.vol_name.values: # vol_name2
        sample_labels = get_labels(sample, labels, pathologies)
        i += 1
        if sample_labels.size == 0:
            logger.debug(f'{sample} has no label')
            continue
        if np.isnan(sample_labels):
            logger.debug(f'{sample} labels contain NA: {sample_labels}')
            continue
        else:
            samples.append(sample)
            logger.debug(sample)
    logger.info(f'Label counts is: {label_to_count}')
    print(f'Label counts is: {label_to_count}')
    return samples
class pil_contrast_strech(object):

    def __init__(self, low=2, high=98):
        self.low, self.high = low, high

    def __call__(self, img):
        # Contrast stretching
        img = np.array(img)
        plow, phigh = np.percentile(img, (self.low, self.high))
        return PIL.Image.fromarray(exposure.rescale_intensity(img, in_range=(plow, phigh)))

def load_tiff(vol_name, tiff_path='/scratch/avram/Hadassah/OCTs'): ##/scratch/avram/Amish/tiff

    pat_folder = vol_name.split('_')[0] +'_'+vol_name.split('_')[1]
    jj=0
    ff_name=''
    for f_p in vol_name.split('_'):
        if jj>1:
            if jj>2:
                ff_name=ff_name+'_'+f_p
            else:
                ff_name = ff_name + f_p

        jj+=1
    try:
        img_paths = os.listdir(f'{tiff_path}/{pat_folder}/{ff_name}')
    except:
        img_paths = os.listdir(f'{tiff_path}/{pat_folder}/{ff_name[1:]}')

    vol = []
    slc_idxs=np.linspace(0, len(img_paths), 20).astype(int) ## *5 , 50

    for img_name in img_paths:
        img = Image.open(f'{tiff_path}/{pat_folder}/{ff_name}/{img_name}')
        vol.append(totensor(img))
    vol=[vol[i] for i in range(len(img_paths)) if i in slc_idxs]

    return vol

#%%




class ConvNext(nn.Module):
    def __init__(self, model):
        super(ConvNext, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)[0]

        return x


"""
Usage to load model:
    1. Initialize a backbone
    2. Create the model on top
if kermany pretrained weights exist (i.e. using the Azure GPU)

>> backbone = load_backbone("kermany").cuda()
else:
>> backbone = load_backbone("scratch").cuda() # initializes a backbone from scratch (optionally, use "imagenet")

>> model = SliverNet2(backbone, n_out=[number_of_targets]).cuda()

"""


def load_backbone(model_name):
    kermany_pretrained_weights = "/scratch/njchiang/slivernet_export/kermany_pretrained.pth"
    # kermany_pretrained_weights ='/home/berkin/projects/LeViT/Kermany_training_resnet50/Resnet50_backbone_3.pth'

    if "convnext" in str(model_name).lower():

        kermany_pretrained_weights = "/home/berkin/projects/Convnext_backbone/Convnext-tiny_backbone3/Kermany_training.pth"
        # kermany_pretrained_weights="/home/berkin/projects/Xray/Convnext-xray_14/Xray_training2_14.pth"
        ####model2 = AutoModelForImageClassification.from_pretrained("facebook/convnext-tiny-224", return_dict=False,
        ####                                                        num_labels=4, ignore_mismatched_sizes=True)
        model2 = AutoModelForImageClassification.from_pretrained("facebook/convnext-tiny-224", return_dict=False,
                                                                 num_labels=4, ignore_mismatched_sizes=True)
        #####from transformers import ConvNextConfig, ConvNextModel
        # Initializing a ConvNext convnext-tiny-224 style configuration
        ####configuration = ConvNextConfig()
        # Initializing a model (with random weights) from the convnext-tiny-224 style configuration
        #####model2 = ConvNextModel(configuration)
        model = ConvNext(model2)
        ##Kermany Pretrained weights
        model_weights = kermany_pretrained_weights
        model.load_state_dict(torch.load(model_weights, map_location=torch.device("cuda")))
        model_tmp = list(model.children())[0]
        model = torch.nn.Sequential(
            *[list(list(model_tmp.children())[0].children())[0], list(list(model_tmp.children())[0].children())[1]])
        return model
    elif "unet" in str(model_name).lower():
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                               in_channels=3, out_channels=1, init_features=32, pretrained=True)
        return model
    elif "resnet50" in str(model_name).lower():
        logging.info("Loading model from scratch")
        model = tmodels.resnet50(num_classes=4, pretrained=False)
        model_weights = kermany_pretrained_weights
        model.load_state_dict(torch.load(model_weights, map_location=torch.device("cuda")))

    elif "frozen_kermany" in str(model_name).lower():
        logging.info("Loading model from Frozen-Kermany")
        model = tmodels.resnet18(num_classes=4, pretrained=False)
        model_weights = kermany_pretrained_weights
        model.load_state_dict(torch.load(model_weights, map_location=torch.device("cuda"))['model'])
        for parameter in model.parameters():
            parameter.requires_grad = False
    elif "kermany" in str(model_name).lower():
        logging.info("Loading model from Kermany")
        model = tmodels.resnet18(num_classes=4, pretrained=False)
        # model = tmodels.resnet50(num_classes=4, pretrained=False)
        model_weights = kermany_pretrained_weights
        model.load_state_dict(torch.load(model_weights, map_location=torch.device("cuda"))['model'])
        # model.load_state_dict(torch.load(model_weights, map_location=torch.device("cuda"))['model'])
    elif "sliver" in str(model_name).lower():
        logging.info("Loading model from Kermany for SLIVER-NET")
        # model = tmodels.resnet18(num_classes=4, pretrained=False)
        model_weights = kermany_pretrained_weights
        # model.load_state_dict(torch.load(model_weights, map_location=torch.device("cuda"))['model'])
    else:
        logging.info("Loading model from scratch")
        model = tmodels.resnet18(pretrained=False)

    # This will be 512 units... HARD CODED b/c resnet
    # after hacking off the FC and pooling layer, Resnet18 downsamples 5x
    # output size: B x C x H // 32 x W // 32
    return torch.nn.Sequential(*list(model.children())[:-2])




