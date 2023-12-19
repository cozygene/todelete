#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
import torch
import torchvision.models as tmodels
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from transformers import AutoModelForImageClassification
import torchvision.models as tmodels
# from google.colab import drive


# In[ ]:


class ConvNext(nn.Module):
    """
    A wrapper class that applies a forward pass through the given model.

    Args:
        model (nn.Module): The model to be wrapped.

    Returns:
        torch.Tensor: The output tensor after applying the forward pass.
    """

    def __init__(self, model):
        super(ConvNext, self).__init__()
        self.model=model

    def forward(self, x):
        x = self.model(x)[0]
        return x


# In[ ]:


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        """
        Attention module that performs multi-head self-attention.

        Args:
            dim (int): The input dimension of the attention module.
            heads (int, optional): The number of attention heads. Defaults to 8.
            dim_head (int, optional): The dimension of each attention head. Defaults to 64.
            dropout (float, optional): Dropout probability. Defaults to 0.
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying multi-head self-attention.
        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# In[ ]:


class FeedForward(nn.Module):
    """
    FeedForward module that applies a two-layer feedforward neural network to the input.

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The dimension of the hidden layer.
        dropout (float, optional): The dropout probability. Default is 0.

    Attributes:
        net (nn.Sequential): The sequential network consisting of linear layers, GELU activation, and dropout.

    Methods:
        forward(x): Forward pass of the network.

    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# In[ ]:


class PreNorm(nn.Module):
    """
    Pre-normalization module that applies layer normalization before passing the input to the given function.

    Args:
        dim (int): The input dimension.
        fn (callable): The function to be applied to the normalized input.

    Attributes:
        norm (nn.LayerNorm): The layer normalization module.
        fn (callable): The function to be applied to the normalized input.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """
        Forward pass of the PreNorm module.

        Args:
            x (torch.Tensor): The input tensor.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            torch.Tensor: The output tensor after applying the function to the normalized input.
        """
        return self.fn(self.norm(x), **kwargs)


# In[ ]:


class Transformer(nn.Module):
    """
    Transformer module that applies self-attention and feed-forward layers to the input.

    Args:
        dim (int): The input dimension.
        depth (int): The number of layers in the Transformer.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        mlp_dim (int): The dimension of the feed-forward layer.
        dropout (float, optional): The dropout rate. Default is 0.

    Returns:
        torch.Tensor: The output tensor after applying the Transformer layers.
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# In[ ]:


class SLIViT(nn.Module):
    """
    SLIViT (Sliced Linear Vision Transformer) module.

    Args:
        backbone (nn.Module): The backbone feature extractor.
        image_size (tuple[int]): The size of the input image (height, width).
        patch_size (tuple[int]): The size of each patch (height, width).
        num_classes (int): The number of output classes.
        dim (int): The dimension of the patch embeddings and transformer layers.
        depth (int): The number of transformer layers.
        heads (int): The number of attention heads in the transformer.
        mlp_dim (int): The dimension of the feed-forward MLP layers in the transformer.
        pool (str): The type of pooling to use, either 'cls' (cls token) or 'mean' (mean pooling).
        channels (int): The number of input channels.
        dim_head (int): The dimension of each attention head.
        dropout (float): The dropout rate.
        emb_dropout (float): The dropout rate for the patch embeddings.

    Attributes:
        backbone (nn.Module): The backbone feature extractor.
        channels (int): The number of input channels.
        to_patch_embedding (nn.Sequential): Sequential module for converting image patches to embeddings.
        pos_embedding (nn.Parameter): Positional embeddings for each patch.
        cls_token (nn.Parameter): Learnable cls token.
        dropout (nn.Dropout): Dropout layer.
        transformer (Transformer): Transformer module.
        pool (str): The type of pooling used.
        to_latent (nn.Identity): Identity module for converting transformer output to latent space.
        mlp_head (nn.Sequential): Sequential module for the final classification MLP head.
        act (nn.Sigmoid): Sigmoid activation function.

    """

    def __init__(self, *, backbone, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls',
                 channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.backbone = backbone
        self.channels=channels
        pair = lambda t: t if isinstance(t, tuple) else (t, t)
        image_height, image_width = pair(image_size)
        _, patch_width = pair(patch_size)
        patch_height=12*patch_width
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width) *channels
        patch_dim = patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c h w) (p1 p2)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )
        #Initialize Positional Embeddings
        tmpp = torch.zeros((1, dim))
        tmp3 = torch.arange(num_patches) + 1
        for i in range(num_patches): tmpp = torch.concat([tmpp, torch.ones((1, dim)) * tmp3[i]], axis=0)
        self.pos_embedding = nn.Parameter(tmpp.reshape((1, tmpp.shape[0], tmpp.shape[1])))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.act = nn.Sigmoid()

    def forward(self,x):
        """
        Forward pass of the SLIViT module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_classes).

        """
        #Convnext Backbone (Feature Extractor)
        x = self.backbone(x)
        x = x.last_hidden_state
        x = x.reshape((x.shape[0], x.shape[1], self.channels, 64))
        x = x.permute(0, 2, 1, 3)
        #Tokenizer
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        #Add Corresponding Slice # as Positional Embedding
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        #ViT
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        x = self.mlp_head(x)
        x = self.act(x)
        x = torch.squeeze(x)
        return x



# In[ ]:


def load_backbone(path):
    model_tmp = AutoModelForImageClassification.from_pretrained("facebook/convnext-tiny-224", return_dict=False,num_labels=4, ignore_mismatched_sizes=True)
    model = ConvNext(model_tmp)
    model.load_state_dict(torch.load(path, map_location=torch.device("cuda")))
    model = torch.nn.Sequential(*[list(list(model_tmp.children())[0].children())[0], list(list(model_tmp.children())[0].children())[1]])
    return model


# In[ ]:


slivit=SLIViT(backbone=load_backbone('/scratch/pterway/slivit/backbone/SLIViT_Backbones/MRI_combined_backbone.pth'),
              image_size=(768, 64), patch_size=64, num_classes=1,
              dim=256, depth=5, heads=19, mlp_dim=512, channels=19,
              dropout=0.2, emb_dropout=0.1)


# In[ ]:


batch_size=4
n_slices=33
n_channels=3
image_size=(256,256)
rand_long_images=torch.rand(batch_size,n_channels,n_slices*image_size[0], image_size[1])


# In[ ]:


scores=slivit(rand_long_images)


# In[ ]:


print(scores.detach().numpy())

