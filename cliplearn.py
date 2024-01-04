#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

# Toy image and text embeddings (replace with actual model outputs for real applications)
image_embeddings = torch.randn(4, 512)  # 4 images, 512-dimensional embeddings
text_embeddings = torch.randn(4, 512)  # 4 text captions, 512-dimensional embeddings

class CLIPContrastiveLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(CLIPContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, image_embeddings, text_embeddings):
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        # Calculate cosine similarity matrix
        similarity_matrix = torch.matmul(image_embeddings, text_embeddings.t())

        # Diagonal mask to exclude self-similarity
        mask = torch.eye(image_embeddings.size(0), dtype=torch.bool)

        # Positive pairs: similarity between the correct image and text pairs
        positive_pairs = similarity_matrix[mask].view(image_embeddings.size(0), -1)

        # Negative pairs: similarity between incorrect image and text pairs
        negative_pairs = similarity_matrix[~mask].view(image_embeddings.size(0), -1)

        # Calculate contrastive loss with margin
        loss = F.margin_ranking_loss(negative_pairs, positive_pairs, target=torch.ones_like(positive_pairs), margin=self.margin)

        return loss

# Create the model and compute the loss
clip_loss_fn = CLIPContrastiveLoss(margin=0.1)
loss = clip_loss_fn(image_embeddings, text_embeddings)

print("Contrastive Loss:", loss.item())
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

#%%
class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output
# %%
# Define the input and output feature sizes
input_feat_size = 256
output_feat_size = 128

# Create an instance of the FeatureResizer class
resizer = FeatureResizer(input_feat_size, output_feat_size, dropout=0.2)

# Generate some random encoder features
encoder_features = torch.randn(10, input_feat_size)

# Pass the encoder features through the FeatureResizer
output_features = resizer(encoder_features)

# Print the output features
print(output_features)
# %%
def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X
# %%
# Define a tensor
X = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]])

# Apply L1 normalization using the l1norm function
normalized_X = l1norm(X, dim=0)

# Print the normalized tensor
print(normalized_X)
# %%
def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

# %%
def func_attention(query, context, smooth=1, raw_feature_norm="softmax", eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax()(attn * smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT