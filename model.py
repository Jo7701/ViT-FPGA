import torch
import torch.nn as nn
import torchvision
import math
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchvision import datasets, models, transforms
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy


class PatchEmbedding(nn.Module):
    """TAKEN FROM https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632"""
    def __init__(self, in_channels=3, patch_size=16, emb_size=512, img_size=224):
        assert img_size % patch_size == 0, print("Image size must be a multiple of patch size")
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.distillation_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn(1, int(img_size**2 / patch_size**2 + 2), emb_size))

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_token = repeat(self.cls_token, '() s d -> b s d', b=b)
        dist_token = repeat(self.distillation_token, '() s d -> b s d', b=b)

        x = torch.cat([cls_token, dist_token, x], dim=1)
        return x + self.pos_embed

class MultiHeadAttention(nn.Module):
    """TAKEN FROM https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec"""
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.heads = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = nn.functional.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)

        return output #returns (batch, heads, seq_len, d_k)


    def forward(self, q, k, v):
        b, _, _ = q.shape
        q = self.q_linear(q).view(b, -1, self.heads, self.d_k).transpose(1, 2) #want (batch, heads, seq_len, d_k)
        k = self.k_linear(k).view(b, -1, self.heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(b, -1, self.heads, self.d_k).transpose(1, 2)

        scores = self.attention(q, k, v).transpose(1, 2).reshape(b, -1, self.d_model)
        return self.out(scores)

class EncoderBlock(nn.Module):
    def __init__(self, heads=8, d_model=512, ff=2048, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(heads, d_model, dropout)
        self.ff_linear = nn.Linear(d_model, ff)
        self.out_linear = nn.Linear(ff, d_model)
        self.relu = nn.LeakyReLU(1/64)
        self.ff_dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, embeddings):
        embeddings_normed = self.layer_norm1(embeddings)
        scores = self.attn_dropout(self.mha(embeddings_normed, embeddings_normed, embeddings_normed))
        scores = scores + embeddings
        scores_normed = self.layer_norm2(scores)
        output = self.relu(self.ff_linear(scores_normed))
        output = self.ff_dropout(self.out_linear(output))
        return output + scores

class ViT(LightningModule):
    def __init__(self, depth=6, heads=8, d_model=512, ff=2048, dropout=0.1, 
                    in_channels=3, patch_size=16, img_size=224, num_classes=10):
        super().__init__()
        self.embedder = PatchEmbedding(in_channels, patch_size, d_model, img_size)
        self.encoder = nn.Sequential(*[EncoderBlock(heads, d_model, ff, dropout) for _ in range(depth)])
        self.classification = nn.Linear(d_model, num_classes)
        self.layer_norm = nn.LayerNorm(d_model)
        self.reduce = Reduce('b n e -> b e', reduction='mean')

    def forward(self, imgs):
        embeddings = self.embedder(imgs)
        cls_encoding = self.encoder(embeddings)[:, 0, :] #the first output sequence corresponds to the [CLS] token
        cls_encoding = self.layer_norm(cls_encoding)
        return self.classification(cls_encoding)

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = nn.functional.cross_entropy(self(x), y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        acc = accuracy(y_hat, y)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-5)

# class ViT(nn.Module):
#     def __init__(self, depth=6, heads=8, d_model=512, ff=2048, dropout=0.1, 
#                     in_channels=3, patch_size=16, img_size=224, num_classes=10):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=3,              
#                 out_channels=16,            
#                 kernel_size=5,              
#                 stride=1,                   
#                 padding=0,                  
#             ),                              
#             nn.ReLU(),                      
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(16, 32, 5, 1, 0),     
#             nn.ReLU(),                      
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 5, 1, 0),     
#             nn.ReLU(),   
#             nn.Flatten(),
#             nn.Linear(5184, num_classes),
#         )

#     def forward(self, imgs):
#         # embeddings = self.embedder(imgs)
#         # cls_encoding = self.encoder(embeddings)[:, 0, :] #the first output sequence corresponds to the [CLS] token
#         # cls_encoding = self.layer_norm(cls_encoding)
#         # return self.classification(cls_encoding)
#         return self.net(imgs)

#     def training_step(self, batch, batch_nb):
#         x, y = batch
#         loss = nn.functional.cross_entropy(self(x), y)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         acc = accuracy(y_hat, y)
#         self.log("val_acc", acc, prog_bar=True)

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=0.01)