import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.utils.prune as prune
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelPruning
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from transformers import DeiTFeatureExtractor, DeiTModel, DeiTForImageClassification
from deit_train import DeiTClassifier
import math
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


batch_size = 8
lr = 5e-5
num_classes = 10

class QuantPatchEmbedding(nn.Module):
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

        self.wscale = 1

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x) / self.wscale #QUANT HERE
        cls_token = repeat(self.cls_token, '() s d -> b s d', b=b)
        dist_token = repeat(self.distillation_token, '() s d -> b s d', b=b)

        x = torch.cat([cls_token, dist_token, x], dim=1)
        return x + self.pos_embed

    def quant_layer(self, quant_bits=8):
        self.wscale = get_scale_factor(self.projection[1].weight, quant_bits, round_to_power2=True)
        self.bscale = get_scale_factor(self.projection[1].bias, quant_bits, round_to_power2=True)
        self.bias_dequant = self.wscale / self.bscale

        self.projection[1].weight = nn.Parameter((self.projection[1].weight * self.wscale).int().float())
        self.projection[1].bias = nn.Parameter((self.projection[1].bias * self.bscale * self.bias_dequant).int().float())
class QuantMultiHeadAttention(nn.Module):
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

        self.q_wscale = 1
        self.q_bscale = 1
        self.k_wscale = 1
        self.k_bscale = 1
        self.v_wscale = 1
        self.v_bscale = 1
        self.out_wscale = 1
        self.out_bscale = 1

        self.inp_scale = 1
        self.q_proj_scale = 1
        self.k_proj_scale = 1
        self.v_proj_scale = 1
        self.scores_scale = 1

        self.inp_range = []
        self.q_proj_range = []
        self.k_proj_range = []
        self.v_proj_range = []
        self.scores_range = []

    def attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) / self.q_proj_scale / self.k_proj_scale
        scores = nn.functional.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)

        return output #returns (batch, heads, seq_len, d_k)


    def forward(self, q, k, v):
        b, _, _ = q.shape
        self.inp_range += torch.amax(torch.abs(q), dim=(1, 2)).tolist()

        q = self.inp_scale * q
        k = self.inp_scale * k
        v = self.inp_scale * v

        q = self.q_proj_scale * self.q_linear(q).view(b, -1, self.heads, self.d_k).transpose(1, 2) / self.q_wscale / self.inp_scale #want (batch, heads, seq_len, d_k)
        k = self.k_proj_scale * self.k_linear(k).view(b, -1, self.heads, self.d_k).transpose(1, 2) / self.k_wscale / self.inp_scale
        v = self.v_proj_scale * self.v_linear(v).view(b, -1, self.heads, self.d_k).transpose(1, 2) / self.v_wscale / self.inp_scale

        self.q_proj_range += torch.amax(torch.abs(q), dim=(1,2,3)).tolist()
        self.k_proj_range += torch.amax(torch.abs(k), dim=(1,2,3)).tolist()
        self.v_proj_range += torch.amax(torch.abs(v), dim=(1,2,3)).tolist()

        scores = self.scores_scale * self.attention(q, k, v).transpose(1, 2).reshape(b, -1, self.d_model)
        self.scores_range += torch.amax(scores, dim=(1, 2))

        return self.out(scores) / self.out_wscale / self.scores_scale

    def quant_layer(self, quant_bits=8):
        # self.inp_scale = get_scale_factor(None, real_range=sum(self.inp_range)/len(self.inp_range), quant_bits=quant_bits, round_to_power2=True)
        # self.q_proj_scale = get_scale_factor(None, real_range=sum(self.q_proj_range)/len(self.q_proj_range), quant_bits=quant_bits, round_to_power2=True)
        # self.k_proj_scale = get_scale_factor(None, real_range=sum(self.k_proj_range)/len(self.k_proj_range), quant_bits=quant_bits, round_to_power2=True)
        # self.v_proj_scale = get_scale_factor(None, real_range=sum(self.v_proj_range)/len(self.v_proj_range), quant_bits=quant_bits, round_to_power2=True)
        # self.scores_scale = get_scale_factor(None, real_range=sum(self.scores_range)/len(self.scores_range), quant_bits=quant_bits, round_to_power2=True)

        self.q_wscale = get_scale_factor(self.q_linear.weight, quant_bits, round_to_power2=True)
        self.q_bscale = get_scale_factor(self.q_linear.bias, quant_bits, round_to_power2=True)
        self.qbias_dequant = self.inp_scale * self.q_wscale / self.q_bscale

        self.k_wscale = get_scale_factor(self.k_linear.weight, quant_bits, round_to_power2=True)
        self.k_bscale = get_scale_factor(self.k_linear.bias, quant_bits, round_to_power2=True)
        self.kbias_dequant = self.inp_scale * self.k_wscale / self.k_bscale

        self.v_wscale = get_scale_factor(self.v_linear.weight, quant_bits, round_to_power2=True)
        self.v_bscale = get_scale_factor(self.v_linear.bias, quant_bits, round_to_power2=True)
        self.vbias_dequant = self.inp_scale * self.v_wscale / self.v_bscale

        self.out_wscale = get_scale_factor(self.out.weight, quant_bits, round_to_power2=True)
        self.out_bscale = get_scale_factor(self.out.bias, quant_bits, round_to_power2=True)
        self.out_bias_dequant = self.scores_scale * self.out_wscale / self.out_bscale

        self.q_linear.weight = nn.Parameter((self.q_linear.weight * self.q_wscale).int().float())
        self.q_linear.bias = nn.Parameter((self.q_linear.bias * self.q_bscale * self.qbias_dequant).int().float())

        self.k_linear.weight = nn.Parameter((self.k_linear.weight * self.k_wscale).int().float())
        self.k_linear.bias = nn.Parameter((self.k_linear.bias * self.k_bscale * self.kbias_dequant).int().float())

        self.v_linear.weight = nn.Parameter((self.v_linear.weight * self.v_wscale).int().float())
        self.v_linear.bias = nn.Parameter((self.v_linear.bias * self.v_bscale * self.vbias_dequant).int().float())

        self.out.weight = nn.Parameter((self.out.weight * self.out_wscale).int().float())
        self.out.bias = nn.Parameter((self.out.bias * self.out_bscale * self.out_bias_dequant).int().float())

class QuantEncoderBlock(nn.Module):
    def __init__(self, heads=8, d_model=512, ff=2048, dropout=0.1):
        super().__init__()
        self.mha = QuantMultiHeadAttention(heads, d_model, dropout)
        self.ff_linear = nn.Linear(d_model, ff)
        self.out_linear = nn.Linear(ff, d_model)
        self.relu = nn.LeakyReLU(1/64)
        self.ff_dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.inter_wscale = 1
        self.out_wscale = 1

    def forward(self, embeddings):
        embeddings_normed = self.layer_norm1(embeddings)
        scores = self.attn_dropout(self.mha(embeddings_normed, embeddings_normed, embeddings_normed))
        scores = scores + embeddings
        scores_normed = self.layer_norm2(scores)
        output = self.relu(self.ff_linear(scores_normed)) / self.inter_wscale
        output = self.ff_dropout(self.out_linear(output)) / self.out_wscale
        return output + scores

    def quant_layer(self, quant_bits=8):
        self.mha.quant_layer()

        self.inter_wscale = get_scale_factor(self.ff_linear.weight, quant_bits, round_to_power2=True)
        self.inter_bscale = get_scale_factor(self.ff_linear.bias, quant_bits, round_to_power2=True)
        self.inter_bias_dequant = self.inter_wscale / self.inter_bscale

        self.out_wscale = get_scale_factor(self.out_linear.weight, quant_bits, round_to_power2=True)
        self.out_bscale = get_scale_factor(self.out_linear.bias, quant_bits, round_to_power2=True)
        self.out_bias_dequant = self.out_wscale / self.out_bscale

        self.ff_linear.weight = nn.Parameter((self.ff_linear.weight * self.inter_wscale).int().float())
        self.ff_linear.bias = nn.Parameter((self.ff_linear.bias * self.inter_bscale * self.inter_bias_dequant).int().float())

        self.out_linear.weight = nn.Parameter((self.out_linear.weight * self.out_wscale).int().float())
        self.out_linear.bias = nn.Parameter((self.out_linear.bias * self.out_bscale * self.out_bias_dequant).int().float())

class QuantViT(LightningModule):
    def __init__(self, depth=6, heads=8, d_model=512, ff=2048, dropout=0.1, 
                    in_channels=3, patch_size=16, img_size=224, num_classes=10):
        super().__init__()
        self.embedder = QuantPatchEmbedding(in_channels, patch_size, d_model, img_size)
        self.encoder = nn.Sequential(*[QuantEncoderBlock(heads, d_model, ff, dropout) for _ in range(depth)])
        self.classification = nn.Linear(d_model, num_classes)
        self.layer_norm = nn.LayerNorm(d_model)
        self.reduce = Reduce('b n e -> b e', reduction='mean')

        self.class_wscale = 1

    def forward(self, imgs):
        embeddings = self.embedder(imgs)
        cls_encoding = self.encoder(embeddings)[:, 0, :] #the first output sequence corresponds to the [CLS] token
        cls_encoding = self.layer_norm(cls_encoding)
        return self.classification(cls_encoding) / self.class_wscale

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

    def quant_layer(self, quant_bits):
        self.embedder.quant_layer(quant_bits)

        for i in range(len(self.encoder)):
            self.encoder[i].quant_layer(quant_bits)

        self.class_wscale = get_scale_factor(self.classification.weight, quant_bits, round_to_power2=True)
        self.class_bscale = get_scale_factor(self.classification.bias, quant_bits, round_to_power2=True)
        self.class_bias_dequant = self.class_wscale / self.class_bscale

        self.classification.weight = nn.Parameter((self.classification.weight * self.class_wscale).int().float())
        self.classification.bias = nn.Parameter((self.classification.bias * self.class_bscale * self.class_bias_dequant).int().float())


def eval(model, loader):
    model.eval()
    cuda = torch.device('cuda')
    model = model.cuda()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.cuda()
            labels = labels.cuda()
            output = model(images)
            pred_y = torch.argmax(output, dim=-1)
            correct += (pred_y == labels).sum().item()
            total += float(labels.size(0))
    print('Test Accuracy of the model on the test images: %.4f' % (correct/total))

"""Performs restricted range symmetric quantization"""
def get_scale_factor(tensor, real_range=None, quant_bits=8, round_to_power2=False):
    def round_pow2(x):
        #obtained from https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-or-equal-to-n-in-python 
        return 1 if x == 0 else 2**(x - 1).bit_length()

    if real_range is None:
        real_range = torch.max(torch.abs(tensor))
    scale = (2 ** (quant_bits-1) - 1) / real_range

    if round_to_power2:
        if isinstance(scale, int) or isinstance(scale, float):
            scale = round_pow2(int(scale))
        else:
            scale = round_pow2(scale.int().item())

    return scale

def quant_tensor(tensor, quant_bits=8):
    scale = get_scale_factor(tensor, quant_bits)
    return nn.Parameter(tensor * scale)

"""Returns scale factors for all weights and biases"""
def get_all_scale_factors(model, quant_bits=8):
    weight_scales = []
    bias_scales = []

    weight_scales.append(get_scale_factor(model.model.deit.embeddings.patch_embeddings.projection.weight, quant_bits))
    bias_scales.append(get_scale_factor(model.model.deit.embeddings.patch_embeddings.projection.weight, quant_bits))

    for i in range(12):
        weight_scales.append(get_scale_factor(model.model.deit.encoder.layer[i].attention.attention.query.weight, quant_bits))
        bias_scales.append(get_scale_factor(model.model.deit.encoder.layer[i].attention.attention.query.bias, quant_bits))

        weight_scales.append(get_scale_factor(model.model.deit.encoder.layer[i].attention.attention.key.weight, quant_bits))
        bias_scales.append(get_scale_factor(model.model.deit.encoder.layer[i].attention.attention.key.bias, quant_bits))

        weight_scales.append(get_scale_factor(model.model.deit.encoder.layer[i].attention.attention.value.weight, quant_bits))
        bias_scales.append(get_scale_factor(model.model.deit.encoder.layer[i].attention.attention.value.bias, quant_bits))

        weight_scales.append(get_scale_factor(model.model.deit.encoder.layer[i].attention.output.dense.weight, quant_bits))
        bias_scales.append(get_scale_factor(model.model.deit.encoder.layer[i].attention.output.dense.bias, quant_bits))

        weight_scales.append(get_scale_factor(model.model.deit.encoder.layer[i].intermediate.dense.weight, quant_bits))
        bias_scales.append(get_scale_factor(model.model.deit.encoder.layer[i].intermediate.dense.bias, quant_bits))

        weight_scales.append(get_scale_factor(model.model.deit.encoder.layer[i].output.dense.weight, quant_bits))
        bias_scales.append(get_scale_factor(model.model.deit.encoder.layer[i].output.dense.bias, quant_bits))

    weight_scales.append(get_scale_factor(model.model.classifier.weight))
    bias_scales.append(get_scale_factor(model.model.classifier.bias))

    return weight_scales, bias_scales



def quant_mha_layers(model, quant_bits=8):
    for i in range(12):
        model.model.deit.encoder.layer[i].attention.attention.query.weight = quant_tensor(model.model.deit.encoder.layer[i].attention.attention.query.weight, quant_bits)
        model.model.deit.encoder.layer[i].attention.attention.query.bias = quant_tensor(model.model.deit.encoder.layer[i].attention.attention.query.bias, quant_bits)

        model.model.deit.encoder.layer[i].attention.attention.key.weight = quant_tensor(model.model.deit.encoder.layer[i].attention.attention.key.weight, quant_bits)
        model.model.deit.encoder.layer[i].attention.attention.key.bias = quant_tensor(model.model.deit.encoder.layer[i].attention.attention.key.bias, quant_bits)

        model.model.deit.encoder.layer[i].attention.attention.value.weight = quant_tensor(model.model.deit.encoder.layer[i].attention.attention.value.weight, quant_bits)
        model.model.deit.encoder.layer[i].attention.attention.value.bias = quant_tensor(model.model.deit.encoder.layer[i].attention.attention.value.bias, quant_bits)

        model.model.deit.encoder.layer[i].attention.output.dense.weight = quant_tensor(model.model.deit.encoder.layer[i].attention.output.dense.weight, quant_bits)
        model.model.deit.encoder.layer[i].attention.output.dense.bias = quant_tensor(model.model.deit.encoder.layer[i].attention.output.dense.bias, quant_bits)

        model.model.deit.encoder.layer[i].intermediate.dense.weight = quant_tensor(model.model.deit.encoder.layer[i].intermediate.dense.weight, quant_bits)
        model.model.deit.encoder.layer[i].intermediate.dense.bias = quant_tensor(model.model.deit.encoder.layer[i].intermediate.dense.bias, quant_bits)

        model.model.deit.encoder.layer[i].output.dense.weight = quant_tensor(model.model.deit.encoder.layer[i].output.dense.weight, quant_bits)
        model.model.deit.encoder.layer[i].output.dense.bias = quant_tensor(model.model.deit.encoder.layer[i].output.dense.bias, quant_bits)
        
def quant_embeddings(model, quant_bits=8):
    model.model.deit.embeddings.patch_embeddings.projection.weight = quant_tensor(model.model.deit.embeddings.patch_embeddings.projection.weight, quant_bits)
    model.model.deit.embeddings.patch_embeddings.projection.bias = quant_tensor(model.model.deit.embeddings.patch_embeddings.projection.bias, quant_bits)

def quant_classification(model, quant_bits=8):
    model.model.classifier.weight = quant_tensor(model.model.classifier.weight, quant_bits)
    model.model.classifier.bias = quant_tensor(model.model.classifier.bias, quant_bits)

def quant_deit(model, quant_bits=8):
    # quant_embeddings(model, quant_bits)
    # quant_mha_layers(model, quant_bits)
    quant_classification(model, quant_bits)

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
    ]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainset_1 = torch.utils.data.Subset(trainset, list(range(0, 50000, 25)))

trainloader = torch.utils.data.DataLoader(trainset_1, batch_size=batch_size,
                                          shuffle=True, num_workers=12)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testset_1 = torch.utils.data.Subset(trainset, list(range(100)))

testloader = torch.utils.data.DataLoader(testset_1, batch_size=batch_size,
                                         shuffle=False, num_workers=12)

load_path = '/home/edith/Documents/projects/ViT/saved_model/super_extreme_pruned_relu_deit'

deit_model = DeiTClassifier('facebook/deit-base-distilled-patch16-224', 10)
config = deit_model.model.deit.config
model = QuantViT(config.num_hidden_layers, config.num_attention_heads, config.hidden_size, 
            config.intermediate_size, config.hidden_dropout_prob, 3, config.patch_size, config.image_size)
model.load_state_dict(torch.load(load_path))

# print(model.encoder[0].mha.q_linear.weight.shape)
# print((model.encoder[0].mha.q_linear.weight[:, 0]==0).sum().item())
# print((model.encoder[0].mha.q_linear.weight[:, 1]==0).sum().item())
# print((model.encoder[0].mha.q_linear.weight[:, 2]==0).sum().item())
# print((model.encoder[0].mha.q_linear.weight[:, 3]==0).sum().item())


eval(model, testloader) #warmup and get activation statistics
model.quant_layer(quant_bits=32)
eval(model, testloader)