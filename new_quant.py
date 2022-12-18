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


batch_size = 1
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
        x = fixed_point_rep(x, int_bits=1, frac_bits=7)
        x = fixed_point_rep(self.projection(x), int_bits=4, frac_bits=4)
        cls_token = repeat(self.cls_token, '() s d -> b s d', b=b)
        dist_token = repeat(self.distillation_token, '() s d -> b s d', b=b)

        x = torch.cat([cls_token, dist_token, x], dim=1)
        return fixed_point_rep(x + self.pos_embed, int_bits=4, frac_bits=4)

    def quant_layer(self, quant_bits=8):
        self.projection[1].weight = nn.Parameter(fixed_point_rep(self.projection[1].weight, int_bits=1, frac_bits=7))
        self.projection[1].bias = nn.Parameter(fixed_point_rep(self.projection[1].bias, int_bits=4, frac_bits=4))
        self.cls_token = nn.Parameter(fixed_point_rep(self.cls_token, int_bits=1, frac_bits=7))
        self.distillation_token = nn.Parameter(fixed_point_rep(self.distillation_token, int_bits=4, frac_bits=4))
        self.pos_embed = nn.Parameter(fixed_point_rep(self.pos_embed, int_bits=4, frac_bits=4))
        
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

    

    def attention(self, q, k, v):
        scores = fixed_point_rep(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k), int_bits=6, frac_bits=2)
        scores = fixed_point_rep(nn.functional.softmax(scores, dim=-1), int_bits=1, frac_bits=7)
        scores = self.dropout(scores)
        output = fixed_point_rep(torch.matmul(scores, v), int_bits=3, frac_bits=5)

        return output #returns (batch, heads, seq_len, d_k)


    def forward(self, q, k, v):
        b, _, _ = q.shape

        q = fixed_point_rep(self.q_linear(q).view(b, -1, self.heads, self.d_k).transpose(1, 2), int_bits=5, frac_bits=3) #want (batch, heads, seq_len, d_k)
        k = fixed_point_rep(self.k_linear(k).view(b, -1, self.heads, self.d_k).transpose(1, 2), int_bits=5, frac_bits=3)  
        v = fixed_point_rep(self.v_linear(v).view(b, -1, self.heads, self.d_k).transpose(1, 2), int_bits=5, frac_bits=3)

        scores = self.attention(q, k, v).transpose(1, 2).reshape(b, -1, self.d_model)
        output = fixed_point_rep(self.out(scores), int_bits=3, frac_bits=5)

        return output

    def quant_layer(self, quant_bits=8):
        # print(torch.min(self.q_linear.weight), torch.max(self.q_linear.weight))
        # print(torch.min(self.q_linear.bias), torch.max(self.q_linear.bias))
        # print(torch.min(self.k_linear.weight), torch.max(self.k_linear.weight))
        # print(torch.min(self.k_linear.bias), torch.max(self.k_linear.bias))
        # print(torch.min(self.v_linear.weight), torch.max(self.v_linear.weight))
        # print(torch.min(self.v_linear.bias), torch.max(self.v_linear.bias))
        # print(torch.min(self.out.weight), torch.max(self.out.weight))
        # print(torch.min(self.out.bias), torch.max(self.out.bias))

        self.q_linear.weight = nn.Parameter(fixed_point_rep(self.q_linear.weight, int_bits=1, frac_bits=7))
        self.q_linear.bias = nn.Parameter(fixed_point_rep(self.q_linear.bias, int_bits=4, frac_bits=4))

        self.k_linear.weight = nn.Parameter(fixed_point_rep(self.k_linear.weight, int_bits=1, frac_bits=7))
        self.k_linear.bias = nn.Parameter(fixed_point_rep(self.k_linear.bias, int_bits=4, frac_bits=4))

        self.v_linear.weight = nn.Parameter(fixed_point_rep(self.v_linear.weight, int_bits=1, frac_bits=7))
        self.v_linear.bias = nn.Parameter(fixed_point_rep(self.v_linear.bias, int_bits=4, frac_bits=4))

        self.out.weight = nn.Parameter(fixed_point_rep(self.out.weight, int_bits=1, frac_bits=7))
        self.out.bias = nn.Parameter(fixed_point_rep(self.out.bias, int_bits=4, frac_bits=4))

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
        embeddings_normed = fixed_point_rep(self.layer_norm1(embeddings), int_bits=5, frac_bits=3)
        scores = fixed_point_rep(self.attn_dropout(self.mha(embeddings_normed, embeddings_normed, embeddings_normed)), int_bits=3, frac_bits=5)
        scores = fixed_point_rep(scores + embeddings, int_bits=5, frac_bits=3)
        
        scores_normed = fixed_point_rep(self.layer_norm2(scores), int_bits=6, frac_bits=2)
        
        output = fixed_point_rep(self.relu(self.ff_linear(scores_normed)), int_bits=3, frac_bits=5)
        output = fixed_point_rep(self.ff_dropout(self.out_linear(output)), int_bits=3, frac_bits=5)
        return fixed_point_rep(output + scores, int_bits=5, frac_bits=3)

    def quant_layer(self, quant_bits=8):
        self.mha.quant_layer()

        # print(torch.min(self.ff_linear.weight), torch.max(self.ff_linear.weight))
        # print(torch.min(self.ff_linear.bias), torch.max(self.ff_linear.bias))
        # print(torch.min(self.out_linear.weight), torch.max(self.out_linear.weight))
        # print(torch.min(self.out_linear.bias), torch.max(self.out_linear.bias))

        self.ff_linear.weight = nn.Parameter(fixed_point_rep(self.ff_linear.weight, int_bits=1, frac_bits=7))
        self.ff_linear.bias = nn.Parameter(fixed_point_rep(self.ff_linear.bias, int_bits=4, frac_bits=4))

        self.out_linear.weight = nn.Parameter(fixed_point_rep(self.out_linear.weight, int_bits=1, frac_bits=7))
        self.out_linear.bias = nn.Parameter(fixed_point_rep(self.out_linear.bias, int_bits=4, frac_bits=4))

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
        cls_encoding = fixed_point_rep(self.layer_norm(cls_encoding), int_bits=5, frac_bits=3)
        output = fixed_point_rep(self.classification(cls_encoding), int_bits=6, frac_bits=2)
        return output

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

        self.classification.weight = nn.Parameter(fixed_point_rep(self.classification.weight, int_bits=1, frac_bits=7))
        self.classification.bias = nn.Parameter(fixed_point_rep(self.classification.bias, int_bits=4, frac_bits=4))


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
        return 1 if x == 0 else 2**((x - 1).bit_length()-1)

    if real_range is None:
        real_range = torch.max(torch.abs(tensor))
    scale = (2 ** (quant_bits-1) - 1) / real_range

    if round_to_power2:
        if isinstance(scale, int) or isinstance(scale, float):
            scale = round_pow2(int(scale))
        else:
            scale = round_pow2(scale.int().item())

    return scale

def fixed_point_rep(tensor, int_bits=4, frac_bits=4):
    """taken from https://stackoverflow.com/questions/37096090/how-to-use-python-to-convert-a-float-number-to-fixed-point-with-predefined-numbe"""
    unsigned_tensor = torch.abs(tensor)
    if int_bits > 0:
        whole = torch.clamp(unsigned_tensor.int(), -2**(int_bits-1), 2**(int_bits-1)-1)
    else:
        whole = torch.zeros_like(unsigned_tensor)
    frac = unsigned_tensor % 1
    sign = torch.sign(tensor)
    f = (1<<frac_bits)
    return sign*(whole + torch.round(frac*f) * (1.0/f))

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


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
        ]
    )

    tuneset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    tuneset_1 = torch.utils.data.Subset(tuneset, list(range(10)))

    tuneloader = torch.utils.data.DataLoader(tuneset_1, batch_size=batch_size,
                                            shuffle=False, num_workers=12)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testset_1 = torch.utils.data.Subset(testset, list(range(100)))

    testloader = torch.utils.data.DataLoader(testset_1, batch_size=batch_size,
                                            shuffle=False, num_workers=12)

    load_path = '/home/edith/Documents/projects/ViT/saved_model/super_extreme_pruned_relu_deit'

    deit_model = DeiTClassifier('facebook/deit-base-distilled-patch16-224', 10)
    config = deit_model.model.deit.config
    model = QuantViT(config.num_hidden_layers, config.num_attention_heads, config.hidden_size, 
                config.intermediate_size, config.hidden_dropout_prob, 3, config.patch_size, config.image_size)
    model.load_state_dict(torch.load(load_path))

    model.quant_layer(quant_bits=8)
    eval(model, testloader)