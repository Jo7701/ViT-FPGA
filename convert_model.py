import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from model import ViT
from torchvision import datasets, models, transforms
import torch.nn.utils.prune as prune
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelPruning
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from transformers import DeiTFeatureExtractor, DeiTModel, DeiTForImageClassification
from deit_train import DeiTClassifier
from torchmetrics.functional import accuracy
import math

batch_size = 32
lr = 5e-5
num_classes = 10

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

def copy_model(relu_model, deit_model):
    #first copy embedding layer
    relu_model.embedder.projection[0].weight = deit_model.model.deit.embeddings.patch_embeddings.projection.weight
    relu_model.embedder.projection[0].bias = deit_model.model.deit.embeddings.patch_embeddings.projection.bias
    relu_model.embedder.cls_token = deit_model.model.deit.embeddings.cls_token
    relu_model.embedder.distillation_token = deit_model.model.deit.embeddings.distillation_token
    relu_model.embedder.pos_embed = deit_model.model.deit.embeddings.position_embeddings

    #copy Encoder layers
    for i in range(self.depth):
        relu_model.encoder[i].mha.q_linear.weight = deit_model.model.deit.encoder.layer[i].attention.attention.query.weight
        relu_model.encoder[i].mha.q_linear.bias = deit_model.model.deit.encoder.layer[i].attention.attention.query.bias
        relu_model.encoder[i].mha.k_linear.weight = deit_model.model.deit.encoder.layer[i].attention.attention.key.weight
        relu_model.encoder[i].mha.k_linear.bias = deit_model.model.deit.encoder.layer[i].attention.attention.key.bias
        relu_model.encoder[i].mha.v_linear.weight = deit_model.model.deit.encoder.layer[i].attention.attention.value.weight
        relu_model.encoder[i].mha.v_linear.bias = deit_model.model.deit.encoder.layer[i].attention.attention.value.bias

        relu_model.encoder[i].mha.out.weight = deit_model.model.deit.encoder.layer[i].attention.output.dense.weight
        relu_model.encoder[i].mha.out.bias = deit_model.model.deit.encoder.layer[i].attention.output.dense.bias

        relu_model.encoder[i].ff_linear.weight = deit_model.model.deit.encoder.layer[i].intermediate.dense.weight
        relu_model.encoder[i].ff_linear.bias = deit_model.model.deit.encoder.layer[i].intermediate.dense.bias

        relu_model.encoder[i].out_linear.weight = deit_model.model.deit.encoder.layer[i].output.dense.weight
        relu_model.encoder[i].out_linear.bias = deit_model.model.deit.encoder.layer[i].output.dense.bias

        relu_model.encoder[i].layer_norm1.weight = deit_model.model.deit.encoder.layer[i].layernorm_before.weight
        relu_model.encoder[i].layer_norm1.bias = deit_model.model.deit.encoder.layer[i].layernorm_before.bias

        relu_model.encoder[i].layer_norm2.weight = deit_model.model.deit.encoder.layer[i].layernorm_after.weight
        relu_model.encoder[i].layer_norm2.bias = deit_model.model.deit.encoder.layer[i].layernorm_after.bias

    #copy classification layer
    relu_model.classification.weight = deit_model.model.classifier.weight
    relu_model.classification.bias = deit_model.model.classifier.bias

    #copy last layer norm
    relu_model.layer_norm.weight = deit_model.model.deit.layernorm.weight
    relu_model.layer_norm.bias = deit_model.model.deit.layernorm.bias 

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
    ]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainset_1 = torch.utils.data.Subset(trainset, list(range(0, 50000, 5)))

trainloader = torch.utils.data.DataLoader(trainset_1, batch_size=batch_size,
                                          shuffle=True, num_workers=12)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testset_1 = torch.utils.data.Subset(trainset, list(range(7500)))

testloader = torch.utils.data.DataLoader(testset_1, batch_size=batch_size,
                                         shuffle=False, num_workers=12)

model = DeiTClassifier('facebook/deit-base-distilled-patch16-224', 10)
model.load_state_dict(torch.load('/home/edith/Documents/projects/ViT/saved_model/final_model'))
config = deit_model.model.deit.config

quant_model = ViT(config.num_hidden_layers, config.num_attention_heads, config.hidden_size, 
                            config.intermediate_size, config.hidden_dropout_prob, 3, config.patch_size, config.image_size)

quant_model.copy_model(model)

trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None, 
    max_epochs=15,
    callbacks=[
        TQDMProgressBar(refresh_rate=20),
    ]
)

trainer.fit(quant_model, trainloader, testloader)
torch.save(model.state_dict(), '/home/edith/Documents/projects/ViT/saved_model/dense_relu_deit')
eval(quant_model, testloader)