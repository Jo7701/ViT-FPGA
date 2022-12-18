import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.utils.prune as prune
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelPruning
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from transformers import DeiTFeatureExtractor, DeiTModel, DeiTForImageClassification
from deit_train import DeiTClassifier
import math

#model and training hyperparams
warmup_epochs = 1
pruning_epochs = 10
finetune_epochs = 3
total_epochs = warmup_epochs + pruning_epochs + finetune_epochs
callback_epoch = 0

batch_size = 8
lr = 5e-5
num_classes = 10

class ViTLayerPrune(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def __init__(self):
        super().__init__()

    def compute_mask(self, t, default_mask):
        global use_global_mask
        global global_saved_mask
        global callback_epoch, warmup_epochs, finetune_epochs
        global all_qkv_nonzero, all_dense_nonzero, all_intermediate_nonzero, all_output_nonzero, num_layers

        if not (warmup_epochs <= callback_epoch < total_epochs-finetune_epochs):
            callback_epoch += 1
            return default_mask

        # if use_global_mask:
        #     use_global_mask = False
        #     return global_saved_mask
        # else:
        #     return default_mask

        mask = torch.empty(0).cuda()

        prev_layer_boundary = 0
        for layer in range(num_layers):
            layer_size = 3*hidden_size*all_qkv_nonzero[layer] + hidden_size*all_dense_nonzero[layer] + hidden_size*all_intermediate_nonzero[layer] + intermediate_size*all_output_nonzero[layer]
            layer_mask = self.compute_layer_mask(layer, t[prev_layer_boundary:prev_layer_boundary+layer_size], default_mask[prev_layer_boundary:prev_layer_boundary+layer_size])

            mask = torch.cat([mask, layer_mask], dim=-1)

            prev_layer_boundary += layer_size

        callback_epoch += 1

        return mask.flatten()

    def compute_layer_mask(self, layer, t, default_mask):
        global callback_epoch, warmup_epochs, pruning_epochs, finetune_epochs
        global all_qkv_nonzero, all_dense_nonzero, all_intermediate_nonzero, all_output_nonzero
        global all_qkv_pruned, all_dense_pruned, all_intermediate_pruned, all_output_pruned
        global prune_schedule


        qkv_nonzero = all_qkv_nonzero[layer]
        dense_nonzero = all_dense_nonzero[layer]
        intermediate_nonzero = all_intermediate_nonzero[layer]
        output_nonzero = all_output_nonzero[layer]

        #total prune rate at each epoch is described by the function y = rate/log(pruning_epochs+1) * log(curr_epoch+1)

        qkv_prune_per_epoch = int(prune_schedule[layer] / math.log(pruning_epochs+1) * math.log((callback_epoch+1-warmup_epochs)+1) * hidden_size - all_qkv_pruned[layer])
        dense_prune_per_epoch = int(prune_schedule[layer] / math.log(pruning_epochs+1) * math.log((callback_epoch+1-warmup_epochs)+1) * hidden_size - all_dense_pruned[layer])  
        intermediate_prune_per_epoch = int(prune_schedule[layer] / math.log(pruning_epochs+1) * math.log((callback_epoch+1-warmup_epochs)+1) * intermediate_size - all_intermediate_pruned[layer])
        output_prune_per_epoch = int(prune_schedule[layer] / math.log(pruning_epochs+1) * math.log((callback_epoch+1-warmup_epochs)+1) * hidden_size - all_output_pruned[layer])

        qkv = t[:3*hidden_size*qkv_nonzero].view(3, hidden_size, qkv_nonzero)
        dense = t[3*hidden_size*qkv_nonzero:3*hidden_size*qkv_nonzero+hidden_size*dense_nonzero].view(hidden_size, dense_nonzero)
        intermediate = t[3*hidden_size*qkv_nonzero+hidden_size*dense_nonzero:-intermediate_size*output_nonzero].view(hidden_size, intermediate_nonzero)
        output = t[-intermediate_size*output_nonzero:].view(intermediate_size, output_nonzero)

        qkv_mask = default_mask[:3*hidden_size*qkv_nonzero].view(3, hidden_size, qkv_nonzero)
        qkv_mask = (qkv_mask==1) ^ (qkv==0)
        dense_mask = default_mask[3*hidden_size*qkv_nonzero:3*hidden_size*qkv_nonzero+hidden_size*dense_nonzero].view(hidden_size, dense_nonzero)
        dense_mask = (dense_mask==1) ^ (dense==0)
        intermediate_mask = default_mask[3*hidden_size*qkv_nonzero+hidden_size*dense_nonzero:-intermediate_size*output_nonzero].view(hidden_size, intermediate_nonzero)
        intermediate_mask = (intermediate_mask==1) ^ (intermediate==0)
        output_mask = default_mask[-intermediate_size*output_nonzero:].view(intermediate_size, output_nonzero)
        output_mask = (output_mask==1) ^ (output==0)

        qkv = qkv.masked_fill(qkv_mask==0, 1e9)
        dense = dense.masked_fill(dense_mask==0, 1e9)
        intermediate = intermediate.masked_fill(intermediate_mask==0, 1e9)
        output = output.masked_fill(output_mask==0, 1e9)

        qkv_indeces = torch.topk(torch.abs(qkv), qkv_prune_per_epoch, dim=-1, largest=False)[1]
        dense_indeces = torch.topk(torch.abs(dense), dense_prune_per_epoch, dim=-1, largest=False)[1]
        intermediate_indeces = torch.topk(torch.abs(intermediate), intermediate_prune_per_epoch, dim=-1, largest=False)[1]
        output_indeces = torch.topk(torch.abs(output), output_prune_per_epoch, dim=-1, largest=False)[1]

        del qkv
        del dense
        del intermediate
        del output

        for batch, mask_2d in enumerate(qkv_indeces):
            for row, idx in enumerate(mask_2d):
                qkv_mask[batch, row, idx] = 0

        for row, idx in enumerate(dense_indeces):
            dense_mask[row, idx] = 0

        for row, idx in enumerate(intermediate_indeces):
            intermediate_mask[row, idx] = 0

        for row, idx in enumerate(output_indeces):
            output_mask[row, idx] = 0

        # print(new_mask, new_mask.shape, (new_mask==0).sum().item())

        all_qkv_nonzero[layer] -= qkv_prune_per_epoch
        all_dense_nonzero[layer] -= dense_prune_per_epoch
        all_intermediate_nonzero[layer] -= intermediate_prune_per_epoch
        all_output_nonzero[layer] -= output_prune_per_epoch

        all_qkv_pruned[layer] += qkv_prune_per_epoch
        all_dense_pruned[layer] += dense_prune_per_epoch
        all_intermediate_pruned[layer] += intermediate_prune_per_epoch
        all_output_pruned[layer] += output_prune_per_epoch

        new_mask = torch.cat([qkv_mask.flatten(), dense_mask.flatten(), intermediate_mask.flatten(), output_mask.flatten()], dim=-1)

        return new_mask.flatten()

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


def update_global_mask(params_to_prune):
    global global_saved_mask
    prev_layer_boundary = 0
    for param in params_to_prune:
        zero_mask = (param[0].weight==0).flatten()
        curr_layer_size = len(param[0].weight.flatten())

        global_saved_mask[prev_layer_boundary:prev_layer_boundary+curr_layer_size] = zero_mask

        prev_layer_boundary += curr_layer_size

def load_model_and_prune(load_model_path, save_model_path=None, save=False):
    global use_global_mask
    global all_qkv_nonzero, all_dense_nonzero, all_intermediate_nonzero, all_output_nonzero, num_layers, hidden_size, intermediate_size
    global all_qkv_pruned, all_dense_pruned, all_intermediate_pruned, all_output_pruned
    global prune_schedule
    global global_saved_mask

    model = DeiTClassifier('facebook/deit-base-distilled-patch16-224', 10)
    model.load_state_dict(torch.load(load_path))
    hidden_size = model.model.config.hidden_size
    num_layers = model.model.config.num_hidden_layers
    intermediate_size = model.model.config.intermediate_size

    prune_schedule = [0.75, 0.75, 0.85, 0.85, 0.85, 0.94, 0.95, 0.96, 0.97, 0.98, 0.98, 0.98]

    all_qkv_pruned = [0 for _ in prune_schedule]
    all_dense_pruned = [0 for _ in prune_schedule]
    all_intermediate_pruned= [0 for _ in prune_schedule]
    all_output_pruned = [0 for _ in prune_schedule]

    all_qkv_nonzero = [hidden_size] * num_layers
    all_dense_nonzero = [hidden_size] * num_layers
    all_intermediate_nonzero = [intermediate_size] * num_layers
    all_output_nonzero = [hidden_size] * num_layers

    params_to_prune = [
        j for i in range(12) for j in [
            (model.model.deit.encoder.layer[i].attention.attention.query, "weight"),
            (model.model.deit.encoder.layer[i].attention.attention.key, "weight"),
            (model.model.deit.encoder.layer[i].attention.attention.value, "weight"),
            (model.model.deit.encoder.layer[i].attention.output.dense, "weight"), 
            (model.model.deit.encoder.layer[i].intermediate.dense, "weight"),
            (model.model.deit.encoder.layer[i].output.dense, "weight")
        ]
    ]

    total_params = sum([len(i[0].weight.flatten()) for i in params_to_prune])
    global_saved_mask = torch.ones(total_params).cuda()

    params_to_prune = [
    j for i in range(12) for j in [
            (model.model.deit.encoder.layer[i].attention.attention.query, "weight"),
            (model.model.deit.encoder.layer[i].attention.attention.key, "weight"),
            (model.model.deit.encoder.layer[i].attention.attention.value, "weight"),
            (model.model.deit.encoder.layer[i].attention.output.dense, "weight"), 
            (model.model.deit.encoder.layer[i].intermediate.dense, "weight"),
            (model.model.deit.encoder.layer[i].output.dense, "weight")
        ]
    ]

    use_global_mask = True
    update_global_mask(params_to_prune)
    print((global_saved_mask==0).sum().item(), " zeros in mask")

    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None, 
        max_epochs=total_epochs,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            ModelPruning(
                pruning_fn = ViTLayerPrune,
                parameters_to_prune=params_to_prune,
                verbose=1,
            )
        ]
    )

    trainer.fit(model, trainloader, testloader)
    torch.save(model.state_dict(), save_path)

def report_pruned_stats(model):
    total = 0
    zeros = 0
    for param in model.parameters():
        total += len(param.flatten())
        zeros += (param==0).sum().item()

    print("{}/{} model parameters pruned. {} nonzero params. {}x model compression".format(zeros, total, total-zeros, total/(total-zeros)))

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda a: a*255)
    ]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainset_1 = torch.utils.data.Subset(trainset, list(range(0, 50000, 25)))

trainloader = torch.utils.data.DataLoader(trainset_1, batch_size=batch_size,
                                          shuffle=True, num_workers=12)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testset_1 = torch.utils.data.Subset(testset, list(range(7500)))

testloader = torch.utils.data.DataLoader(testset_1, batch_size=batch_size,
                                         shuffle=False, num_workers=12)

load_path = '/home/edith/Documents/projects/ViT/saved_model/pruned_model'
save_path = '/home/edith/Documents/projects/ViT/saved_model/extreme_pruned_model'

# for _ in range(4):
#     callback_epoch = 0
#     load_model_and_prune(load_path, save_path, True)
#     model = DeiTClassifier('facebook/deit-base-distilled-patch16-224', 10)
#     model.load_state_dict(torch.load(save_path))
#     report_pruned_stats(model)  

#     load_path = save_path
#     break

model = DeiTClassifier('facebook/deit-base-distilled-patch16-224', 10)
model.load_state_dict(torch.load(save_path))


eval(model, testloader)