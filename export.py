import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from transformers import DeiTFeatureExtractor, DeiTModel
from deit_train import DeiTClassifier
from new_quant import QuantViT, fixed_point_rep
from tqdm import tqdm

batch_size = 1

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

def create_embedder_golden_vals(model, dataloader):
    """Saves intermediate values into a text file for comparison"""
    for img, label in dataloader:
        # img = fixed_point_rep(img, int_bits=0, frac_bits=8)
        rearranged = model.embedder.projection[0](img).flatten().tolist()
        embeddings = fixed_point_rep(model.embedder(img), int_bits=6, frac_bits=2).flatten().tolist()
        model.embedder.projection[1].bias = nn.Parameter(torch.zeros_like(model.embedder.projection[1].bias))
        linear_out = fixed_point_rep(model.embedder.projection(img), int_bits=6, frac_bits=6).flatten().tolist()
        break

    assert len(rearranged) == 224*224*3
    assert len(linear_out) == 768*196
    assert len(embeddings) == 768*198

    with open('export/intermediates/embedder_rearranged.txt', 'w') as f:
        for i in rearranged:
            f.write(str(i) + "\n")
    
    with open('export/intermediates/embedder_linear_out.txt', 'w') as f:
        for i in linear_out:
            f.write(str(i) + "\n")

    with open('export/intermediates/embedder_embeddings.txt', 'w') as f:
        for i in embeddings:
            f.write(str(i) + "\n")

def create_mha_goldenvals(model, layer, dataloader):
    for img, label in dataloader:
        embeddings = model.embedder(img)
        # model.encoder[layer].mha.q_linear.bias = nn.Parameter(torch.zeros_like(model.encoder[layer].mha.q_linear.bias))
        print(embeddings[0, 1, :].tolist())
        exit()
        q_proj = fixed_point_rep(model.encoder[layer].mha.q_linear(embeddings).view(1, -1, 12, 64), int_bits=5, frac_bits=3)
        print(q_proj)
        exit()
        k_proj = fixed_point_rep(model.encoder[layer].mha.k_linear(embeddings).view(1, -1, 12, 64).transpose(1, 2), int_bits=5, frac_bits=3)
        v_proj = fixed_point_rep(model.encoder[layer].mha.v_linear(embeddings).view(1, -1, 12, 64).transpose(1, 2), int_bits=5, frac_bits=3)
        break
    
    with open('export/intermediates/mha_embeddings_inp.txt', 'w') as f:
        for i in embeddings.flatten().tolist():
            f.write(str(i) + "\n")
    with open('export/intermediates/q_proj.txt', 'w') as f:
        for i in q_proj.flatten().tolist():
            f.write(str(i) + "\n")
    with open('export/intermediates/k_proj.txt', 'w') as f:
        for i in k_proj.flatten().tolist():
            f.write(str(i) + "\n")
    with open('export/intermediates/v_proj.txt', 'w') as f:
        for i in v_proj.flatten().tolist():
            f.write(str(i) + "\n")
    
def export_mlp(model, layer):
    linear1 = model.encoder[layer].ff_linear
    linear2 = model.encoder[layer].out_linear

    lin1_col_idx = torch.nonzero(linear1.weight)[:, 1].reshape(3072, -1).tolist()
    lin1_nonzero = linear1.weight[linear1.weight != 0].reshape(3072, -1).tolist()
    lin2_col_idx = torch.nonzero(linear2.weight)[:, 1].reshape(768, -1).tolist()
    lin2_nonzero = linear2.weight[linear2.weight != 0].reshape(768, -1).tolist()

    num_nonzero1 = len(lin1_nonzero[0])
    num_nonzero2 = len(lin2_nonzero[0])
    
    cpp = """#ifndef __FUSED_MLP_H__
#define __FUSED_MLP_H__

#include <ap_fixed.h>

static const int num_nonzero1 = %s;
static const int num_nonzero2 = %s;

const ap_fixed<8, 1> linear1_weight[3072][num_nonzero1] = {%s};
const ap_fixed<8, 1> linear2_weight[768][num_nonzero2] = {%s};

const ap_uint<12> linear1_idx[3072][num_nonzero1] = {%s};
const ap_uint<12> linear2_idx[786][num_nonzero2] = {%s};

const ap_fixed<8, 4> linear1_bias[3072] = {%s};
const ap_fixed<8, 4> linear2_bias[768] = {%s};


#endif""" % (
    num_nonzero1, num_nonzero2,
    ",\n".join(["{" + ",".join([str(i) for i in lin1_nonzero[j]]) + "}" for j in range(len(lin1_nonzero))]),
    ",\n".join(["{" + ",".join([str(i) for i in lin2_nonzero[j]]) + "}" for j in range(len(lin2_nonzero))]),
    ",\n".join(["{" + ",".join([str(i) for i in lin1_col_idx[j]]) + "}" for j in range(len(lin1_col_idx))]),
    ",\n".join(["{" + ",".join([str(i) for i in lin2_col_idx[j]]) + "}" for j in range(len(lin2_col_idx))]),
    ",".join([str(i) for i in linear1.bias.tolist()]),
    ",".join([str(i) for i in linear2.bias.tolist()]),
    )

    with open('export/cpp_headers/fused_mlp.h', 'w') as f:
        f.write(cpp)
    

def export_mha(model, layer):
    q_linear = model.encoder[layer].mha.q_linear
    k_linear = model.encoder[layer].mha.k_linear
    v_linear = model.encoder[layer].mha.v_linear
    out_linear = model.encoder[layer].mha.out

    q_weights_col_idx = torch.nonzero(q_linear.weight)[:, 1].reshape(768, -1)
    q_nonzero = q_linear.weight[q_linear.weight != 0].reshape(768, -1)
    k_weights_col_idx = torch.nonzero(k_linear.weight)[:, 1].reshape(768, -1)
    k_nonzero = k_linear.weight[k_linear.weight != 0].reshape(768, -1)
    v_weights_col_idx = torch.nonzero(v_linear.weight)[:, 1].reshape(768, -1)
    v_nonzero = v_linear.weight[v_linear.weight != 0].reshape(768, -1)
    out_weight_col_idx = torch.nonzero(out_linear.weight)[:, 1].reshape(768, -1).tolist()
    out_nonzero = out_linear.weight[out_linear.weight != 0].reshape(768, -1).tolist()

    num_nonzero = int(q_nonzero.shape[1])

    with open('export/q_linear.txt', 'w') as f:
        for i in q_nonzero.flatten().tolist() + q_weights_col_idx.flatten().tolist() + q_linear.bias.flatten().tolist():
            f.write(str(i) + "\n")
    with open('export/k_linear.txt', 'w') as f:
        for i in k_nonzero.flatten().tolist() + k_weights_col_idx.flatten().tolist() + k_linear.bias.flatten().tolist():
            f.write(str(i) + "\n")
    with open('export/v_linear.txt', 'w') as f:
        for i in v_nonzero.flatten().tolist() + v_weights_col_idx.flatten().tolist() + v_linear.bias.flatten().tolist():
            f.write(str(i) + "\n")

    q_nonzero = q_nonzero.tolist()
    k_nonzero = k_nonzero.tolist()
    v_nonzero = v_nonzero.tolist()
    q_weights_col_idx = q_weights_col_idx.tolist()
    k_weights_col_idx = k_weights_col_idx.tolist()
    v_weights_col_idx = v_weights_col_idx.tolist()

    cpp = """#ifndef __MHA_H__
#define __MHA_H__

#include <ap_fixed.h>

static const int num_nonzero = %s;
static const int TILE_SIZE = 16;

const ap_fixed<8, 1> q_linear_weight[768][num_nonzero] = {%s};
const ap_fixed<8, 1> k_linear_weight[768][num_nonzero] = {%s};
const ap_fixed<8, 1> v_linear_weight[768][num_nonzero] = {%s};
const ap_fixed<8, 1> out_linear_weight[768][num_nonzero] = {%s};

const ap_uint<12> q_linear_idx[786][num_nonzero] = {%s};
const ap_uint<12> k_linear_idx[786][num_nonzero] = {%s};
const ap_uint<12> v_linear_idx[786][num_nonzero] = {%s};
const ap_uint<12> out_linear_idx[786][num_nonzero] = {%s};

const ap_fixed<8, 4> q_linear_bias[768] = {%s};
const ap_fixed<8, 4> k_linear_bias[768] = {%s};
const ap_fixed<8, 4> v_linear_bias[768] = {%s};
const ap_fixed<8, 4> out_linear_bias[768] = {%s};


#endif""" % (
    num_nonzero, 
    ",\n".join(["{" + ",".join([str(i) for i in q_nonzero[j]]) + "}" for j in range(len(q_nonzero))]),
    ",\n".join(["{" + ",".join([str(i) for i in k_nonzero[j]]) + "}" for j in range(len(k_nonzero))]),
    ",\n".join(["{" + ",".join([str(i) for i in v_nonzero[j]]) + "}" for j in range(len(v_nonzero))]),
    ",\n".join(["{" + ",".join([str(i) for i in out_nonzero[j]]) + "}" for j in range(len(out_nonzero))]),
    ",\n".join(["{" + ",".join([str(i) for i in q_weights_col_idx[j]]) + "}" for j in range(len(q_weights_col_idx))]),
    ",\n".join(["{" + ",".join([str(i) for i in k_weights_col_idx[j]]) + "}" for j in range(len(k_weights_col_idx))]),
    ",\n".join(["{" + ",".join([str(i) for i in v_weights_col_idx[j]]) + "}" for j in range(len(v_weights_col_idx))]),
    ",\n".join(["{" + ",".join([str(i) for i in out_weight_col_idx[j]]) + "}" for j in range(len(out_weight_col_idx))]),
    ",".join([str(i) for i in q_linear.bias.tolist()]),
    ",".join([str(i) for i in k_linear.bias.tolist()]),
    ",".join([str(i) for i in v_linear.bias.tolist()]),
    ",".join([str(i) for i in out_linear.bias.tolist()]),
    )

    with open('export/cpp_headers/mha'+str(layer)+'.h', 'w') as f:
        f.write(cpp)


"""Format is weights, bias, pos_embedding, cls_token, distill_token"""
def export_embedder(model):
    weights = model.embedder.projection[1].weight
    bias = model.embedder.projection[1].bias
    pos_emb = model.embedder.pos_embed.detach()
    cls_token = model.embedder.cls_token.flatten().tolist()
    distillation_token = model.embedder.distillation_token.flatten().tolist()

    for row in range(2, 198):
        pos_emb[0, row, :] += bias

    with open('export/embedder.txt', 'w') as f:
        for i in weights.flatten().tolist():
            f.write(str(i) + "\n")
        for i in pos_emb.flatten().tolist():
            f.write(str(i) + "\n")

    cpp = """#ifndef __MATMUL_H__
#define __MATMUL_H__

#include <ap_fixed.h>

static const int emb_nonzero = 16;
static const int TILE_SIZE=8;
static const int emb_size = 768;

void emb_matmul(
		ap_fixed<20,20> weight_bram[768][emb_nonzero],
		ap_fixed<8, 4> bias_bram[768],
		ap_fixed<8,0> input_bram[198][768],
		ap_fixed<8, 6> output_bram[198][768]
);

const ap_fixed<8, 0> cls_token[768] = {%s};
const ap_fixed<8, 4> distillation_token[768] = {%s};

#endif
""" % (", ".join([str(i) for i in cls_token]), ", ".join([str(i) for i in distillation_token]))

    with open('export/embedder_matmul.h', 'w') as f:
        f.write(cpp)

def export_validation_set(dataloader):
    for _, (img, label) in enumerate(tqdm(dataloader)):
        img = fixed_point_rep(img, int_bits=0, frac_bits=8)
        np.savetxt('export/images.txt', torch.permute(img, dims=[0, 2, 3, 1]).flatten().numpy())
        np.savetxt('export/labels.txt', label.flatten().int().numpy())
        break

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
    ]
)

tuneset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
tuneset_1 = torch.utils.data.Subset(tuneset, list(range(100)))

tuneloader = torch.utils.data.DataLoader(tuneset_1, batch_size=1,
                                          shuffle=False, num_workers=12)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testset_1 = torch.utils.data.Subset(testset, list(range(1000)))

testloader = torch.utils.data.DataLoader(testset_1, batch_size=1000,
                                         shuffle=False, num_workers=12)

load_path = '/home/edith/Documents/projects/ViT/saved_model/super_extreme_pruned_relu_deit'

deit_model = DeiTClassifier('facebook/deit-base-distilled-patch16-224', 10)
config = deit_model.model.deit.config
model = QuantViT(config.num_hidden_layers, config.num_attention_heads, config.hidden_size, 
            config.intermediate_size, config.hidden_dropout_prob, 3, config.patch_size, config.image_size)
model.load_state_dict(torch.load(load_path))

model.quant_layer(quant_bits=8)
# eval(model, tuneloader) #warmup and get activation statistics
# export_embedder(model)
# export_mha(model, 0)
export_mlp(model, 0)
# create_embedder_golden_vals(model, tuneloader)
# create_mha_goldenvals(model, 0, tuneloader)
# export_validation_set(testloader)