import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning import Trainer, LightningModule
from model import ViT
from tqdm import tqdm
from transformers import DeiTFeatureExtractor, DeiTModel, DeiTForImageClassification
from PIL import Image

#model and training hyperparams
epochs = 3
batch_size = 32
lr = 5e-5
num_classes = 10

class DeiTClassifier(LightningModule):
    def __init__(self, model_type, num_classes):
        super().__init__()
        self.model = DeiTForImageClassification.from_pretrained(model_type, num_labels=num_classes)

    def forward(self, images):
        return self.model(images).logits

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        output = self(imgs)

        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(output, labels)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            imgs, labels = batch
            output = self(imgs)

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output, labels)

            preds = torch.argmax(output, dim=-1)
            correct = (preds == labels).sum().item()

            accuracy = correct / len(labels)

            self.log("val_acc", accuracy, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=lr)


def load_pretrained(model_type):
    feature_extractor = DeiTFeatureExtractor.from_pretrained(model_type)
    model = DeiTModel.from_pretrained(model_type)
    model.pooler = None

    return feature_extractor, model

def train(model, trainloader, testloader):
    # Train the model
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None, 
        max_epochs=epochs,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
    )
    trainer.fit(model, trainloader, testloader)
    torch.save(model.state_dict(), '/home/edith/Documents/projects/ViT/saved_model/vit_model')

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
            output = model(images).logits
            pred_y = torch.argmax(output, dim=-1)
            correct += (pred_y == labels).sum().item()
            total += float(labels.size(0))
    print('Test Accuracy: %.4f' % (correct/total))

def load_model(model, path):
    model.load_state_dict(torch.load(path))

if __name__ == "__main__":
        
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
        ]
    )

    #=======================================IMAGENET=======================================#
    # trainset = torchvision.datasets.ImageFolder(root='./imagenet/train', transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=12)
    # testset = torchvision.datasets.ImageFolder(root='./imagenet/val', transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=12)
    #=======================================IMAGENET=======================================#

    #=======================================CIFAR10=======================================#
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainset_1 = torch.utils.data.Subset(trainset, list(range(5000)))

    trainloader = torch.utils.data.DataLoader(trainset_1, batch_size=batch_size,
                                            shuffle=True, num_workers=12)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testset_1 = torch.utils.data.Subset(testset, list(range(5000)))
    testloader = torch.utils.data.DataLoader(testset_1, batch_size=batch_size,
                                            shuffle=False, num_workers=12)
    #=======================================CIFAR10=======================================#

    model = DeiTClassifier('facebook/deit-base-distilled-patch16-224', 10)
    # model.load_state_dict(torch.load('/home/edith/Documents/projects/ViT/saved_model/vit_model'))

    train(model, trainloader, testloader)
