from pathlib import Path
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import lightning as L
import torchvision
from torchvision import models
from torchvision.transforms import v2
from torch import nn, optim
from torchmetrics import MetricCollection 
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.classification import Accuracy, Recall, Precision, F1Score, Specificity, ConfusionMatrix, ROC, AUROC
import time
from colorama import Fore, Back, Style

torch.manual_seed(42)
torch.set_float32_matmul_precision('high')

lr=1e-3
loss_function = nn.CrossEntropyLoss()
ticklabels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
EPOCHS = 1
NUM_CLASSES = 10
BATCH_SIZE = 32
CHECKPOINTS_DIR = Path('./multiplegpus_logs')

multiclass_metrics = MetricCollection({
    'acc': Accuracy('multiclass', num_classes=NUM_CLASSES),
    'acc_per_class': Accuracy('multiclass', num_classes=NUM_CLASSES, average=None),
    'recall': Recall('multiclass', num_classes=NUM_CLASSES),
    'recall_per_class': Recall('multiclass', num_classes=NUM_CLASSES, average=None),
    'precision': Precision('multiclass', num_classes=NUM_CLASSES),
    'precision_per_class': Precision('multiclass', num_classes=NUM_CLASSES, average=None),
    'f1': F1Score('multiclass', num_classes=NUM_CLASSES),
    'f1_per_class': F1Score('multiclass', num_classes=NUM_CLASSES, average=None),
    'specificity': Specificity('multiclass', num_classes=NUM_CLASSES),
    'specificity_per_class': Specificity('multiclass', num_classes=NUM_CLASSES, average=None),
    # 'auroc': AUROC('multiclass', num_classes=NUM_CLASSES),
    # 'auroc_per_class': AUROC('multiclass', num_classes=NUM_CLASSES, average=None),
    # 'roccurve_to_plot': ROC('multiclass', num_classes=NUM_CLASSES),
    'cm-to-plot': ConfusionMatrix('multiclass', num_classes=NUM_CLASSES)
})

roccurve = ROC('multiclass', num_classes=NUM_CLASSES)
aucroc = AUROC('multiclass', num_classes=NUM_CLASSES)


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(20, 10))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.0001)


class ClassificationModule(L.LightningModule):
    def __init__(self, pretrained_model, loss_function, optimizer):
        super().__init__()
        self.model = pretrained_model
        self.optimizer = optimizer
        self.loss_module = loss_function
        self.test_step_outputs = []
        self.training_step_outputs = []
        self.metrics_train = multiclass_metrics.clone(prefix='train-')
        self.metrics_test = multiclass_metrics.clone(prefix='test-')
        
    
    def shared_step(self, batch, stage):
        inputs, classes = batch
        preds = self.model(inputs)
        loss = self.loss_module(preds, classes)
                
        outputs = {
            "loss": loss,
            "labels": classes,
            "preds": preds,
        }
        if stage == 'train':
            self.training_step_outputs.append(outputs)
        return outputs


    def shared_epoch_end(self, outputs, stage):
        labels = torch.cat([x["labels"] for x in outputs])
        preds_prob = torch.cat([x["preds"] for x in outputs])
        preds = torch.max(preds_prob, dim=1).indices

        self.loss_ = self.loss_module(preds_prob, labels)

        
        if stage == 'test':
            tensorboard = self.logger.experiment
            metrics = self.metrics_test(preds, labels)
            metrics['test-loss'] = self.loss_
            
            # confusion matrix
            fig, ax = plt.subplots()
            cm = metrics['test-cm-to-plot']
            sns.heatmap(cm.cpu().numpy(), annot=True, cmap=sns.color_palette("mako", as_cmap=True).reversed(), xticklabels=ticklabels,
                        yticklabels=ticklabels, ax=ax, fmt='d')
            plt.ylabel('True Labels')
            plt.xlabel('Predict')
            plt.savefig(f'{CHECKPOINTS_DIR}/figs/fig_cm.png')
            plt.show()
            tensorboard.add_figure(f'plot-Confusion matrix (test)', fig)
            
            
            # roc curve
            metrics['test-aucroc'] = aucroc(preds_prob, labels)
            metrics['test-roccurve-to-plot'] = roccurve(preds_prob, labels)
            fig_, ax_ = roccurve.plot(metrics['test-roccurve-to-plot'], score=True)  
            fig_.savefig(f'{CHECKPOINTS_DIR}/figs/fig_ROC.png')
            tensorboard.add_figure(f'plot-ROC curve (test)', fig_)
        else:
            metrics = self.metrics_train(preds, labels)
            metrics['train-loss'] = self.loss_
            metrics['train-aucroc'] = aucroc(preds_prob, labels)
            metrics['train-roccurve-to-plot'] = roccurve(preds_prob, labels)

        global_metrics = {key: value for key, value in metrics.items() if not '_' in key and not 'to' in key}
        self.log_dict(global_metrics, prog_bar=True)
        if NUM_CLASSES>2:
            metrics_per_class = {key: value for key, value in metrics.items() \
                                 if '_' in key and not 'to' in key}
            # CONFIGURAR O LOG DE METRICAS POR CLASSE


    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")


    def on_train_epoch_end(self):
        outputs = self.training_step_outputs.copy()
        self.training_step_outputs.clear()
        return self.shared_epoch_end(outputs,"train") 


    def test_step(self, batch, batch_idx):
        self.test_step_outputs.append(self.shared_step(batch, "test"))
        return self.shared_step(batch, "test")  


    def on_test_epoch_end(self):
        return self.shared_epoch_end(self.test_step_outputs, "test")


    def on_test_end(self):
        self.test_step_outputs.clear()
        

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=lr)


# Transfer learning (fine tuning):
# Instanciação VGG16_BN
weights = models.VGG16_BN_Weights.DEFAULT # melhores pesos até o momento
vgg16 = models.vgg16_bn(weights=weights)
# transformações específicas da rede
vgg16_preprocess = weights.transforms()
# modificando a camada de classificação
num_ftrs = vgg16.classifier[-1].in_features
vgg16.classifier[-1] = nn.Linear(num_ftrs, 10)
vgg16.classifier


data_transforms = {
    'train': v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    'test': v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224)),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


train_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, transform=data_transforms['train'], download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, transform=data_transforms['test'], download=True)

class_names = train_dataset.classes

# Definição dos dataloaders
trainloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=15) # AVALIAR
testloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=15) # AVALIAR

# plot
# print('###  PLOT - EXAMPLE  ###')
# inputs, classes = next(iter(trainloader))
# out_test = torchvision.utils.make_grid(inputs)
# imshow(out_test, title=[class_names[x] for x in classes])


model = vgg16
logger = TensorBoardLogger('multiplegpus_logs')
# DEBUG:
# trainer = L.Trainer(fast_dev_run=8) # a execução do trainer se limitará a fast_dev_run batchs
trainer = L.Trainer(accelerator='gpu', max_epochs=EPOCHS, logger=logger)


# treino
star_train = time.time()
lighting_model = ClassificationModule(model, loss_function, torch.optim.Adam)
trainer.fit(model=lighting_model, train_dataloaders=trainloader)
end_train = time.time()
print(Fore.GREEN + f'TEMPO DE TREINAMENTO: {end_train - star_train} s')

# teste 
star_test = time.time()
trainer.test(model=lighting_model, dataloaders=testloader)
end_test = time.time()
print(Fore.GREEN + f'TEMPO DE TESTE: {end_test - star_test} s')
