import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import torchmetrics
from copy import deepcopy
import pdb




class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x



class SiameseModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        threshold: float = 0.5,
        criterion: nn.Module = None,
    ):
        super().__init__()
        self.model = model
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.threshold = threshold
        self.learning_rate = learning_rate
        
        self.criterion = criterion
        acc = torchmetrics.Accuracy(task='binary', threshold=self.threshold, average='micro') 
        self.train_acc = acc
        self.val_acc = deepcopy(acc)
        self.test_acc = deepcopy(acc)

    
    def forward(self, x):
        x1, x2 = x
        out1 = self.model(x1)
        out2 = self.model(x2)
        return out1, out2

    def step(self, x1, x2, labels, split):
        v1, v2 = self.forward((x1, x2))
        loss = self.criterion(v1, v2, torch.where(labels==0, -1, 1))
        
        sim = self.cosine_sim(v1, v2)
        acc = getattr(self, f'{split}_acc')
        acc(torch.where(sim>self.threshold, 1, 0), labels)

        self.log_dict({
            f'{split}_loss': loss,
            f'{split}_acc': acc,
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def training_step(self, batch, batch_idx):
        x1, x2, labels = batch
        return self.step(x1, x2, labels, 'train')
    
    def validation_step(self, batch, batch_idx):
        x1, x2, labels = batch
        return self.step(x1, x2, labels, 'val')
    
    def test_step(self, batch, batch_idx):
        x1, x2, labels = batch
        return self.step(x1, x2, labels, 'test')
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=self.trainer.callbacks[0].mode,
            factor=0.2,
            patience=7,
        )

        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.trainer.callbacks[0].monitor,
                'frequency': 1,
                'interval': 'epoch'
            }
        }
    

if __name__ == '__main__':
    model = Net(384, 384, 384)
    siamese_model = SiameseModule(model)
    print(siamese_model)
    
    v1 = torch.randn(10, 384)
    v2 = torch.randn(10, 384)
    labels = torch.randn(10, 1)
    sim = siamese_model((v1, v2))
    loss = F.mse_loss(sim, labels)
    print(sim.shape)
    print(loss)
    pdb.set_trace()