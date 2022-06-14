gpus = "2"
import os
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from omegaconf import OmegaConf
import argparse

from src.dataset import get_data
from src.model import get_model, get_loss
from src.optimizer import get_optimizer
from src.metrics import get_metric



class Model(pl.LightningModule):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.model = get_model(self.args.model)
        if "load_from" in self.args.model:
            self.load_state_dict(torch.load(self.args.model.load_from, map_location = "cpu")['state_dict'])
        self.criterion = get_loss(self.args.loss)
        self.metric = get_metric(self.args.metric)
        self.save_hyperparameters(args)

    def prepare_data(self):
        (self.ds_train, self.ds_valid, self.ds_test), (self.dl_train, self.dl_valid, self.dl_test) = get_data(self.args.data)

    def train_dataloader(self):
        return self.dl_train()

    def val_dataloader(self):
        return self.dl_valid()

    def predict_dataloader(self):
        return self.dl_test()

    def configure_optimizers(self):
        optimizer, scheduler = get_optimizer(self, self.args.train)
        return [optimizer], [scheduler]

    def on_fit_start(self):
        metric_placeholder = {"valid_metric": 0}
        self.logger.log_hyperparams(self.hparams, metrics = metric_placeholder)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.criterion(yhat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.criterion(yhat, y)
        self.log("valid_loss", loss, prog_bar = True)
        return y, yhat

    def predict_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        return yhat

    def validation_step_end(self, output):
        return output

    def validation_epoch_end(self, outputs):
        y = torch.cat([_[0] for _ in outputs]).detach()
        yhat = torch.cat([_[1] for _ in outputs]).detach()
        for k, v in self.metric(y, yhat).items():
            self.log(k, v, prog_bar = True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest = "config", type = str)
    cmd_args = parser.parse_args().config
    args = OmegaConf.load(cmd_args)
    pl.seed_everything(args.get("seed", 0))
    logger = TensorBoardLogger("./logs", name = args.name, version = args.get("version", None), default_hp_metric = False)
    callbacks = [
        ModelCheckpoint(
            filename = '{epoch}_{valid_metric:.3f}',
            save_last = True,
            save_weights_only = True,
            mode = "max",
            monitor = 'valid_metric'),
        RichProgressBar(leave = True)
    ]

    model = Model(args)
    trainer = pl.Trainer(
        gpus = len(gpus.split(",")), 
        precision = 16, 
        strategy = "dp",
        max_epochs = args.train.num_epochs,
        stochastic_weight_avg = args.train.swa,
        logger = logger,
        callbacks = callbacks
    )
    trainer.fit(model)
