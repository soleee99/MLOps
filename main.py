import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch


import wandb

from model import ColaModel
from datamodule import DataModule


if __name__ == '__main__':
    print("Initializing WandB")
    wandb.init(project="MLOps Basics", entity="soleee99")
    wandb_logger = WandbLogger(project="MLOps Basics")

    #cola_dataset = datasets.load_dataset("glue", "cola")
    print("Initialize DataModule")
    cola_data = DataModule()

    print("Initialize Model (BERT)")
    cola_model = ColaModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )

    trainer = pl.Trainer(
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=3,
        fast_dev_run=False,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(cola_model, cola_data)
