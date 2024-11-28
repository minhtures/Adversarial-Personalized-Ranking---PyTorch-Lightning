import torch
from APR import BPR
from dataset import BPRDataModule
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor 
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

import argparse

parser = argparse.ArgumentParser(description='Get args')
parser.add_argument('--config', type=str, default='config/BPR.yaml',
                        help='dataset, model and training config')
parser.add_argument('--pretrained', type=str, default=None,
                        help='get pretrained model')

args = parser.parse_args()
config = OmegaConf.load(args.config)
pretrained = args.pretrained

datamodule = BPRDataModule(**config.dataset)
datamodule.prepare_data()
datamodule.setup(stage="fit")
config.model_config.user_num = datamodule.user_num + 1
config.model_config.item_num = datamodule.item_num + 1

if pretrained is not None:
    model = BPR.load_from_checkpoint(pretrained, **config.model_config)
else:
    model = BPR(**config.model_config)

checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{ndcg:.4f}',
        save_top_k = 5,
        monitor ='ndcg',
        mode='max',
        )

lr_monitor = LearningRateMonitor(logging_interval='step')
callbacks = [checkpoint_callback, lr_monitor]

trainer = Trainer(
    **config.training, 
    callbacks=callbacks,
    logger = TensorBoardLogger(save_dir="exp", name=config.name)
    )

trainer.fit(model, datamodule)