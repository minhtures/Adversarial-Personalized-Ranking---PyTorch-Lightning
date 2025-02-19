import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor 
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
from torchmetrics.retrieval import RetrievalHitRate, RetrievalNormalizedDCG, RetrievalAUROC


def test_model( model, test_dl):
    trainer = Trainer()
    preds = trainer.predict(model, test_dl)
    num_neg = preds[0].size(1) - 1
    preds= torch.cat(preds,dim=0).flatten()
    index = torch.tensor(test_dl.dataset.df['user']).repeat_interleave(num_neg + 1)
    target = torch.tensor([True]+ [False]*num_neg).repeat(len(test_dl.dataset.df))
    
    for top_k in range(1,10):
        metric_hr = RetrievalHitRate(top_k=top_k)
        metric_ndcg = RetrievalNormalizedDCG(top_k=top_k)
        metric_auroc = RetrievalAUROC(top_k=top_k)
        
        hr = metric_hr(preds, target, indexes=index)
        ndcg = metric_ndcg(preds, target, indexes=index)
        auroc = metric_auroc(preds, target, indexes=index)
        
        print(f"top {top_k:d} : hr: {hr:.3f}, ndcg: {ndcg:.3f}, auroc: {auroc:.3f}")
        
    for top_k in range(10,100+1,10):
        metric_hr = RetrievalHitRate(top_k=top_k)
        metric_ndcg = RetrievalNormalizedDCG(top_k=top_k)
        metric_auroc = RetrievalAUROC(top_k=top_k)
    
        
        hr = metric_hr(preds, target, indexes=index)
        ndcg = metric_ndcg(preds, target, indexes=index)
        auroc = metric_auroc(preds, target, indexes=index)
        
        print(f"top {top_k:d} : hr: {hr:.3f}, ndcg: {ndcg:.3f}, auroc: {auroc:.3f}")