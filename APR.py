import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


def bpr_loss(prediction):
    prediction_i = prediction[:,0].reshape(-1,1)
    prediction_j = prediction[:,1:]
    loss = - torch.nn.functional.logsigmoid(prediction_i - prediction_j).mean()
    return loss

def batch_NDCG(prediction, top_k=10):
    """
    prediction: num_user x num_item
    each user has one postive item, indicated by 'positive'
    NDCG = log(2)/log(rank+1) if rank <=topk
    """
    positive_score = prediction[:,0].reshape(-1,1)
    rank = (prediction - positive_score) > 0
    rank = rank.to(torch.int).sum(dim=-1)            # rank >=0

    mask = (rank < top_k).to(torch.int)         # NDCG <0 if rank>top_k  
    NDCG = torch.log(torch.tensor(2.0)) / (torch.log(rank + 2.0))
    NDCG *= mask

    return NDCG.mean()

class BPR(LightningModule):
    def __init__(self, user_num, item_num, embed_size, optimizer=None, scheduler=None,  
                 top_k=10, eps=0.0, reg = 0.0, reg_adv=0.0):
        super().__init__()
        """
        user_num: number of users;
        item_num: number of items;
        embed_size: number of predictive factors.
        """        
        self.user_num = user_num
        self.item_num = item_num
        self.embed_size = embed_size
        self.top_k=top_k
        self.eps=eps
        self.reg=reg
        self.reg_adv=reg_adv

        self.embed_user = nn.Embedding(self.user_num, self.embed_size)
        self.embed_item = nn.Embedding(self.item_num, self.embed_size)
        
        self.automatic_optimization = False
        
        self.optimizer_config = optimizer
        self.schedulerr_config = scheduler
        self.save_hyperparameters()
     
    def forward(self, user, item):
        user_embed = self.embed_user(user)       # N x in
        item_embed = self.embed_item(item)       # N x (1+neg) x in
        # N x 1 x in  * N x in x (1+neg) => N x (1+neg)
        prediction = torch.bmm(user_embed.unsqueeze(1), item_embed.transpose(1,2)).squeeze(1)
            
        return prediction, user_embed, item_embed
    
    @torch.no_grad()
    def _get_adversarial(self, user, item, user_embed, item_embed):
        grad_user = torch.index_select(self.embed_user.weight.grad.clone(), 0, user) 
        grad_item = torch.index_select(self.embed_item.weight.grad.clone(), 0, item.flatten()).reshape(item.size(0), item.size(1),-1)

        # normalization: new_grad = (grad / |grad|) * eps
        delta_user = self.eps * nn.functional.normalize(grad_user, p=2, dim=-1)
        delta_item = self.eps * nn.functional.normalize(grad_item, p=2, dim=-1)

        delta_user = torch.max(user_embed.abs(), dim=0, keepdim=True)[0] * delta_user
        delta_item = torch.max(item_embed.abs(), dim=0, keepdim=True)[0] * delta_item

        return delta_user, delta_item

    def training_step(self, batch):
        user, item = batch
        opt = self.optimizers()
        opt.zero_grad()
        
        self.embed_user.weight.retain_grad()
        self.embed_item.weight.retain_grad()
        
        prediction, user_embed, item_embed = self(user,item)
        
        loss = bpr_loss(prediction)
        self.log("train_loss", loss, prog_bar=True)
        reg = torch.linalg.norm(user_embed, dim =-1).mean() + torch.linalg.norm(item_embed, dim =-1).mean()
        self.log("train_reg_loss", reg, prog_bar=True)
        loss += self.reg * reg
        
        if self.reg_adv > 0:
            self.manual_backward(loss,retain_graph=True)    
            delta_user, delta_item = self._get_adversarial(user, item, user_embed, item_embed)
            
            user_adv = user_embed + delta_user
            item_adv = item_embed + delta_item
            
            pred_adv = torch.bmm(user_adv.unsqueeze(1), item_adv.transpose(1,2)).squeeze(1)
            
            loss_adv = bpr_loss(pred_adv)
            self.log("train_adv_loss", loss_adv, prog_bar=True)
            reg_adv = torch.linalg.norm(user_adv, dim =-1).mean() + torch.linalg.norm(item_adv, dim =-1).mean()
            self.log("train_reg_adv_loss", reg_adv, prog_bar=True)
            
            loss_adv = self.reg_adv*(loss_adv + self.reg * reg_adv)
            self.manual_backward(loss_adv)  
        else:
            self.manual_backward(loss)  
            
        opt.step()
        return loss
    
    def validation_step(self, batch, batch_idx):
        user, item = batch
        
        prediction, user_embed, item_embed = self(user,item)
        
        loss = bpr_loss(prediction)
        self.log("bpr_loss", loss, prog_bar=True)
        reg = torch.linalg.norm(user_embed, dim =-1).mean() + torch.linalg.norm(item_embed, dim =-1).mean()
        self.log("reg_loss", reg, prog_bar=True)
        loss += self.reg * reg
        
        ndcg = batch_NDCG(prediction, self.top_k)
        self.log("ndcg", ndcg, prog_bar=True)
        return loss 
      
    def predict_step(self, batch):
        user, item = batch
        prediction, user_embed, item_embed = self(user,item)
        return prediction

    def configure_optimizers(self):
        if self.optimizer_config is None:
            optimizer = torch.optim.AdamW(self.parameters())
        else:
            optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_config)
        if self.schedulerr_config is None:
            return optimizer
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **self.schedulerr_config)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step'
                }
            }