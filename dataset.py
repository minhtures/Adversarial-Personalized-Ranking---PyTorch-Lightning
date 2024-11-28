import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import pandas as pd

class BPRData(Dataset):
    def __init__(self, df, num_neg=1):
        super(BPRData, self).__init__()
        self.df = df
        if num_neg is not None:
            self.num_neg= num_neg
        else:
            self.num_neg=0
        
        # load ratings as a dok matrix
        self.features = self.df.values

        self.user_num = self.features[:,0].max() + 1 # 1 for unknown
        self.item_num = self.features[:,1:].max() + 1

        self.pos_item = {}
        if self.num_neg > 0:
            for user in range(self.user_num):
                self.pos_item[user] = self.df[self.df['user']==user]['item'].tolist()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        user = self.features[idx][0]
        item = self.features[idx][1:] # positive
        
        user = max(0, min(user,self.user_num))
        item = [max(0, min(i,self.item_num)) for i in item]
        
        if len(item) == 1 and self.num_neg >0:
            pos = self.pos_item[user]
            pos_i = torch.randint(0, len(pos)+1, (self.num_neg,))
            pos.append(0)
            pos.append(self.item_num)
            pos, _  = torch.sort(torch.tensor(pos))
            for j in pos_i:
                low = pos[j]
                high = pos[j+1]
                if low < high:
                    item_j = int(torch.randint(low, high,(1,))[0])
                else:
                    item_j = 0       # <UNK>
                item.append(item_j)
        return user, torch.tensor(item)
    
def collate_data(batch):
    user, item  = zip(*batch)
    user = torch.tensor(user)
    item = torch.nn.utils.rnn.pad_sequence(item, batch_first=True)
    item= torch.nan_to_num(item)

    return user, item

class BPRDataModule(LightningDataModule):
    def __init__(self, data_dir, tr_neg=1, val_neg=99, tr_bs=128, val_bs=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.tr_neg = tr_neg
        self.val_neg = val_neg
        self.tr_bs = tr_bs
        self.val_bs = val_bs
        self.num_workers = num_workers
        
    def prepare_data(self):    
        self.df_train = pd.read_csv(self.data_dir+'.train.rating', usecols=[0,1], sep='\t', names = ['user', 'item'])
        self.df_val = pd.read_csv(self.data_dir+'.test.rating', usecols=[0,1], sep='\t', names = ['user', 'item'])
        self.df_test = pd.read_csv(self.data_dir+'.test.negative', sep='\t', names = ['user', 'item']+[f'item_neg_{i+1}' for i in range(99)])
        self.df_test['user'] = [t[1] for t in self.df_test['user']]
        
        self.user_num = self.df_train['user'].max()
        self.item_num = self.df_train['item'].max()
        self.get_df_info()
        
    def get_df_info(self):    
        print(self.df_train.describe())
        # print(self.df_train.head())
        # print(self.df_val.head())
        # print(self.df_test.head())
        
        number_interaction = len(self.df_train)
        sparsity = 100 - 100.0*number_interaction/ (self.user_num*self.user_num)
        print(f'Number of users: {self.user_num}')
        print(f'Number of items: {self.item_num}')
        print(f'Number of interactions: {number_interaction}')
        print(f'Sparsity: {sparsity:6f} %')

    def setup(self, stage: str):
        if stage == "fit":
            self.train_ds = BPRData(self.df_train, num_neg=self.tr_neg)
            self.val_ds = BPRData(self.df_val, num_neg=self.val_neg)
        if stage == "test" or stage == "predict":
            self.test_ds = BPRData(self.df_val, num_neg=99)
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.tr_bs, 
                        persistent_workers=True,
                        num_workers=self.num_workers ,shuffle=True, collate_fn = collate_data)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.val_bs, 
                        persistent_workers=True,
                        num_workers=self.num_workers ,shuffle=False, collate_fn = collate_data)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.val_bs, 
                        persistent_workers=True,
                        num_workers=self.num_workers ,shuffle=False, collate_fn = collate_data)