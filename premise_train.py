
import numpy as np
import pandas as pd
import re
import json
from random import random, shuffle, choice

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import BertModel,BertConfig,BertTokenizer,AutoTokenizer,BertConfig,BertTokenizer
from sklearn.model_selection import KFold
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
from torch import nn
#from transformers import AdamW
import torch.optim as optim
from torch.utils.data import Dataset


#logging 
import logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

#read train dataset
train = pd.read_csv('SMM4H/', sep='\t')
test = pd.read_csv('SMM4H/', sep='\t')
# predict = aug_dataset['Premise']
# dataset1['Premise'] = predict
column_names = {'Premise':'labels'}
train = train.rename(columns = column_names)
test = test.rename(columns = column_names)

#column_names = {'Stance':'labels'}
#train = train.rename(columns = column_names)
#train.loc[train["labels"] == "AGAINST", "labels"] = 0
#train.loc[train["labels"] == "FAVOR", "labels"] = 1
#train.loc[train["labels"] == "NONE", "labels"] = 2

tokenizer =  BertTokenizer.from_pretrained('covid-twitter-bert-v2/')
# bert_config = BertConfig.from_pretrained('covid-twitter-bert-v2/')

class My_Dataset(Dataset):
    def __init__(self,train_data,max_length,tokenizer):
        self.data = train_data
        self.max_length = max_length
        self.max_num_tokens = max_length-3
        self.tokenizer = tokenizer
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,idx):
#         input_ids,segment_ids,label = [],[],[]
        text = self.data['Tweet'].iloc[idx]
        label = self.data['labels'].iloc[idx]
        tokens = self.tokenizer.encode_plus(text,
                                add_special_tokens=True,
                                max_length= self.max_length,
                                padding='max_length',
                                truncation = True)
        input_ids = torch.tensor(tokens['input_ids'],dtype=torch.long)
        attention_mask = torch.tensor(tokens['attention_mask'],dtype=torch.long)
        #input_ids = tokenizer.tokenize(text)                                                       
        return input_ids,attention_mask,label


####my_model#####
class MODEL(nn.Module):
    
    def __init__(self, hidden_dim, dropout, name):
        super(MODEL, self).__init__()
        
        self.model_name = name
        
        self.F_obj = BertModel.from_pretrained('covid-twitter-bert-v2', return_dict=True)
        self.F_subj = BertModel.from_pretrained('covid-twitter-bert-v2', return_dict=True)

        self.dropout = dropout
        self.hidden_dim = hidden_dim
    
        # OBJECTIVE VIEW
        self.objective_domain_discriminator = nn.Sequential(
                                        nn.Linear(1024, self.hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(self.hidden_dim, 1)
        )
        
        self.objective_classifier = nn.Sequential(
                                        nn.Linear(1024, self.hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(self.hidden_dim, 2)
        )
        
        # SUBJECTIVE VIEW
        self.subjective_domain_discriminator = nn.Sequential(
                                        nn.Linear(1024, self.hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(self.hidden_dim, 1)
        )
        
        self.subjective_classifier = nn.Sequential(
                                        nn.Linear(1024, self.hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(self.hidden_dim, 2)
        )
        
        # FUSION Layer
        self.g = nn.Sequential(
                        nn.Linear(2048, 1024),
                        nn.Sigmoid()
        )
        
        self.stance_classifier = nn.Sequential(
                                        nn.Linear(1024, self.hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(self.hidden_dim, 2)
        )
        
    def forward(self, input_ids, attention_mask, a=None):
        f_obj = self.F_obj(input_ids, attention_mask)['pooler_output']
        f_subj = self.F_subj(input_ids, attention_mask)['pooler_output']
        
        # FUSION
        f_obj_subj = torch.cat((f_obj, f_subj), dim=1)
        g = self.g(f_obj_subj)
        f_dual = (g * f_subj) + ((1 - g) * f_obj)
        stance_prediction = self.stance_classifier(f_dual)
        
        if a is not None:
            objective_prediction = self.objective_classifier(f_obj)
            subjective_prediction = self.subjective_classifier(f_subj)
            
            reverse_f_obj = f_obj
            objective_domain_prediction = self.objective_domain_discriminator(reverse_f_obj)
            
            reverse_f_subj = f_subj
            subjective_domain_prediction = self.subjective_domain_discriminator(reverse_f_subj)
            
            return stance_prediction, objective_prediction, subjective_prediction, objective_domain_prediction, subjective_domain_prediction
        
        return stance_prediction


seed = 2023
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
max_length = 128
learn_rate = 1e-6
batch_size = 16
model_path = 'trainmodel'
#epochs = 16
epochs = 20
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6'
device = "cuda:1" if torch.cuda.is_available() else "cpu"

from sklearn.metrics import f1_score
test_dataset = My_Dataset(test,max_length,tokenizer)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
train_dataset = My_Dataset(train,max_length,tokenizer)
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
model = MODEL(1024,0.15,"MODEL")
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(),lr=learn_rate,weight_decay=0.01,amsgrad=True)
model.to(device)
best_loss = 100000000.0
best_score = 0.
patience=0
for epoch in range(epochs):
    model.train()
    f1_each_epoch=0.
    #loss_each_epoch = []
    tq = tqdm(train_dataloader, desc="Iteration",ncols=150)
    for step,batch in enumerate(tq):
        batch = tuple(t.to(device) for t in batch)
        input_ids,attention_mask, label = batch
        prdiction = model(input_ids,attention_mask)
        loss = criterion(prdiction,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #loss_each_epoch.append(loss.item())
        
        label_pred=np.argmax(prdiction.detach().to("cpu").numpy(), axis=1)
        
        f1 = f1_score(label.detach().to("cpu").numpy(), label_pred, average='macro')
        tq.set_postfix(epoch=epoch, loss=loss.item(), f1_score=f1) #/label.size()[0]
    
    #eval
    print('======================Begining with testing on evaluation set !======================')
    model.eval()
    label_predict_list = np.array([],dtype=int)
    test_loss = []
    for step,batch in enumerate(tqdm(test_dataloader,ncols=150)):
        batch = tuple(t.to(device) for t in batch)
        input_ids,attention_mask, label = batch
        with torch.no_grad():
            prdiction = model(input_ids,attention_mask)
            loss = criterion(prdiction, label)
            test_loss.append(loss.item())    
        label_pred=np.argmax(prdiction.detach().to("cpu").numpy(), axis=1)
        label_predict_list=np.concatenate([label_predict_list,label_pred])
    f1_each_epoch = f1_score(np.array(test['labels'],dtype=int), label_predict_list, average='macro')
    if f1_each_epoch>best_score:
        print('=============The eval f1 score is increasing, we save the model!=============')
        best_score = f1_each_epoch
        print('the current f1:{},the best f1:{}'.format(f1_each_epoch,best_score))
        torch.save(model.state_dict(),os.path.join(model_path,'./epoch_{}_best_f1_{}.pth'.format(epoch,best_score)))
        patience = 0
    # else:
    #     patience += 1
    #     print('the current f1:{},the best f1:{}'.format(f1_each_epoch,best_score))
    current_loss = np.mean(test_loss)
    if current_loss<best_loss:
        print('=============The train loss is decreasing, we save the model!=============')
        best_loss = current_loss
        print('the current loss:{},the best loss:{}'.format(current_loss,best_loss))
        torch.save(model.state_dict(),os.path.join(model_path,'./epoch_{}_best_loss_{}.pth'.format(epoch,best_loss))) 