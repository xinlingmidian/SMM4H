
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
from transformers import AdamW
from torch.utils.data import Dataset
from sklearn.metrics import f1_score

#logging 
import logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

tokenizer = BertTokenizer.from_pretrained('covid-twitter-bert-v2/')
bert_config = BertConfig.from_pretrained('covid-twitter-bert-v2/')

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
        text = self.data['tweet_text'].iloc[idx]
        #label = self.data['labels'].iloc[idx]
        tokens = self.tokenizer.encode_plus(text,
                                add_special_tokens=True,
                                max_length= self.max_length,
                                padding='max_length',
                                truncation = True)
        input_ids = torch.tensor(tokens['input_ids'],dtype=torch.long)
        attention_mask = torch.tensor(tokens['attention_mask'],dtype=torch.long)
        #input_ids = tokenizer.tokenize(text)                                                       
        return input_ids,attention_mask#,label


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
                                        nn.Linear(self.hidden_dim, 3)
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
learn_rate = 3e-6
batch_size = 4
model_path = 'trainmodel'
epochs = 10
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6'
device = "cuda:1" if torch.cuda.is_available() else "cpu"


#read validation dataset
val = pd.read_csv('SMM4H/', sep='\t')  
model = MODEL(1024,0.15,"MODEL")
model.to(device)
model.load_state_dict(torch.load("trainmodel/.pth"))
model.eval()
eval_dataset = My_Dataset(val,max_length,tokenizer)
eval_dataloader = DataLoader(eval_dataset,batch_size=batch_size,shuffle=False)
predict_list = np.array([],dtype=int)
for step,batch in enumerate(tqdm(eval_dataloader)):
      batch = tuple(t.to(device) for t in batch)
      input_ids,attention_mask = batch
      with torch.no_grad():
            prdiction = model(input_ids,attention_mask)
      label_pred=np.argmax(prdiction.detach().to("cpu").numpy(), axis=1)
      predict_list=np.concatenate([predict_list,label_pred])
val['predictions'] = predict_list           
      
#Mapping back integer labels to words
val.loc[val["predictions"] == 0, "predictions"] = "AGAINST"
val.loc[val["predictions"] == 1, "predictions"] = "FAVOR"
val.loc[val["predictions"] == 2, "predictions"] = "NONE"
#Final submission is need to be in the following format
#val = val.rename(columns={'predictions':'Stance','text':'Tweet'})     
val = val.rename(columns={'predictions':'Stance'})    
#save model predictions to .tsv file
val.to_csv('file_tsv/', sep='\t', index=False)