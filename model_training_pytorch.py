import pandas as pd
import numpy as np
import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm


#creating HeadlineData class (inheriting from pytorch Dataset)
class HeadlineData(Dataset):
    def __init__(self,tokenizer, max_len):
        super(HeadlineData, self).__init__()
        self.train_csv = pd.read_csv("./data/headlines_resampled.tsv", sep="\t")
        self.tokenizer = tokenizer
        self.target = self.train_csv["dominant_emotion"].values
        self.max_len = max_len
    
    def __len__(self):
        return len(self.train_csv)
    
    def __getitem__(self, index):
        text = self.train_csv.loc[index, "headline"]
        
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
        )
        
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]
        
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            'target': torch.tensor(self.target[index], dtype=torch.long)
        }
        

#loading in tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

#specifying parameters
max_len = 128
batch_size = 16

#creating the Dataset and DataLoader
train_data = HeadlineData(tokenizer, max_len)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

#creating specialized BERT class

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.out = nn.Linear(768, 6)
        
    def forward(self, ids, mask, token_type_ids):
        _, output = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = self.out(output)
        return output
    
#creating the model
model = BERT()

#defining loss function
loss_fn = nn.BCEWithLogitsLoss()

#initalize optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)


#specify not retraining pretrained BERT model
for param in model.bert_model.parameters():
    param.requires_grad = False
    
#fine-tuning the model
def finetune(epochs, dataloader, model, loss_fn, optimizer):
    for epoch in range(epochs):
        print(epoch)
        loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
        
        for batch, dl in loop:
            ids = dl["ids"]
            mask = dl["mask"]
            token_type_ids = dl["token_type_ids"]
            target = dl["target"]
            target = target.unsqueeze(1)
            
            optimizer.zero_grad()
            
            output = model(ids, mask, token_type_ids, label = target.type_as(output))
            
            loss = loss_fn(output, target.type_as(output))
            loss.backward()
            
            optimizer.step()
            
            pred = np.where(output >= 0, 1, 0)
            
            num_correct = sum(1 for a, b in zip(pred, target) if a[0] == b[0])
            num_samples = pred.shape[0]
            acc = num_correct / num_samples
            
            print(f"Epoch {epoch}, Loss: {loss.item()}, Accuracy: {acc}")
            
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item(), acc=acc)
            
    return model

#training the model
model = finetune(5, train_loader, model, loss_fn, optimizer)
            
    
            




