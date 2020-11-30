import os
import time
import torch
import random
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, AdamW, BertForSequenceClassification,XLMRobertaTokenizer
import torch.nn as nn
from sklearn.metrics import f1_score
from transformers import XLMRobertaModel

from attention import AttentionModel

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

tokenizer =XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
bert = XLMRobertaModel.from_pretrained('xlm-roberta-base')
# bert = BertModel.from_pretrained("bert-base-multilingual-cased")

'''confusion'''
# bert = BertForSequenceClassification.from_pretrained(
#     "bert-base-multilingual-cased",
#     num_labels = 3, 
#     output_attentions = False,
#     output_hidden_states = False,
# )

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

max_input_length = 56

######################
df = pd.read_csv("/Users/taseefrahman/Documents/CS 695/codemix/data/Hinglish_train_14k_split_conll.tsv", delimiter='\t', header=None, names=['sentence','label'])
# print(df)



sentences = df.sentence.values
labels = df.label.values
input_ids = []
attention_masks = []


for sent in sentences:
    encoded_dict = tokenizer.encode_plus(
                        sent,
                        add_special_tokens = True,
                        max_length = max_input_length,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        truncation = True,
                   )
      
    input_ids.append(encoded_dict['input_ids'])
    
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

train_dataset = TensorDataset(input_ids, attention_masks,labels)

# print("ip id len is")
# print(input_ids[0])
######################
df = pd.read_csv("/Users/taseefrahman/Documents/CS 695/codemix/data/Hinglish_dev_3k_split_conll.tsv", delimiter='\t', header=None, names=['sentence','label'])

print('Number of validating sentences: {:,}\n'.format(df.shape[0]))

sentences = df.sentence.values
labels = df.label.values
input_ids = []
attention_masks = []


for sent in sentences:
    encoded_dict = tokenizer.encode_plus(
                        sent,
                        add_special_tokens = True,
                        max_length = max_input_length,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        truncation = True,
                   )
      
    input_ids.append(encoded_dict['input_ids'])
    
    attention_masks.append(encoded_dict['attention_mask'])


input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

val_dataset = TensorDataset(input_ids, attention_masks, labels)
######################

OUTPUT_DIM = 3
DROPOUT = 0.3
# N_FILTERS = 100
# FILTER_SIZES = [2,3,4]
HIDDEN_DIM = 100
BATCH_SIZE=10
######################
train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = BATCH_SIZE,
            drop_last=True
        )
validation_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset),
            batch_size = BATCH_SIZE,
            drop_last=True
        )
######################



model=AttentionModel(bert, BATCH_SIZE, OUTPUT_DIM, HIDDEN_DIM, 768)
# optimizers = [optim.Adam(models[0].parameters()), optim.Adam(models[1].parameters())]

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
nll_loss = nn.NLLLoss()
log_softmax = nn.LogSoftmax()

######################
def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

######################
def categorical_accuracy(preds, y):
    count0,count1,count2 = torch.zeros(1),torch.zeros(1),torch.zeros(1)
    total0,total1,total2 = torch.FloatTensor(1),torch.FloatTensor(1),torch.FloatTensor(1)
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    predictions = max_preds.squeeze(1)
    true_correct = [0,0,0]
    for j,i in enumerate(y.cpu().numpy()):
      true_correct[y.cpu().numpy()[j]]+=1
      if i==0:
        count0+=correct[j]
        total0+=1
      elif i==1:
        count1+=correct[j]
        total1+=1
      elif i==2:
        count2+=correct[j]
      else:
        total2+=1
    metric=torch.FloatTensor([count0/true_correct[0],count1/true_correct[1],count2/true_correct[2],f1_score(y.cpu().numpy(),predictions.cpu().numpy(),average='macro')])
    return correct.sum() / torch.FloatTensor([y.shape[0]]),metric

######################
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc





######################



def train(model, train_dataloader, optimizer, criterion):

    # for epoch in range(N_EPOCHS):
    for i in range(1):
        epoch_loss = 0
        epoch_acc = 0
    
        model.train()
        print("training...")
    
        for step,batch in enumerate(train_dataloader):
            print(step)
            optimizer.zero_grad()
            # model(batch[0])
            # break
        
           

            predictions =  model(batch[0]).squeeze(1)

            loss = criterion(predictions, batch[2])
            acc,_= categorical_accuracy(predictions, batch[2])
            # acc= binary_accuracy(predictions.view(OUTPUT_DIM,BATCH_SIZE), batch[2])

            loss.backward()

            clip_gradient(model, 1e-1)

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
                # break

        return epoch_loss / len(train_dataloader), epoch_acc / len(train_dataloader)
                


######################
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_all_acc = torch.FloatTensor([0,0,0,0])
    confusion_mat = torch.zeros((3,3))
    confusion_mat_temp = torch.zeros((3,3))

    model.eval()
    print("validating/testing.....")
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch[0]).squeeze(1)
            # else:
            #   predictions = model(batch.text,batch_size=len(batch)).squeeze(1)
            
            loss = criterion(predictions, batch[2])
            acc,all_acc = categorical_accuracy(predictions, batch[2])
            # acc = binary_accuracy(predictions.view(OUTPUT_DIM,BATCH_SIZE), batch[2])

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            # epoch_all_acc += all_acc
            # confusion_mat+=confusion_mat_temp
            # break
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
######################

'''N_EPOCHS=5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
	# train(model, train_dataloader, optimizer, criterion)
    train_loss, train_acc = train(model, train_dataloader, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, validation_dataloader, criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'lstm-model.pt')


    # print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')'''
###################### 

'''testing portion'''  
df = pd.read_csv("/Users/taseefrahman/Documents/CS 695/codemix/data/Hinglish_test_labeled_conll_updated.tsv", delimiter='\t', header=None, names=['sentence','label'])

print('Number of test sentences: {:,}\n'.format(df.shape[0]))

sentences = df.sentence.values
labels = df.label.values
input_ids = []
attention_masks = []

for sent in sentences:
    encoded_dict = tokenizer.encode_plus(
                        sent,
                        add_special_tokens = True,
                        max_length = max_input_length,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        truncation = True,
                   )
      
    input_ids.append(encoded_dict['input_ids'])
    
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

test_dataset = TensorDataset(input_ids, attention_masks, labels)
prediction_dataloader = DataLoader(
            test_dataset,
            sampler = SequentialSampler(test_dataset),
            batch_size = BATCH_SIZE,
            drop_last=True
        
        )
model.load_state_dict(torch.load('lstm-model.pt'))

test_loss, test_acc = evaluate(model, prediction_dataloader, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
######################










