from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast,DistilBertForSequenceClassification,AdamW,Trainer,TrainingArguments
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from time import time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def acc(pred,labels):
    accuracy = 0.0
    pred = pred.detach()
    for i in range(0,len(labels)):
        p_val,p_clas = torch.max(F.softmax(pred[i]),0)
        v_val,v_clas = torch.max(labels[i],0)
        if p_clas.item() == v_clas.item():
            accuracy += 1.0
    accuracy = accuracy/len(labels)
    return accuracy
            

texts = []
labels = []
for cat in ['sense','nonsense']:
    file = cat + '.txt'
    f = open(file,'r')
    lines = f.readlines()
    for line in lines:
        texts.append(line)
        labels.append(0 if cat == 'sense' else 1)
    f.close()

train_texts, test_texts, train_labels,test_labels = train_test_split(texts,labels,test_size=.2)
train_texts, val_texts, train_labels,val_labels = train_test_split(train_texts, train_labels,test_size=.4)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

class Dataset(torch.utils.data.Dataset):
    def __init__(self,encodings,labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)
test_dataset = Dataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=True)

distil = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
for param in distil.base_model.parameters():
    param.requires_grad = False

distil.to(device)
distilOptim = AdamW(distil.parameters(),lr=5e-5)

epochs = []

losses = []
accs = []
valLosses = []
valAccs = []
testLosses = []
testAccs = []

for epoch in range(20):
    epochs.append(epoch)
    distil.train()
    num_batches = 0.0
    runningDistilLoss = 0.0
    runningDistilAcc = 0.0
    runningDistilValLoss = 0.0
    runningDistilValAcc = 0.0
    runningDistilTestLoss = 0.0
    runningDistilTestAcc = 0.0

    epStart = time()
    for batch in train_loader:
        start = time()
        num_batches += 1.0
        distilOptim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        distilOutputs = distil(input_ids,attention_mask=attention_mask,labels=labels)
        distilLoss = distilOutputs[0]
        runningDistilLoss += distilLoss.detach()
        distilAcc = acc(distilOutputs[1],labels)
        runningDistilAcc += distilAcc
        print("epoch: ",epoch+1,"batch: ",num_batches,"loss: ",distilLoss.item(),"accuracy: ",distilAcc)
        distilLoss.backward()
        distilOptim.step()
        print("batch time: ",time()-start)

    print("total number of batches: ",num_batches)
    print("total time: ",time()-epStart)
    print(doodle)
    runningDistilLoss /= num_batches
    runningDistilAcc /= num_batches

    losses.append(runningDistilLoss)
    accs.append(runningDistilAcc)
    
    print("epoch: ",epoch+1)
    print("overall distilbert loss: ",runningDistilLoss)
    print("overall distilbert accuracy: ",runningDistilAcc)
    
    distil.eval()
    num_batches = 0.0
    for batch in val_loader:
        num_batches += 1.0
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        distilOutputs = distil(input_ids,attention_mask=attention_mask,labels=labels)
        distilLoss = distilOutputs[0]
        runningDistilValLoss += distilLoss.detach()
        distilAcc = acc(distilOutputs[1],labels)
        runningDistilValAcc += distilAcc.detach()

    runningDistilValLoss /= num_batches
    runningDistilValAcc /= num_batches

    valLosses.append(runningDistilValLoss)
    valAccs.append(runningDistilValAcc)
    
    print("epoch: ",epoch+1)
    print("overall distilbert validation loss: ",runningDistilValLoss.item())
    print("overall distilbert validation accuracy: ",runningDistilValAcc)
    
    for batch in test_loader:
        num_batches += 1.0
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        distilOutputs = distil(input_ids,attention_mask=attention_mask,labels=labels)
        distilLoss = distilOutputs[0]
        runningDistilTestLoss += distilLoss.detach()
        distilAcc = acc(distilOutputs[1],labels)
        runningDistilTestAcc += distilAcc.detach()
    
    runningDistilTestLoss /= num_batches
    runningDistilTestAcc /= num_batches

    testLosses.append(runningDistilTestLoss)
    testAccs.append(runningDistilTestAcc)

print("final distilbert test loss: ",runningDistilTestLoss.item())
print("final distilbert test accuracy: ",runningDistilTestAcc)

plt.subplot(2,1,1)
plt.plot(epochs,losses,label="Training")
plt.plot(epochs,valLosses,label="Validation")
plt.plot(epochs,testLosses,label="Test")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

plt.subplot(2,1,2)
plt.plot(epochs,accs,label="Training")
plt.plot(epochs,valAccs,label="Validation")
plt.plot(epochs,testAccs,label="Test")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()
'''
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)
trainer = Trainer(model=distil,args=training_args,train_dataset=train_dataset,eval_dataset=val_dataset)
trainer.train()
'''
