import readline
import torch
import spacy
import matplotlib.pyplot as plt

import torchtext
from torchtext import data

from time import time

TEXT = data.Field(sequential=True,lower=True, tokenize='spacy', include_lengths=True)
LABELS = data.Field(sequential=False, use_vocab=False)
    
train_data, val_data, test_data = data.TabularDataset.splits(path='data/', train='train.tsv',validation='validation.tsv', test='test.tsv', format='tsv',skip_header=True, fields=[('text', TEXT), ('label', LABELS)])

TEXT.build_vocab(train_data, val_data, test_data)

TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
vocab = TEXT.vocab
train_iter, val_iter, test_iter = data.BucketIterator.splits(      (train_data, val_data,test_data), batch_sizes=(64, 64, 64, 64),sort_key=lambda x: len(x.text), device=None,sort_within_batch=True, repeat=False)

baseline = torch.load('model_baseline.pt')
CNN = torch.load('model_cnn.pt')
RNN = torch.load('model_rnn.pt')

def evaluate(model,val_loader,thresh):
    truePos = 0.0
    trueNeg = 0.0
    falsePos = 0.0
    falseNeg = 0.0
    for j,batch in enumerate(val_loader,1):
        truePos_batch = 0.0
        trueNeg_batch = 0.0
        falsePos_batch = 0.0
        falseNeg_batch = 0.0
        inputs,lengths = batch.text
        labels = batch.label
        outputs = model.forward(inputs,lengths=lengths)
        outputs = torch.sigmoid(outputs)
        for k in range(0,len(outputs)):
            if labels[k] == 1 and outputs[k] > thresh:
                truePos_batch += 1.0
            if labels[k] == 0 and outputs[k] <= thresh:
                trueNeg_batch += 1.0
            if labels[k] == 1 and outputs[k] <= thresh:
                falseNeg_batch += 1.0
            if labels[k] == 0 and outputs[k] > thresh:
                falsePos_batch += 1.0
        truePos_batch /= 64
        truePos += truePos_batch
        trueNeg_batch /= 64
        trueNeg += trueNeg_batch
        falsePos_batch /= 64
        falsePos += falsePos_batch
        falseNeg_batch /= 64
        falseNeg += falseNeg_batch
    truePos /= j
    falseNeg /= j
    trueNeg /= j
    falsePos /= j
    truePos /= (truePos+falseNeg)
    falsePos /= (falsePos+trueNeg)
    return truePos,falsePos

def test(model,thresh):
    ######
    # 3.2 Processing of the data
    # the code below assumes you have processed and split the data into
    # the three files, train.tsv, validation.tsv and test.tsv
    # and those files reside in the folder named "data".
    #####
    
    #4.3/4.4
    trainTruePos, trainFalsePos = evaluate(model,train_iter,thresh)
    valTruePos,valFalsePos = evaluate(model,train_iter,thresh)
    testTruePos,testFalsePos = evaluate(model,test_iter,thresh)
    truePos = 0.64*trainTruePos + 0.16*valTruePos + 0.2*testTruePos
    falsePos = 0.64*trainFalsePos + 0.16*valFalsePos + 0.2*testFalsePos
    return truePos,falsePos

BTP = []
BFP = []
CTP = []
CFP = []
RTP = []
RFP = []

for thresh in range(0,25):
    start = time()
    print("threshold of: ",thresh/25.0)
    btp, bfp = test(baseline,thresh/25.0)
    print("baseline sensitivity: ",btp)
    print("baseline specificity: ",1-bfp)
    BTP.append(btp)
    BFP.append(bfp)
    ctp, cfp = test(CNN,thresh/25.0)
    print("cnn sensitivity: ",ctp)
    print("cnn specificity: ",1-cfp)
    CTP.append(ctp)
    CFP.append(cfp)
    rtp, rfp = test(RNN,thresh/25.0)
    print("rnn sensitivity: ",rtp)
    print("rnn specificity: ",1-rfp)
    RTP.append(rtp)
    RFP.append(rfp)
    print("time taken: ",time()-start)

plt.plot(BFP,BTP)
plt.xlabel("False Positive rate")
plt.ylabel("True Positive rate")
plt.title("Baseline")
plt.savefig("baseline_ROC.png")
plt.show()

plt.plot(CFP,CTP)
plt.xlabel("False Positive rate")
plt.ylabel("True Positive rate")
plt.title("CNN")
plt.savefig("CNN_ROC.png")
plt.show()

plt.plot(RFP,RTP)
plt.xlabel("False Positive rate")
plt.ylabel("True Positive rate")
plt.title("RNN")
plt.savefig("RNN_ROC.png")
plt.show()
