import torch
import torch.optim as optim
import torch.nn as nn

import torchtext
from torchtext import data
import spacy

import argparse
import os

import matplotlib.pyplot as plt

from models import *

torch.manual_seed(0)

def evaluate(model,val_loader):
    total_corr = 0
    val_loss = 0.0
    val_acc = 0.0
    truePos = 0.0
    trueNeg = 0.0
    falsePos = 0.0
    falseNeg = 0.0
    loss_fnc = nn.BCEWithLogitsLoss()
    for i, batch in enumerate(val_loader,1):
        acc_batch = 0.0
        truePos_batch = 0.0
        trueNeg_batch = 0.0
        falsePos_batch = 0.0
        falseNeg_batch = 0.0
        inputs, lengths = batch.text
        labels = batch.label
        outputs = model(inputs,lengths=lengths)
        loss = loss_fnc(outputs.squeeze(),labels.float())
        outputs = torch.sigmoid(outputs)
        for j in range(0,len(outputs)):
            if outputs[j] > 0.5 and labels[j] == 1:
                acc_batch += 1.0
                truePos_batch += 1.0
            if outputs[j] <= 0.5 and labels[j] == 0:
                acc_batch += 1.0
                trueNeg_batch += 1.0
            if outputs[j] > 0.5 and labels[j] == 0:
                falsePos_batch += 1.0
            if outputs[j] <= 0.5 and labels[j] == 1:
                falseNeg_batch += 1.0
        acc_batch = acc_batch/len(outputs)
        val_loss += loss.item()
        val_acc += acc_batch
        truePos_batch /= len(outputs)
        truePos += truePos_batch
        trueNeg_batch /= len(outputs)
        trueNeg += trueNeg_batch
        falsePos_batch /= len(outputs)
        falsePos += falsePos_batch
        falseNeg_batch /= len(outputs)
        falseNeg += falseNeg_batch
    val_acc = val_acc/i
    val_rates = [truePos/i,trueNeg/i,falsePos/i,falseNeg/i]
    return val_loss, val_acc, val_rates

def main(args):
    ######
    # 3.2 Processing of the data
    # the code below assumes you have processed and split the data into
    # the three files, train.tsv, validation.tsv and test.tsv
    # and those files reside in the folder named "data".
    ######

    # 3.2.1
    TEXT = data.Field(sequential=True,lower=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)
    
    train_data, val_data, test_data = data.TabularDataset.splits(
            path='data/', train='train.tsv',
            validation='validation.tsv', test='test.tsv', format='tsv',
            skip_header=True, fields=[('text', TEXT), ('label', LABELS)])

    # 3.2.3
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
      (train_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size, args.batch_size),
    sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)

    # 3.2.4
    TEXT.build_vocab(train_data, val_data, test_data)

    # 4.1
    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=args.emb_dim))
    vocab = TEXT.vocab

    print("Shape of Vocab:",TEXT.vocab.vectors.shape)
    
    #4.3/4.4
    model = Baseline(args.emb_dim,vocab)
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    loss_fnc = nn.BCEWithLogitsLoss()

    epochs = []
    losses = []
    accs = []
    losses_val = []
    accs_val = []
    losses_test = []
    accs_test = []

    # 5 Training and Evaluation
    model = CNN(args.emb_dim,vocab,args.num_filt,[1,6])
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    loss_fnc = nn.BCEWithLogitsLoss()

    epochs = []
    losses = []
    accs = []
    losses_val = []
    accs_val = []
    
    rates = []
    rates_val = []

    optimizer = optim.Adam(model.parameters(),lr=args.lr)

    for epoch in range(0,args.epochs):
        loss = 0.0
        acc = 0.0
        truePos = 0.0
        trueNeg = 0.0
        falsePos = 0.0
        falseNeg = 0.0
        for j,batch in enumerate(train_iter,1):
            acc_batch = 0.0
            truePos_batch = 0.0
            trueNeg_batch = 0.0
            falsePos_batch = 0.0
            falseNeg_batch = 0.0
            inputs,lengths = batch.text
            labels = batch.label
            optimizer.zero_grad()
            outputs = model.forward(inputs,lengths=lengths)
            batch_loss = loss_fnc(outputs.squeeze(),labels.float())
            batch_loss.backward()
            loss += batch_loss
            optimizer.step()
            outputs = torch.sigmoid(outputs)
            for k in range(0,len(outputs)):
                if labels[k] == 1 and outputs[k] > 0.5:
                    acc_batch += 1.0
                    truePos_batch += 1.0
                if labels[k] == 0 and outputs[k] <= 0.5:
                    acc_batch += 1.0
                    trueNeg_batch += 1.0
                if labels[k] == 1 and outputs[k] <= 0.5:
                    falseNeg_batch += 1.0
                if labels[k] == 0 and outputs[k] > 0.5:
                    falsePos_batch += 1.0
            acc_batch = acc_batch/args.batch_size
            acc += acc_batch
            truePos_batch /= args.batch_size
            truePos += truePos_batch
            trueNeg_batch /= args.batch_size
            trueNeg += trueNeg_batch
            falsePos_batch /= args.batch_size
            falsePos += falsePos_batch
            falseNeg_batch /= args.batch_size
            falseNeg += falseNeg_batch
        losses.append(loss)
        accs.append(acc/j)
        rate = [truePos/j,trueNeg/j,falsePos/j,falseNeg/j]
        print("RNN loss for epoch",epoch,"=",loss.item())
        print("RNN accuracy for epoch",epoch,"=",acc/j)
        val_loss, val_acc, val_rate = evaluate(model,val_iter)
        losses_val.append(val_loss)
        accs_val.append(val_acc)
        epochs.append(epoch)
        if acc/j - val_acc > args.threshold/100.0:
            break

    plt.subplot(2,1,1)
    plt.plot(epochs,losses,label="Training")
    plt.plot(epochs,losses_val,label="Validation")
    plt.title('Loss vs. epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(epochs,accs,label="Training")
    plt.plot(epochs,accs_val,label="Validation")
    plt.title('Accuracy vs. epoch')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('cnn_results.png')
    plt.show()

    test_loss, test_acc, test_rate = evaluate(model,test_iter)

    print("final training loss for CNN = ",loss)
    print("final training acc for CNN = ",acc/j)
    print("final validation loss for CNN = ",val_loss)
    print("final validation accuracy for CNN = ",val_acc)
    print("final test loss for CNN = ",test_loss)
    print("final test accuracy for CNN = ",test_acc)
    
    print("final true sense rate for Baseline =",rate[0]/(rate[0]+rate[3]))
    print("final true nonsense rate for Baseline = ",rate[1]/(rate[1]+rate[2]))
    
    print("final true validation sense rate for Baseline =",val_rate[0]/(val_rate[0]+val_rate[3]))
    print("final true validation nonsense rate for Baseline = ",val_rate[1]/(val_rate[1]+val_rate[2]))
    
    print("final true test sense rate for Baseline =",test_rate[0]/(test_rate[0]+test_rate[3]))
    print("final true test nonsense rate for Baseline = ",test_rate[1]/(test_rate[1]+test_rate[2]))

    torch.save(model,'model_cnn.pt')

    # #6 Training and Evaluation
#    model = RNN(args.emb_dim,vocab,args.rnn_hidden_dim)
#    optimizer = optim.Adam(model.parameters(),lr=10*args.lr)
#    loss_fnc = nn.BCEWithLogitsLoss()
#
#    epochs = []
#    losses = []
#    accs = []
#    losses_val = []
#    accs_val = []
#
#    optimizer = optim.Adam(model.parameters(),lr=args.lr)
#
#    for epoch in range(0,args.epochs):
#        loss = 0.0
#        acc = 0.0
#        for j,batch in enumerate(train_iter,1):
#            print("batch # =",j)
#            acc_batch = 0.0
#            inputs,lengths = batch.text
#            labels = batch.label
#            optimizer.zero_grad()
#            print('inputs[-1] = ',inputs[-1])
#            print('lengths = ',lengths)
#            outputs = model.forward(inputs,lengths=lengths)
#            batch_loss = loss_fnc(outputs.squeeze(),labels.float())
#            batch_loss.backward()
#            loss += batch_loss
#            optimizer.step()
#            outputs = torch.sigmoid(outputs)
#            for k in range(0,len(outputs)):
#                if labels[k] == 1 and outputs[k] > 0.5:
#                    acc_batch += 1.0
#                if labels[k] == 0 and outputs[k] <= 0.5:
#                    acc_batch += 1.0
#            acc_batch = acc_batch/args.batch_size
#            acc += acc_batch
#        losses.append(loss)
#        accs.append(acc/j)
#        val_loss, val_acc = evaluate(model,val_iter)
#        losses_val.append(val_loss)
#        accs_val.append(val_acc)
#        epochs.append(epoch)
#
#    plt.subplot(2,1,1)
#    plt.plot(epochs,losses,label="Training")
#    plt.plot(epochs,losses_val,label="Validation")
#    plt.title('Loss vs. epoch')
#    plt.xlabel('epoch')
#    plt.ylabel('loss')
#    plt.legend()
#
#    plt.subplot(2,1,2)
#    plt.plot(epochs,accs,label="Training")
#    plt.plot(epochs,accs_val,label="Validation")
#    plt.title('Accuracy vs. epoch')
#    plt.xlabel('epoch')
#    plt.ylabel('accuracy')
#    plt.legend()
#    plt.show()
#
#    test_loss, test_acc = evaluate(model,test_iter)
#
#    print("final training loss for RNN = ",loss)
#    print("final training acc for RNN = ",acc/j)
#    print("final validation loss for RNN = ",val_loss)
#    print("final validation accuracy for RNN = ",val_acc)
#    print("final test loss for RNN = ",test_loss)
#    print("final test accuracy for RNN = ",test_acc)
#
#    torch.save(model,'model_rnn.pt')
    ######


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_size', type=int, default=64)
    parser.add_argument('lr', type=float, default=0.001)
    parser.add_argument('epochs', type=int, default=25)
    parser.add_argument('model', type=str, default='baseline',
                        help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('emb_dim', type=int, default=100)
    parser.add_argument('rnn_hidden_dim', type=int, default=100)
    parser.add_argument('num_filt', type=int, default=50)
    parser.add_argument('threshold',type=int,default=10)
    
    args = parser.parse_args(['64','0.001','30','cnn','100','100','50','6'])
    main(args)
