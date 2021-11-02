"""
    6. Testing on Your Own Sentence.

    You will write a python script that prompts the user for a sentence input on the command line, and prints the
    classification from each of the three models, as well as the probability that this sentence is nonsense.

    An example console output:

        Enter a sentence
        What once seemed creepy now just seems campy

        Model baseline: sense (0.419)
        Model rnn: sense (0.363)
        Model cnn: nonsense (0.992)

        Enter a sentence
"""
import readline
import torch
import spacy

import torchtext
from torchtext import data

TEXT = data.Field(sequential=True,lower=True, tokenize='spacy', include_lengths=True)
LABELS = data.Field(sequential=False, use_vocab=False)
    
train_data, val_data, test_data = data.TabularDataset.splits(path='data/', train='train.tsv',validation='validation.tsv', test='test.tsv', format='tsv',skip_header=True, fields=[('text', TEXT), ('label', LABELS)])

TEXT.build_vocab(train_data, val_data, test_data)

TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
vocab = TEXT.vocab

baseline = torch.load('model_baseline.pt')
CNN = torch.load('model_cnn.pt')
RNN = torch.load('model_rnn.pt')

def tokenizer(text):
    spacy_en = spacy.load('en')
    return [tok.text for tok in spacy_en(text)]

def subjOrObj(x):
    if x > 0.5:
        return "nonsense"
    else:
        return "sense"

def firstFourDigits(x):
    string = str(x[0].item())
    return string[0:5]

while True:
    print("Enter a sentence:")
    sentenceGiven = input()
    if sentenceGiven == "Quit":
        break
    tokens = tokenizer(sentenceGiven)
    token_ints = [vocab.stoi[tok] for tok in tokens]
    token_tensor = torch.LongTensor(token_ints).view(-1,1)
    lengths = torch.Tensor([len(token_ints)])
    baseOut = torch.sigmoid(baseline(token_tensor,lengths))
    try:
        CNNOut = torch.sigmoid(CNN(token_tensor,lengths))
        skipCNN = False
    except:
        skipCNN = True
    RNNOut = torch.sigmoid(RNN(token_tensor,lengths))
    print("Model baseline: ",subjOrObj(baseOut)," (",firstFourDigits(baseOut),")")
    if not skipCNN:
        print("Model cnn: ",subjOrObj(CNNOut)," (",firstFourDigits(CNNOut),")")
    print("Model rnn: ",subjOrObj(RNNOut)," (",firstFourDigits(RNNOut),")")


