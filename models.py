import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):

    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x, lengths=None):
        #x = [sentence length, batch size]
        embedded = self.embedding(x)

        average = embedded.mean(0) # [sentence length, batch size, embedding_dim]
        output = self.fc(average).squeeze(1)

	# Note - using the BCEWithLogitsLoss loss function
        # performs the sigmoid function *as well* as well as
        # the binary cross entropy loss computation
        # (these are combined for numerical stability)

        return output

class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(CNN, self).__init__()

        ######

        # Section 5.0 YOUR CODE HERE
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.conv1 = nn.Conv2d(1,n_filters,kernel_size=(filter_sizes[0],embedding_dim))
        self.conv2 = nn.Conv2d(1,n_filters,kernel_size=(filter_sizes[1],embedding_dim))
        self.fc2 = nn.Linear(2*n_filters,1)

    def forward(self, x, lengths=None):
        ######

        # Section 5.0 YOUR CODE HERE
        #print("size of x: ",x.size())
        embedded = self.embedding(x)
        #print("initial size of embedding:",embedded.size())
        embedded = embedded.transpose(0,1)
        #print("size of embedding after transpose:",embedded.size())
        embedded = embedded.unsqueeze(1)
        #print("size of embedding:",embedded.size())
        pool = nn.MaxPool2d(1,embedded.size()[2])
        x1 = pool(F.relu(self.conv1(embedded)))
        x2 = pool(F.relu(self.conv2(embedded)))
        x = torch.cat((x1,x2),dim=1)
        x = x.squeeze()
        result = self.fc2(x)
        #print("size of result:",result.size())
        return result
        ######

class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(RNN, self).__init__()
        pass

        ######

        # Section 6.0 YOUR CODE HERE
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc2 = nn.Linear(hidden_dim,1)
        self.GRU = nn.GRU(embedding_dim,hidden_dim)
        ######

    def forward(self, x, lengths=None):
        ######

        # Section 6.0 YOUR CODE HERE
        embedded = self.embedding(x)
        #print("embedded.size() = ",embedded.size())
        embedded = nn.utils.rnn.pack_padded_sequence(embedded,lengths,enforce_sorted=False)
        #print("embedded.data.size() = ",embedded.data.size())
        x, hidden = self.GRU(embedded)
        #print("hidden = ",hidden)
        #print("output.size() = ",output.size())
        return self.fc2(hidden.squeeze())
