import torch
import torch.nn as nn
import torch.nn.functional as F

def accuracy(predictions,labels):
    predictions = predictions.detach()
    total_corr = 0
    index = 0
