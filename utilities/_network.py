from ._complement import *
import torch.nn as nn
import torch.nn.functional as F
import math


### Dataloader


def CustomDataLoader(training_fraction, batch_size, device, prime=97, no_a=None, sym=False):
    """
    generate train loader and test loader
    
    args:
    training_fraction (float): training fraction
    batch_size (int): batch size
    device (device): device
    prime (int): prime. Defalut 97
    no_a (None | int): whether to eliminate no_a in the train loader. Default None
    sym (bool): whether to generate symmetric training dataset
    
    return:
    train_loader, val_loader
    """
    if no_a is None:
        if not sym:
            x = torch.arange(0, prime)
            y = torch.arange(0, prime)
            data = torch.cartesian_prod(x, torch.tensor([prime]), y, torch.tensor([prime+1])).to(device)
            f = lambda x: torch.fmod(x[0] + x[2], prime)
            label = f(data.T).to(device)
            dataset = torch.utils.data.TensorDataset(data, label)
            train_num = int(training_fraction * len(dataset))
            val_num = len(dataset) - train_num
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_num, val_num])
        else:
            x = torch.arange(0, prime)
            y = torch.arange(0, prime)
            data = []
            for i in range(prime):
                data.append(torch.cartesian_prod(torch.tensor([i]), torch.tensor([prime]), torch.arange(i, prime), torch.tensor([prime+1])).to(device))
            datar = torch.vstack(data)
            f = lambda x: torch.fmod(x[0] + x[2], prime)
            labelr = f(datar.T).to(device)
            datasetr = torch.utils.data.TensorDataset(datar, labelr)
            train_num = int(training_fraction * len(datasetr))
            val_num = len(datasetr) - train_num
            train_datasetr, val_datasetr = torch.utils.data.random_split(datasetr, [train_num, val_num])
            train_datasetl = torch.utils.data.TensorDataset(*train_complement(train_datasetr))
            val_datasetl = torch.utils.data.TensorDataset(*train_complement(val_datasetr))
            train_dataset = train_datasetl + train_datasetr
            val_dataset = val_datasetl + val_datasetr
    else:
        x = torch.tensor([i for i in range(0, prime) if i != no_a])
        y = torch.tensor([i for i in range(0, prime) if i != no_a])
        data_n0 = torch.cartesian_prod(x, torch.tensor([prime]), y, torch.tensor([prime+1])).to(device)
        lst = [[no_a, prime, no_a, prime+1]]
        for i in range(1, prime):
            lst.append([no_a, prime, i, prime+1]); lst.append([i, prime, no_a, prime+1])
        data_0 = torch.tensor(lst).to(device)
        f = lambda x: torch.fmod(x[0] + x[2], prime)
        label_n0 = f(data_n0.T).to(device)
        label_0 = f(data_0.T).to(device)
        dataset_n0 = torch.utils.data.TensorDataset(data_n0, label_n0)
        dataset_0 = torch.utils.data.TensorDataset(data_0, label_0)
        train_num = int(training_fraction * (len(dataset_n0)+len(dataset_0)))
        val_num = len(dataset_n0) - train_num
        train_dataset, val_dataset = torch.utils.data.random_split(dataset_n0, [train_num, val_num])
        val_dataset = val_dataset + dataset_0
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


def KDataLoader(training_fraction, batch_size, device, prime=97, k=2):
    """
    generate train loader and test loader for k summation
    
    args:
    training_fraction (float): training fraction
    batch_size (int): batch size
    device (device): device
    prime (int): prime. Defalut 97
    k (int): k-sum. Default 2
    
    return:
    train_loader, val_loader
    """
    data = torch.cartesian_prod(*([torch.arange(0, prime)]*k), torch.tensor([prime+1])).to(device)
    f = lambda x: torch.fmod(x.sum(dim=0)-1, prime)
    label = f(data.T).to(device)
    dataset = torch.utils.data.TensorDataset(data, label)

    train_num = int(training_fraction * len(dataset))
    val_num = len(dataset) - train_num
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_num, val_num])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


### Transformer


class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.attention1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.ReLU(),
                                nn.Linear(d_ff, d_model)
                                )
        self.layernorm2 = nn.LayerNorm(d_model)
    
    def generate_square_subsequent_mask(self, sz):
        return torch.triu(
            torch.full((sz, sz), float("-inf")),
            diagonal=1,
        )
    
    def forward(self, X):
        msk = self.generate_square_subsequent_mask(X.shape[0]).to(X.device)
        X1, _ = self.attention1(X, X, X, attn_mask=msk)
        X1 = self.layernorm1(X + X1)
        X2 = self.layernorm2(X1 + self.ffn(X1))
        return X2
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)
        self.max_len = max_len

    def forward(self, x):
        x = x + self.pe[:self.max_len-1, :x.size(1), :].to(x.device)
        return self.dropout(x)


class Decoder(nn.Module):
    def __init__(self, d_model, nhead, d_ff, ntoken, dropout=0.1, num_layers=2, max_len=5):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(ntoken+2, d_model)
            # S B -> S B D
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
            # S B D -> S B D
        self.blks = nn.Sequential(
            *([DecoderBlock(d_model, nhead, d_ff, dropout=dropout)]*num_layers),
            nn.Linear(d_model, ntoken)
        )
    
    def forward(self, X):
        Xe = self.pos_encoder(self.embedding(X))
        return self.blks(Xe)[-1, :, :]
    
    
### MLP


class ResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(ResidualBlock, self).__init__()
        self.layernorm = nn.LayerNorm(in_features)
        self.blks = nn.Sequential(nn.Linear(in_features, hidden_features),
                                  #nn.LayerNorm(hidden_features),
                                  nn.ReLU(),
                                  nn.Linear(hidden_features, in_features),
                                 )

    def forward(self, X):
        return self.layernorm(F.relu(X + self.blks(X)))
    
    
class ResMLP(nn.Module):
    def __init__(self, d_model, hidden, ntoken, layers):
        super(ResMLP, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(ntoken+2, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.blks = nn.Sequential()
        for i in range(layers):
            self.blks.add_module('res'+str(i), ResidualBlock(d_model*4, hidden*4))
        self.blks.add_module('fnn', nn.Linear(d_model*4, ntoken))
    def forward(self, X):
        Xe = self.pos_encoder(self.embedding(X))
        Xe = Xe.transpose(0, 1).reshape(-1, self.d_model*4)
        return self.blks(Xe)


### LSTM


class LSTM(nn.Module):
    
    def __init__(self, d_model, hidden, ntoken, layers, dropout=0.1):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(ntoken+2, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.lstm = nn.LSTM(d_model, hidden, num_layers=layers, dropout=dropout)
        self.fc = nn.Linear(hidden, ntoken)
        

    def forward(self, X):
        Xe = self.pos_encoder(self.embedding(X))
        out, _ = self.lstm(Xe)
        out = self.fc(out)
        return out[-1, :, :]

    def init_hidden(self):
        return (torch.zeros(2, self.hidden_size), torch.zeros(2, self.hidden_size))