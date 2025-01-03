import torch


### C_T


def train_complement(train_dataset):
    """
    compute C_T
    
    args:
    train_dataset (torch.utils.data.dataset.Subset): dataset of V
    
    return:
    (X, y): X is the data of C_T, y is the label of C_T
    """
    comp, comp_label = dict(), dict()
    for i in train_dataset:
        X, label = i[0], i[1]
        device = X.device
        a, b, p = X[2], X[0], X[1]
        if a.item() != b.item():
            comp[(a.item(), b.item())] = X
            comp_label[(a.item(), b.item())] = label
            if ((a.item(), b.item()) in comp.keys()) and ((b.item(), a.item()) in comp.keys()):
                del comp[(a.item(), b.item())]; del comp[(b.item(), a.item())]
                del comp_label[(a.item(), b.item())]; del comp_label[(b.item(), a.item())]
    cmp, cmp_label = [], []
    for (a, b) in comp.keys():
        cmp.append([a, p, b, p+1])
        cmp_label.append(comp_label[(a, b)])
    return torch.tensor(cmp).to(device), torch.tensor(cmp_label).to(device)