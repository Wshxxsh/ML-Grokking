import torch


### One step


def train_step(model, trainloader, loss_fn, optimizer):
    model.train()
    train_loss, train_acc = 0, 0

    for X, y in trainloader:
        y_pred = model(X.T)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    train_loss = train_loss / len(trainloader)
    train_acc = train_acc / len(trainloader)
    return train_loss, train_acc

def test_step(model, testloader, loss_fn):
    model.eval() 
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for X, y in testloader:
            test_pred_logits = model(X.T)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    test_loss = test_loss / len(testloader)
    test_acc = test_acc / len(testloader)
    return test_loss, test_acc


def test_add(model, device, prime=97, assoc_size=10000, no_a=0):
    x = torch.arange(0, prime).to(device)
    y = torch.arange(0, prime).to(device)
    p = torch.tensor([prime]).to(device)
    op = torch.tensor([prime+1]).to(device)
    data = torch.cartesian_prod(x, p, y, op)
    
    # comm acc
    add_res = torch.argmax(torch.softmax(model(data.T), dim=1), dim=1)
    # x + y y + x
    M = add_res.reshape(prime, prime)
    comm_acc = (M == M.T).sum().item() / prime**2

    # assoc acc
    rep = torch.randint(low=0, high=prime**3, size=(assoc_size,)).to(device)
    data = torch.cartesian_prod(rep, p, p, op).to(device)
    X = torch.fmod(rep, prime)
    Y = torch.round(torch.fmod(rep-X, prime**2) / prime)
    Z = torch.round((rep-X-Y*prime) / prime**2)
    
    data_l = data.clone(); data_l[:, 0] = X; data_l[:, 2] = Y
    data_r = data.clone(); data_r[:, 0] = Y; data_r[:, 2] = Z
    add_l = torch.argmax(torch.softmax(model(data_l.T), dim=1), dim=1)
    add_r = torch.argmax(torch.softmax(model(data_r.T), dim=1), dim=1)

    data_r[:, 0] = add_l; data_l[:, 2] = add_r
    A1 = torch.argmax(torch.softmax(model(data_l.T), dim=1), dim=1)
    A2 = torch.argmax(torch.softmax(model(data_r.T), dim=1), dim=1)
    assoc_acc = (A1 == A2).sum().item() / assoc_size

    # add acc
    data_adda = torch.cartesian_prod(x, p, torch.tensor([no_a]).to(device), op)
    add_a = torch.argmax(torch.softmax(model(data_adda.T), dim=1), dim=1)
    x_add = torch.round(x + no_a)
    adda_acc = (add_a == x_add).sum().item() / prime
    return comm_acc, assoc_acc, adda_acc