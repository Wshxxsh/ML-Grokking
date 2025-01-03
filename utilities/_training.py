from ._complement import *
from ._update import *
from ._helper import *
from sklearn.manifold import TSNE


def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, 
          scheduler = None, oriInfo=False, tsneInfo=False, add_test=False, complement=False,
          prime=97, tolmax=10, assoc_size=10000, no_a=0):
    """
    Training
    
    args:
    model : model to train
    train_dataloader (torch.utils.data.dataloader.DataLoader): train loader
    test_dataloader (torch.utils.data.dataloader.DataLoader): test loader
    optimizer (torch.optim.xx): optimizer
    loss_fn (torch.nn.modules.loss.xx): loss function
    epochs (int): maximum epochs
    scheduler (None | torch.optim.lr_scheduler.StepLR): scheduler. Default None
    oriInfo (bool): whether to save the original frequency information of output layer. Default False
    tsneInfo (bool): whether to save the original frequency information of TSNE projection of output layer. Default False
    add_test (bool): whether to test addition properties. Default False
    complement (bool): whether to test acc on C_T and V-C_T. Default False
    prime (int): prime of dataset. Default 97
    tolmax (int): the training process will stop once the testing acc attains 99% tolmax times. Default 10
    assoc_size (int): the number of datas used to test associativity. Default 10000
    no_a (int): the accuracy of 'adding no_a' will be calculated when testing addition properties. Default 0
    
    return:
    res (dict): the training information. Including:
        train_loss, train_acc, test_loss, test_acc
        thetaOri  if oriInfo is True
        thetaTSNE  if tsneInfo is True
        comm_acc, assoc_acc, add0_acc  if add_test is True
        comp, ncomp  if complement is True
    """
    results = {"train_loss": [],"train_acc": [],"test_loss": [],"test_acc": [], "thetaOri": [], 'thetaTSNE': [], 
              'comm_acc' : [], 'assoc_acc': [], 'add0_acc': [], 'comp': [], 'ncomp': []}
    tol = 0
    if complement:
        comp, comp_label = train_complement(train_dataloader.dataset)
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model=model,trainloader=train_dataloader,loss_fn=loss_fn,optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,testloader=test_dataloader,loss_fn=loss_fn)
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        if scheduler is not None:
            scheduler.step()
        if oriInfo:
            for m in model.modules():
                W = m
            W = W.weight.cpu().detach().numpy()
            Z = []
            for i in range(prime):
                Z.append(W[(i+1)%prime, :] - W[i%prime, :])
            Z = np.array(Z)
            M = Z @ Z.T
            S = [np.diag(M).mean()]
            for i in range(1, prime):
                Md = np.concatenate([np.diag(M, i), np.diag(M, i-prime)])
                S.append((Md.sum())/prime)
            theta = myDCT(S)
            results['thetaOri'].append(theta)
        if tsneInfo:
            for m in model.modules():
                W = m
            W = W.weight.cpu().detach().numpy()
            tsne = TSNE(n_components=2, perplexity=prime-1, n_iter=500)
            X_embedded = tsne.fit_transform(W)
            Rx, Ry = [], []
            for i in range(prime):
                Rx.append(X_embedded[(i+1)%prime, 0] - X_embedded[i%prime, 0])
                Ry.append(X_embedded[(i+1)%prime, 1] - X_embedded[i%prime, 1])
            Rx = np.array(Rx)
            Ry = np.array(Ry)
            theta = myDCT(Rx / np.sqrt(Rx**2 + Ry**2))
            results['thetaTSNE'].append(theta)
        if add_test:
            comm_acc, assoc_acc, add0_acc = test_add(model, 'cuda', prime=prime, assoc_size=assoc_size, no_a=no_a)
            results['comm_acc'].append(comm_acc)
            results['assoc_acc'].append(assoc_acc)
            results['add0_acc'].append(add0_acc)
        if complement and len(comp) > 0:
            comp_pred = torch.argmax(torch.softmax(model(comp.T), dim=1), dim=1)
            num_acc = (comp_pred == comp_label).sum().item()
            comp_acc = num_acc / len(comp_label)
            ncomp_acc = (round(test_acc*len(test_dataloader.dataset)) - num_acc) / (len(test_dataloader.dataset) - len(comp_label) + 1e-6)  
            results['comp'].append(comp_acc)
            results['ncomp'].append(ncomp_acc)
        if test_acc > .99:
            tol += 1
            if tolmax is not None and tol > tolmax:
                break
    results['thetaOri'] = np.array(results['thetaOri'])
    results['thetaTSNE'] = np.array(results['thetaTSNE'])
    return results