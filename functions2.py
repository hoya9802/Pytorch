import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def Train(model, train_DL, val_DL, criterion, optimizer, epoch, BATCH_SIZE, TRAIN_RATIO, save_model_path, save_history_path):
    loss_history = {'train':[], 'val':[]}
    acc_history = {'train':[], 'val':[]}
    best_loss = 9999
    
    for it in range(epoch):
        epoch_start = time.time()
        model.train()
        train_loss, train_acc, _ = loss_epoch(model, train_DL, criterion, optimizer)
        loss_history['train'] += [train_loss]
        acc_history['train'] += [train_acc]

        model.eval()
        with torch.no_grad():
            val_loss, val_acc, _ = loss_epoch(model, val_DL, criterion, optimizer)
            loss_history['val'] += [val_loss]
            acc_history['val'] += [val_acc]

            if best_loss > val_loss:
                hest_loss = val_loss
                torch.save({'model':model,
                            'epo':it,
                            'optimizer':optimizer,
                            'scheduler':scheduler}, save_model_path)
        # print loss
        print(f"train loss: {round(train_loss,5)}, "
              f"val loss: {round(val_loss,5)} \n"
              f"train acc: {round(train_acc,1)} %, "
              f"val acc: {round(val_acc,1)} %, time: {round(time.time()-epoch_start()} s")
        print('='*30)

    torch.save({'loss_history': loss_history,
                'acc_history': acc_history,
                'EPOCH': EPOCH,
                'BATCH_SIZE': BATCH_SIZE,
                'TRAIN_RATIO': TRAIN_RATIO}, save_history_path)

    return loss_history

def Test(model, test_DL, criterion):
    model.eval()
    with torch.no_grad():
        test_loss, test_acc, rcorrect = loss_epoch(model, test_DL, criterion)
    print()
    print(f'Test Loss: {round(test_loss, 5)}')
    print(f'Test accuracy: {rcorrect}/{len(test_DL.dataset)} ({round(test_acc, 1)} %)')
    return round(test_acc, 1)

def loss_epoch(model, DL, criterion, optimizer = None):
    N = len(DL.dataset)
    rloss = 0; rcorrect = 0
    for x_batch, y_batch in tqdm(DL):
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        y_hat = model(x_batch)
        loss = criterion(y_hat, y_batch)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_b = loss.item() * x_batch.shape[0]
        rloss += loss_b
        
        pred = y_hat.argmax(dim=1)
        correct_b = torch.sum(pred == y_batch).item()
        rcorrect += correct_b
        
    loss_e = rloss/N
    accuracy_e = rcorrect/N * 100

    return loss_e, accuracy_e, rcorrect

def Test_plot(model, test_DL):
    model.eval()
    with torch.no_grad():
        x_batch, y_batch = next(iter(test_DL))
        x_batch = x_batch.to(DEVICE)
        y_hat = model(x_batch)
        pred = y_hat.argmax(dim=1)
    x_batch = x_batch.to('cpu')

    plt.figure(figsize=(8,4))
    for idx in range(6):
        plt.subplot(2,3, idx+1, xticks=[], yticks=[])
        plt.imshow(x_batch[idx].permute(1,2,0).squeeze(), cmap='gray')
        pred_class = test_DL.dataset.classes[pred[idx]]
        true_class = test_DL.dataset.classes[y_batch[idx]]
        plt.title(f'({pred_class}) ({true_class})', color='g' if pred_class==true_class else 'r')
 
def count_params(model):
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num








