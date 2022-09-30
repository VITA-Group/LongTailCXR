import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm

from copy import deepcopy

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, classification_report, confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import LabelBinarizer

def set_seed(seed):
    """Set all random seeds and settings for reproducibility (deterministic behavior)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def val_worker_init_fn(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def bal_mixup_train(model, device, loss_fxn, optimizer, data_loader, history, epoch, model_dir, classes, mixup_alpha):
    """Train PyTorch model for one epoch on NIH ChestXRay14 dataset.
    Parameters
    ----------
        model : PyTorch model
        device : PyTorch device
        loss_fxn : PyTorch loss function
        optimizer : PyTorch optimizer
        data_loader : PyTorch data loader
        history : pandas DataFrame
            Data frame containing history of training metrics
        epoch : int
            Current epoch number (1-K)
        model_dir : str
            Path to output directory where metrics, model weights, etc. will be stored
        classes : list[str]
            Ordered list of names of output classes
    Returns
    -------
        history : pandas DataFrame
            Updated history data frame with metrics from completed training epoch
    """
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch}')

    running_loss = 0.
    y_true, y_hat = [], []
    for i, batch in pbar:
        x = batch[0][0].to(device)
        y = batch[0][1].to(device)
        bal_x = batch[0][0].to(device)
        bal_y = batch[0][1].to(device)

        lam = np.random.beta(mixup_alpha, mixup_alpha)
        mixed_x = (1 - lam) * x + lam * bal_x

        out = model(mixed_x)

        loss = mixup_criterion(loss_fxn, out, y, bal_y, lam)

        for param in model.parameters():
            param.grad = None
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        y_hat.append(out.softmax(dim=1).detach().cpu().numpy())
        y_true.append(y.detach().cpu().numpy())

        pbar.set_postfix({'loss': running_loss / (i + 1)})

    # Collect true and predicted labels into flat numpy arrays
    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)

    # Compute metrics
    auc = roc_auc_score(y_true, y_hat, average='macro', multi_class='ovr')
    b_acc = balanced_accuracy_score(y_true, y_hat.argmax(axis=1))
    mcc = matthews_corrcoef(y_true, y_hat.argmax(axis=1))

    print('Balanced Accuracy:', round(b_acc, 3), '|', 'MCC:', round(mcc, 3), '|', 'AUC:', round(auc, 3))

    current_metrics = pd.DataFrame([[epoch, 'train', running_loss / (i + 1), b_acc, mcc, auc]], columns=history.columns)
    current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

    return history.append(current_metrics)

def train(model, device, loss_fxn, optimizer, data_loader, history, epoch, model_dir, classes, mixup, mixup_alpha):
    """Train PyTorch model for one epoch on NIH ChestXRay14 dataset.
    Parameters
    ----------
        model : PyTorch model
        device : PyTorch device
        loss_fxn : PyTorch loss function
        optimizer : PyTorch optimizer
        data_loader : PyTorch data loader
        history : pandas DataFrame
            Data frame containing history of training metrics
        epoch : int
            Current epoch number (1-K)
        model_dir : str
            Path to output directory where metrics, model weights, etc. will be stored
        classes : list[str]
            Ordered list of names of output classes
    Returns
    -------
        history : pandas DataFrame
            Updated history data frame with metrics from completed training epoch
    """
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch}')

    running_loss = 0.
    y_true, y_hat = [], []
    for i, (x, y) in pbar:
        x = x.to(device)
        y = y.to(device)

        if mixup:
            x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha, True)

        out = model(x)

        if mixup:
            loss = mixup_criterion(loss_fxn, out, y_a, y_b, lam)
        else:
            loss = loss_fxn(out, y)

        for param in model.parameters():
            param.grad = None
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        y_hat.append(out.softmax(dim=1).detach().cpu().numpy())
        y_true.append(y.detach().cpu().numpy())

        pbar.set_postfix({'loss': running_loss / (i + 1)})

    # Collect true and predicted labels into flat numpy arrays
    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)

    # Compute metrics
    auc = roc_auc_score(y_true, y_hat, average='macro', multi_class='ovr')
    b_acc = balanced_accuracy_score(y_true, y_hat.argmax(axis=1))
    mcc = matthews_corrcoef(y_true, y_hat.argmax(axis=1))

    print('Balanced Accuracy:', round(b_acc, 3), '|', 'MCC:', round(mcc, 3), '|', 'AUC:', round(auc, 3))

    current_metrics = pd.DataFrame([[epoch, 'train', running_loss / (i + 1), b_acc, mcc, auc]], columns=history.columns)
    current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

    return history.append(current_metrics)

def validate(model, device, loss_fxn, optimizer, data_loader, history, epoch, model_dir, early_stopping_dict, best_model_wts, classes):
    """Evaluate PyTorch model on validation set of NIH ChestXRay14 dataset.
    Parameters
    ----------
        model : PyTorch model
        device : PyTorch device
        loss_fxn : PyTorch loss function
        ls : int
            Ratio of label smoothing to apply during loss computation
        optimizer : PyTorch optimizer
        data_loader : PyTorch data loader
        history : pandas DataFrame
            Data frame containing history of training metrics
        epoch : int
            Current epoch number (1-K)
        model_dir : str
            Path to output directory where metrics, model weights, etc. will be stored
        early_stopping_dict : dict
            Dictionary of form {'epochs_no_improve': <int>, 'best_loss': <float>} for early stopping
        best_model_wts : PyTorch state_dict
            Model weights from best epoch
        classes : list[str]
            Ordered list of names of output classes
        fusion : bool
            Whether or not fusion is being performed (image + metadata inputs)
        meta_only : bool
            Whether or not to train on *only* metadata as input
    Returns
    -------
        history : pandas DataFrame
            Updated history data frame with metrics from completed training epoch
        early_stopping_dict : dict
            Updated early stopping metrics
        best_model_wts : PyTorch state_dict
            (Potentially) updated model weights (if best validation loss achieved)
    """
    model.eval()
    
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[VAL] Epoch {epoch}')

    running_loss = 0.
    y_true, y_hat = [], []
    with torch.no_grad():
        for i, (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)

            out = model(x)

            loss = loss_fxn(out, y)

            running_loss += loss.item()

            y_hat.append(out.softmax(dim=1).detach().cpu().numpy())
            y_true.append(y.detach().cpu().numpy())

            pbar.set_postfix({'loss': running_loss / (i + 1)})

    # Collect true and predicted labels into flat numpy arrays
    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)

    # Compute metrics
    auc = roc_auc_score(y_true, y_hat, average='macro', multi_class='ovr')
    b_acc = balanced_accuracy_score(y_true, y_hat.argmax(axis=1))
    mcc = matthews_corrcoef(y_true, y_hat.argmax(axis=1))

    print('[VAL] Balanced Accuracy:', round(b_acc, 3), '|', 'MCC:', round(mcc, 3), '|', 'AUC:', round(auc, 3))

    current_metrics = pd.DataFrame([[epoch, 'val', running_loss / (i + 1), b_acc, mcc, auc]], columns=history.columns)
    current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

    # Early stopping: save model weights only when val (balanced) accuracy has improved
    if b_acc > early_stopping_dict['best_acc']:
        print(f'--- EARLY STOPPING: Accuracy has improved from {round(early_stopping_dict["best_acc"], 3)} to {round(b_acc, 3)}! Saving weights. ---')
        early_stopping_dict['epochs_no_improve'] = 0
        early_stopping_dict['best_acc'] = b_acc
        best_model_wts = deepcopy(model.state_dict())
        torch.save({'weights': best_model_wts, 'optimizer': optimizer.state_dict()}, os.path.join(model_dir, f'chkpt_epoch-{epoch}.pt'))
    else:
        print(f'--- EARLY STOPPING: Accuracy has not improved from {round(early_stopping_dict["best_acc"], 3)} ---')
        early_stopping_dict['epochs_no_improve'] += 1

    return history.append(current_metrics), early_stopping_dict, best_model_wts


def evaluate(model, device, loss_fxn, dataset, split, batch_size, history, model_dir, weights):
    """Evaluate PyTorch model on test set of NIH ChestXRay14 dataset. Saves training history csv, summary text file, training curves, etc.
    Parameters
    ----------
        model : PyTorch model
        device : PyTorch device
        loss_fxn : PyTorch loss function
        ls : int
            Ratio of label smoothing to apply during loss computation
        batch_size : int
        history : pandas DataFrame
            Data frame containing history of training metrics
        model_dir : str
            Path to output directory where metrics, model weights, etc. will be stored
        weights : PyTorch state_dict
            Model weights from best epoch
        n_TTA : int
            Number of augmented copies to use for test-time augmentation (0-K)
        fusion : bool
            Whether or not fusion is being performed (image + metadata inputs)
        meta_only : bool
            Whether or not to train on *only* metadata as input
    """
    model.load_state_dict(weights)  # load best weights
    model.eval()

    ## INFERENCE
    data_loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8 if split == 'test' else 2, pin_memory=True, worker_init_fn=val_worker_init_fn)

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[{split.upper()}] EVALUATION')

    running_loss = 0.
    y_true, y_hat = [], []
    with torch.no_grad():
        for i, (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)

            out = model(x)

            loss = loss_fxn(out, y)

            running_loss += loss.item()

            y_hat.append(out.softmax(dim=1).detach().cpu().numpy())
            y_true.append(y.detach().cpu().numpy())

            pbar.set_postfix({'loss': running_loss / (i + 1)})

    # Collect true and predicted labels into flat numpy arrays
    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)

    # Compute metrics
    auc = roc_auc_score(y_true, y_hat, average='macro', multi_class='ovr')
    b_acc = balanced_accuracy_score(y_true, y_hat.argmax(axis=1))
    conf_mat = confusion_matrix(y_true, y_hat.argmax(axis=1))
    accuracies = conf_mat.diagonal() / conf_mat.sum(axis=1)
    mcc = matthews_corrcoef(y_true, y_hat.argmax(axis=1))
    cls_report = classification_report(y_true, y_hat.argmax(axis=1), target_names=dataset.CLASSES, digits=3)

    print(f'[{split.upper()}] Balanced Accuracy: {round(b_acc, 3)} | MCC: {round(mcc, 3)} | AUC: {round(auc, 3)}')

    # Collect and save true and predicted disease labels for test set
    pred_df = pd.DataFrame(y_hat, columns=dataset.CLASSES)
    true_df = pd.DataFrame(LabelBinarizer().fit(range(len(dataset.CLASSES))).transform(y_true), columns=dataset.CLASSES)

    pred_df.to_csv(os.path.join(model_dir, f'{split}_pred.csv'), index=False)
    true_df.to_csv(os.path.join(model_dir, f'{split}_true.csv'), index=False)

    # Plot confusion matrix
    fig, ax = plot_confusion_matrix(conf_mat, figsize=(24, 24), colorbar=True, show_absolute=True, show_normed=True, class_names=dataset.CLASSES)
    fig.savefig(os.path.join(model_dir, f'{split}_cm.png'), dpi=300, bbox_inches='tight')

    # Plot loss curves
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'loss'], label='train')
    ax.plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'loss'], label='val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.savefig(os.path.join(model_dir, 'loss.png'), dpi=300, bbox_inches='tight')

    # Plot accuracy curves
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'balanced_acc'], label='train')
    ax.plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'balanced_acc'], label='val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Balanced Accuracy')
    ax.legend()
    fig.savefig(os.path.join(model_dir, 'balanced_acc.png'), dpi=300, bbox_inches='tight')
    
    # Plot AUROC learning curves
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'auroc'], label='train')
    ax.plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'auroc'], label='val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUROC')
    ax.legend()
    fig.savefig(os.path.join(model_dir, 'auc.png'), dpi=300, bbox_inches='tight')
    
    # Create summary text file describing final performance
    summary = f'Balanced Accuracy: {round(b_acc, 3)}\n'
    summary += f'Matthews Correlation Coefficient: {round(mcc, 3)}\n'
    summary += f'Mean AUC: {round(auc, 3)}\n\n'

    summary += 'Class:| Accuracy\n'
    for i, c in enumerate(dataset.CLASSES):
        summary += f'{c}:| {round(accuracies[i], 3)}\n'
    summary += '\n'
    
    summary += cls_report

    f = open(os.path.join(model_dir, f'{split}_summary.txt'), 'w')
    f.write(summary)
    f.close()