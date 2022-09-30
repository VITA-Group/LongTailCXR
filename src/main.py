import os
import shutil

import argparse
import numpy as np
import pandas as pd
import torch
import torchvision

from sklearn.utils import class_weight

from datasets import *
from utils import *
from losses import *

def main(args):
    # Set model/output directory name
    MODEL_NAME = args.dataset
    MODEL_NAME += f'_{args.model_name}'
    MODEL_NAME += f'_rand' if args.rand_init else ''
    MODEL_NAME += f'_bal-mixup-{args.mixup_alpha}' if args.bal_mixup else ''
    MODEL_NAME += f'_mixup-{args.mixup_alpha}' if args.mixup else ''
    MODEL_NAME += f'_decoupling-{args.decoupling_method}' if args.decoupling_method != '' else ''
    MODEL_NAME += f'_rw-{args.rw_method}' if args.rw_method != '' else ''
    MODEL_NAME += f'_{args.loss}'
    MODEL_NAME += '-drw' if args.drw else ''
    MODEL_NAME += f'_cb-beta-{args.cb_beta}' if args.rw_method == 'cb' else ''
    MODEL_NAME += f'_fl-gamma-{args.fl_gamma}' if args.loss == 'focal' else ''
    MODEL_NAME += f'_lr-{args.lr}'
    MODEL_NAME += f'_bs-{args.batch_size}'

    # Create output directory for model (and delete if already exists)
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    model_dir = os.path.join(args.out_dir, MODEL_NAME)
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    # Set all seeds for reproducibility
    set_seed(args.seed)

    # Create datasets + loaders
    if args.dataset == 'nih-cxr-lt':
        dataset = NIH_CXR_Dataset
        N_CLASSES = 20
    else:
        dataset = MIMIC_CXR_Dataset
        N_CLASSES = 19

    train_dataset = dataset(data_dir=args.data_dir, label_dir=args.label_dir, split='train')
    val_dataset = dataset(data_dir=args.data_dir, label_dir=args.label_dir, split='balanced-val')
    bal_test_dataset = dataset(data_dir=args.data_dir, label_dir=args.label_dir, split='balanced-test')
    test_dataset = dataset(data_dir=args.data_dir, label_dir=args.label_dir, split='test')

    if args.bal_mixup:
        cls_weights = [len(train_dataset) / cls_count for cls_count in train_dataset.cls_num_list]
        instance_weights = [cls_weights[label] for label in train_dataset.labels]
        sampler = torch.utils.data.WeightedRandomSampler(torch.Tensor(instance_weights), len(train_dataset))
        bal_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn, sampler=sampler)

        imbal_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

        train_loader = ComboLoader([imbal_train_loader, bal_train_loader])
    elif args.decoupling_method == 'cRT':
        cls_weights = [len(train_dataset) / cls_count for cls_count in train_dataset.cls_num_list]
        instance_weights = [cls_weights[label] for label in train_dataset.labels]
        sampler = torch.utils.data.WeightedRandomSampler(torch.Tensor(instance_weights), len(train_dataset))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn, sampler=sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=val_worker_init_fn)

    # Create csv documenting training history
    history = pd.DataFrame(columns=['epoch', 'phase', 'loss', 'balanced_acc', 'mcc', 'auroc'])
    history.to_csv(os.path.join(model_dir, 'history.csv'), index=False)

    # Set device
    device = torch.device('cuda:0')

    # Instantiate model
    model = torchvision.models.resnet50(pretrained=(not args.rand_init))
    model.fc = torch.nn.Linear(model.fc.in_features, N_CLASSES)

    if args.decoupling_method == 'tau_norm':
        msg = model.load_state_dict(torch.load(args.decoupling_weights, map_location='cpu')['weights'])
        print(f'Loaded weights from {args.decoupling_weights} with message: {msg}')

        model.fc.bias.data = torch.zeros_like(model.fc.bias.data)
        fc_weights = model.fc.weight.data.clone()

        weight_norms = torch.norm(fc_weights, 2, 1)

        model.fc.weight.data = torch.stack([fc_weights[i] / torch.pow(weight_norms[i], -4) for i in range(N_CLASSES)], dim=0)
    elif args.decoupling_method == 'cRT':
        msg = model.load_state_dict(torch.load(args.decoupling_weights, map_location='cpu')['weights'])
        print(f'Loaded weights from {args.decoupling_weights} with message: {msg}')

        model.fc = torch.nn.Linear(model.fc.in_features, N_CLASSES)  # re-initialize classifier head

    model = model.to(device)        

    # Set loss and weighting method
    if args.rw_method == 'sklearn':
        weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_dataset.labels), y=np.array(train_dataset.labels))
        weights = torch.Tensor(weights).to(device)
    elif args.rw_method == 'cb':
        weights = get_CB_weights(samples_per_cls=train_dataset.cls_num_list, beta=args.cb_beta)
        weights = torch.Tensor(weights).to(device)
    else:
        weights = None

    if weights is None:
        print('No class reweighting')
    else:
        print(f'Class weights with rw_method {args.rw_method}:')
        for i, c in enumerate(train_dataset.CLASSES):
            print(f'\t{c}: {weights[i]}')

    loss_fxn = get_loss(args, None if args.drw else weights, train_dataset)

    # Set optimizer
    if args.decoupling_method != '':
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)    
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train with early stopping
    if args.decoupling_method != 'tau_norm':
        epoch = 1
        early_stopping_dict = {'best_acc': 0., 'epochs_no_improve': 0}
        best_model_wts = None
        while epoch <= args.max_epochs and early_stopping_dict['epochs_no_improve'] <= args.patience:
            if args.bal_mixup:
                history = bal_mixup_train(model=model, device=device, loss_fxn=loss_fxn, optimizer=optimizer, data_loader=train_loader, history=history, epoch=epoch, model_dir=model_dir, classes=train_dataset.CLASSES, mixup_alpha=args.mixup_alpha)    
            else:
                history = train(model=model, device=device, loss_fxn=loss_fxn, optimizer=optimizer, data_loader=train_loader, history=history, epoch=epoch, model_dir=model_dir, classes=train_dataset.CLASSES, mixup=args.mixup, mixup_alpha=args.mixup_alpha)
            history, early_stopping_dict, best_model_wts = validate(model=model, device=device, loss_fxn=loss_fxn, optimizer=optimizer, data_loader=val_loader, history=history, epoch=epoch, model_dir=model_dir, early_stopping_dict=early_stopping_dict, best_model_wts=best_model_wts, classes=val_dataset.CLASSES)

            if args.drw and epoch == 10:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.1  # anneal LR
                loss_fxn = get_loss(args, weights, train_dataset)  # get class-weighted loss
                early_stopping_dict['epochs_no_improve'] = 0  # reset patience

            epoch += 1
    else:
        best_model_wts = model.state_dict()
    
    # Evaluate on balanced test set
    evaluate(model=model, device=device, loss_fxn=loss_fxn, dataset=bal_test_dataset, split='balanced-test', batch_size=args.batch_size, history=history, model_dir=model_dir, weights=best_model_wts)

    # Evaluate on imbalanced test set
    evaluate(model=model, device=device, loss_fxn=loss_fxn, dataset=test_dataset, split='test', batch_size=args.batch_size, history=history, model_dir=model_dir, weights=best_model_wts)

if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/ssd1/greg/NIH_CXR/images', type=str)
    parser.add_argument('--label_dir', default='labels/', type=str)
    parser.add_argument('--out_dir', default='results/', type=str, help="path to directory where results and model weights will be saved")
    parser.add_argument('--dataset', required=True, type=str, choices=['nih-lt', 'mimic-cxr-lt'])
    parser.add_argument('--loss', default='ce', type=str, choices=['ce', 'focal', 'ldam'])
    parser.add_argument('--drw', action='store_true', default=False)
    parser.add_argument('--rw_method', default='', choices=['', 'sklearn', 'cb'])
    parser.add_argument('--cb_beta', default=0.9999, type=float)
    parser.add_argument('--fl_gamma', default=2., type=float)
    parser.add_argument('--bal_mixup', action='store_true', default=False)
    parser.add_argument('--mixup', action='store_true', default=False)
    parser.add_argument('--mixup_alpha', default=0.2, type=float)
    parser.add_argument('--decoupling_method', default='', choices=['', 'cRT', 'tau_norm'], type=str)
    parser.add_argument('--decoupling_weights', type=str)
    parser.add_argument('--model_name', default='resnet50', type=str, help="CNN backbone to use")
    parser.add_argument('--max_epochs', default=60, type=int, help="maximum number of epochs to train")
    parser.add_argument('--batch_size', default=256, type=int, help="batch size for training, validation, and testing (will be lowered if TTA used)")
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--patience', default=15, type=int, help="early stopping 'patience' during training")
    parser.add_argument('--rand_init', action='store_true', default=False)
    parser.add_argument('--n_TTA', default=0, type=int, help="number of augmented copies to use during test-time augmentation (TTA), default 0")
    parser.add_argument('--seed', default=0, type=int, help="set random seed")

    args = parser.parse_args()

    print(args)

    main(args)