

import numpy as np
import pandas as pd
import os
import torch
from torch import nn

from models.patchtst_ssl.backbone import PatchTSTBackbone
from src.learner import Learner, get_model as unwrap_model
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import *


import argparse

parser = argparse.ArgumentParser()
# Dataset and dataloader
parser.add_argument('--dset_pretrain', type=str, default='etth1', help='dataset name')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
# Patch
parser.add_argument('--patch_len', type=int, default=12, help='patch length')
parser.add_argument('--stride', type=int, default=12, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=512, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
# Pretrain mask
parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio for the input')
# Optimization args
parser.add_argument('--n_epochs_pretrain', type=int, default=10, help='number of pre-training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--aux_weight', type=float, default=0.01, help='weight for auxiliary load balancing loss')
# model id to keep track of the number of models saved
parser.add_argument('--pretrained_model_id', type=int, default=1, help='id of the saved pretrained model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')


args = parser.parse_args()
print('args:', args)
args.save_pretrained_model = 'patchtst_pretrained_cw'+str(args.context_points)+'_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-pretrain' + str(args.n_epochs_pretrain) + '_mask' + str(args.mask_ratio)  + '_model' + str(args.pretrained_model_id)
args.save_path = 'saved_models/' + args.dset_pretrain + '/masked_patchtst/' + args.model_type + '/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)


# get available GPU devide
set_device()


class MaskedPatchTST(nn.Module):
    def __init__(self, c_in: int, args):
        super().__init__()
        self.backbone = PatchTSTBackbone(
            n_vars=c_in,
            seq_len=args.context_points,
            patch_len=args.patch_len,
            stride=args.stride,
            n_layers=args.n_layers,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
            attn_dropout=args.dropout,
            act='relu',
            revin=bool(args.revin),
        )
        self.head = PretrainHead(self.backbone.d_model, args.patch_len, args.head_dropout)

    def forward(self, x):
        latent = self.backbone(x)
        preds = self.head(latent)
        return preds, self.backbone.aux_loss


class PretrainHead(nn.Module):
    def __init__(self, d_model: int, patch_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = latent.transpose(2, 3)
        x = self.linear(self.dropout(x))
        x = x.permute(0, 2, 1, 3)
        return x


def build_model(c_in, args):
    model = MaskedPatchTST(c_in=c_in, args=args)
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def find_lr():
    # get dataloader
    dls = get_dls(args)    
    model = build_model(dls.vars, args)
    # get loss
    loss_func = nn.HuberLoss()
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio)]
        
    # define learner
    learn = Learner(dls, model,
                        loss_func,
                        lr=args.lr,
                        cbs=cbs,
                        aux_weight=args.aux_weight,
                        )
    # fit the data to the model
    suggested_lr = learn.lr_finder()
    print('suggested_lr', suggested_lr)
    return suggested_lr


def pretrain_func(lr=args.lr):
    # get dataloader
    dls = get_dls(args)
    # get model     
    model = build_model(dls.vars, args)
    # get loss
    loss_func = nn.HuberLoss()
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [
         PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio),
         SaveModelCB(monitor='valid_loss', fname=args.save_pretrained_model,                       
                        path=args.save_path)
        ]
    # define learner
    learn = Learner(dls, model,
                        loss_func,
                        lr=lr,
                        cbs=cbs,
                        #metrics=[mse]
                        aux_weight=args.aux_weight,
                        )
    # fit the data to the model
    learn.fit_one_cycle(n_epochs=args.n_epochs_pretrain, lr_max=lr)

    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(args.save_path + args.save_pretrained_model + '_losses.csv', float_format='%.6f', index=False)
    save_backbone_checkpoint(learn, os.path.join(args.save_path, args.save_pretrained_model + '_backbone.pth'))


def save_backbone_checkpoint(learner: Learner, path: str) -> None:
    state = {'backbone_state_dict': unwrap_model(learner.model).backbone.state_dict()}
    torch.save(state, path)
    print(f'saved backbone checkpoint to {path}')


if __name__ == '__main__':
    
    args.dset = args.dset_pretrain
    suggested_lr = find_lr()
    # Pretrain
    pretrain_func(suggested_lr)
    print('pretraining completed')
    

