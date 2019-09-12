import os
import click
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from global_constants import income_const
from model import IncomeClassifier
from dataset import FeatDataset
from focal_loss import FocalLoss


def train_model(model,datasets,exp_const):
    utils.mkdir_if_not_exist(exp_const.exp_dir)
    utils.mkdir_if_not_exist(exp_const.log_dir)
    utils.mkdir_if_not_exist(exp_const.model_dir)

    dataloaders = {}
    dataloaders['train'] = DataLoader(
        datasets['train'],
        batch_size=exp_const.batch_size,
        shuffle=True,
        num_workers=exp_const.num_workers)
    dataloaders['val'] = DataLoader(
        datasets['val'],
        batch_size=exp_const.batch_size,
        shuffle=False,
        num_workers=exp_const.num_workers)

    opt = optim.Adam(
        model.parameters(),
        lr=exp_const.lr)

    #criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(exp_const.gamma)

    for epoch in range(exp_const.num_epochs):
        for step,data in enumerate(dataloaders['train']):
            model.train()
            
            logits,probs = model(data['feat'])
            #loss = criterion(logits,data['label'].long())
            loss = criterion(probs,data['label'])
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step%10==0:
                to_log = {
                    'Epoch': epoch,
                    'Step': step,
                    'Loss': loss.item(),
                }
                log_str = '[Train] '
                for k,v in to_log.items():
                    log_str += f'{k}: {v} | '
                
                print(log_str)

            if step%100==0:
                validation(model,dataloaders['val'],exp_const,epoch,step)


def validation(model,dataloader,exp_const,epoch,step):
    criterion = nn.CrossEntropyLoss()

    total_correct = 0
    total_loss = 0
    total_samples = 0
    for step,data in enumerate(dataloader):
        model.eval()
        
        logits,probs = model(data['feat'])
        loss = criterion(logits,data['label'].long())
        pred_label = probs[:,1] > 0.5
        correct = torch.sum(pred_label==data['label'].bool())

        batch_size = logits.size(0)
        total_correct += correct.item()
        total_loss += loss.item()*batch_size
        total_samples += batch_size
    
    total_loss = total_loss / total_samples
    acc = total_correct / total_samples

    to_log = {
        'Epoch': epoch,
        'Step': step,
        'Loss': total_loss,
        'Accuracy': acc
    }
    log_str = '[Val] '
    for k,v in to_log.items():
        log_str += f'{k}: {v} | '
    
    print(log_str)


@click.command()
@click.option(
    '--exp_name',
    default='default_exp',
    type=str,
    help='Name of the experiment')
def main(**kwargs):
    exp_const = utils.Constants()
    exp_const.exp_dir = os.path.join(income_const['exp_dir'],kwargs['exp_name'])
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'logs')
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.lr = 1e-3
    exp_const.num_epochs = 100
    exp_const.batch_size = 256
    exp_const.num_workers = 1
    exp_const.gamma = 0.1

    model = IncomeClassifier(105,105)

    datasets = {
        'train': FeatDataset('train'),
        'val': FeatDataset('val')
    }

    train_model(model,datasets,exp_const)


if __name__=='__main__':
    main()