import os
import click
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils
from global_constants import income_const
from model import IncomeClassifier, IncomeClassifierConstants
from dataset import FeatDataset
from focal_loss import FocalLoss


def eval_model(model_const,dataset,exp_const):
    model = IncomeClassifier(model_const)
    loaded_object = torch.load(model_const.model_path)
    model.load_state_dict(loaded_object['State'])
    epoch, step, it = [loaded_object[k] for k in ['Epoch','Step','Iter']]
    train_acc = loaded_object['Accuracy']['train']
    val_acc = loaded_object['Accuracy']['val']

    print('Creating dataloaders ...')
    dataloader = DataLoader(
        dataset,
        batch_size=exp_const.batch_size,
        shuffle=False,
        num_workers=exp_const.num_workers)
    
    with torch.no_grad():
        to_log, log_str = evaluation(
            model,
            dataloader,
            exp_const,
            epoch,
            it,
            step)

    log_str += f'Train Accuracy: {train_acc} | Val Accuracy: {val_acc}'
    print(log_str)



def evaluation(model,dataloader,exp_const,epoch,it,step):
    criterion = nn.CrossEntropyLoss()

    total_correct = 0
    total_loss = 0
    total_samples = 0
    for data in dataloader:
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
        'Iter': it,
        'Step': step,
        'Loss': round(total_loss,4),
        'Test Accuracy': round(acc*100,2)
    }

    log_str = f'[test] '
    for k,v in to_log.items():
        log_str += f'{k}: {v} | '

    return to_log, log_str


@click.command()
@click.option(
    '--exp_name',
    default='default_exp',
    show_default=True,
    type=str,
    help='Name of the experiment')
@click.option(
    '--num_hidden_blocks',
    default=2,
    show_default=True,
    type=int,
    help='Number of hidden blocks in the classifier')
def main(**kwargs):
    exp_const = utils.Constants()
    exp_const.exp_dir = os.path.join(income_const['exp_dir'],kwargs['exp_name'])
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.batch_size = 256
    exp_const.num_workers = 1

    model_const = IncomeClassifierConstants()
    model_const.num_hidden_blocks = kwargs['num_hidden_blocks']
    model_const.model_path = os.path.join(exp_const.model_dir,'best_model')

    dataset = FeatDataset('test')

    eval_model(model_const,dataset,exp_const)


if __name__=='__main__':
    main()