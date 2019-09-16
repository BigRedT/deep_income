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


def train_model(model_const,datasets,exp_const):
    torch.manual_seed(exp_const.seed)
    np.random.seed(exp_const.seed)

    utils.mkdir_if_not_exist(exp_const.exp_dir)
    utils.mkdir_if_not_exist(exp_const.log_dir)
    utils.mkdir_if_not_exist(exp_const.model_dir)

    print('Create tensorboard writer ...')
    tb_writer = SummaryWriter(log_dir=exp_const.log_dir)

    print('Creating model ...')
    model = IncomeClassifier(model_const)
    print(model)

    print('Creating dataloaders ...')
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

    print('Creating optimizer ...')
    opt = optim.Adam(
        model.parameters(),
        lr=exp_const.lr,
        weight_decay=exp_const.weight_decay)

    if exp_const.loss=='cross_entropy':
        print('Cross Entropy Loss selected for training')
        criterion = nn.CrossEntropyLoss()
    elif exp_const.loss=='focal':
        print('Focal Loss selected for training')
        criterion = FocalLoss(exp_const.gamma)
    else:
        assert(False),'Requested loss not implemented'

    best_val_acc = 0
    step = 0
    for epoch in range(exp_const.num_epochs):
        for it,data in enumerate(dataloaders['train']):
            model.train()
            
            logits,probs = model(data['feat'])

            if exp_const.loss=='cross_entropy':
                loss = criterion(logits,data['label'].long())
            elif exp_const.loss=='focal':
                loss = criterion(probs,data['label'])
            else:
                assert(False), 'Requested loss not implemented'
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step%20==0:
                to_log = {
                    'Epoch': epoch,
                    'Iter': it,
                    'Step': step,
                    'Loss': round(loss.item(),4),
                }
                log_str = '[train] '
                for k,v in to_log.items():
                    log_str += f'{k}: {v} | '
                
                print(log_str)
                tb_writer.add_scalar('Loss/TrainBatch',to_log['Loss'],step)


            if step%100==0:
                print('-'*100)
                print('Evaluation')
                with torch.no_grad():
                    train_loss, train_acc = validation(
                        model,
                        dataloaders['train'],
                        exp_const,
                        epoch,
                        it,
                        step,
                        'train')
                    val_loss, val_acc = validation(
                        model,
                        dataloaders['val'],
                        exp_const,
                        epoch,
                        it,
                        step,
                        'val')
                print('-'*100)

                tb_writer.add_scalar('Loss/Train',train_loss,step)
                tb_writer.add_scalar('Loss/Val',val_loss,step)
                tb_writer.add_scalar('Accuracy/Train',train_acc,step)
                tb_writer.add_scalar('Accuracy/val',val_acc,step)

                if val_acc > best_val_acc:
                    to_save = {
                        'State': model.state_dict(),
                        'Accuracy': {'val': val_acc,'train': train_acc},
                        'Iter': it,
                        'Step': step,
                        'Epoch': epoch
                    }

                    model_path = os.path.join(exp_const.model_dir,'best_model')
                    torch.save(to_save,model_path)

                    best_val_acc = val_acc

            step += 1
    
    tb_writer.close()


def validation(model,dataloader,exp_const,epoch,it,step,subset):
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
        'Accuracy': round(acc*100,2)
    }

    log_str = f'[{subset}] '
    for k,v in to_log.items():
        log_str += f'{k}: {v} | '
        
    print(log_str)

    return to_log['Loss'], to_log['Accuracy']


@click.command()
@click.option(
    '--exp_name',
    default='default_exp',
    show_default=True,
    type=str,
    help='Name of the experiment')
@click.option(
    '--loss',
    default='cross_entropy',
    show_default=True,
    type=click.Choice(['cross_entropy','focal']),
    help='Loss used for training')
@click.option(
    '--num_hidden_blocks',
    default=2,
    show_default=True,
    type=int,
    help='Number of hidden blocks in the classifier')
def main(**kwargs):
    exp_const = utils.Constants()
    exp_const.exp_dir = os.path.join(income_const['exp_dir'],kwargs['exp_name'])
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'logs')
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.lr = 1e-3
    exp_const.weight_decay = 0
    exp_const.num_epochs = 50
    exp_const.batch_size = 256
    exp_const.num_workers = 2
    exp_const.gamma = 0.5
    exp_const.loss = kwargs['loss']
    exp_const.seed = 0

    model_const = IncomeClassifierConstants()
    model_const.num_hidden_blocks = kwargs['num_hidden_blocks']

    datasets = {
        'train': FeatDataset('train'),
        'val': FeatDataset('val')
    }

    train_model(model_const,datasets,exp_const)

    model_path = os.path.join(exp_const.model_dir,'best_model')

    print('Performance of best model selected during training ...')
    state = torch.load(model_path)
    print('Accuracy:\n\tTrain:',state['Accuracy']['train'],
        '\n\tVal:',state['Accuracy']['val'])
    print('Early stopping: \n\tEpoch:',state['Epoch'],
        '\n\tIter:',state['Iter'],'\n\tStep:',state['Step'])


if __name__=='__main__':
    main()