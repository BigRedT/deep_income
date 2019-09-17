import os
import click
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from global_constants import income_const


class FeatDataset(Dataset):
    def __init__(self,subset):
        self.subset = subset
        self.feats, self.labels, self.sample_ids = self.load_data()

    def load_data(self):
        if self.subset=='test':
            subset_npy = 'test_npy'
        else:
            subset_npy = 'train_val_npy'

        feats = np.load(os.path.join(
            income_const['proc_dir'],
            income_const[subset_npy]['feat']))
        labels = np.load(os.path.join(
            income_const['proc_dir'],
            income_const[subset_npy]['label']))
        sample_ids = np.load(os.path.join(
            income_const['proc_dir'],
            income_const['sample_ids_npy'][self.subset]))
        
        return feats, labels, sample_ids

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self,i):
        idx = self.sample_ids[i]
        
        to_return = {
            'feat': self.feats[idx],
            'label': self.labels[idx],
        }

        return to_return


def test_dataset():
    train_dataset = FeatDataset('train')
    val_dataset = FeatDataset('val')
    test_dataset = FeatDataset('test')
    import pdb; pdb.set_trace()


def test_dataloader():
    dataset = FeatDataset('train')
    dataloader = DataLoader(
        dataset,
        batch_size=5,
        num_workers=2,
        shuffle=True,
        drop_last=False)
    for data in dataloader:
        import pdb; pdb.set_trace()


@click.command()
@click.option(
    '--test',
    type=click.Choice(['dataset','dataloader']),
    help='Which test to run')
def test_main(test):
    if test=='dataset':
        test_dataset()
    elif test=='dataloader':
        test_dataloader()
    else:
        assert(False),'test not implemented'


if __name__=='__main__':
    test_main()