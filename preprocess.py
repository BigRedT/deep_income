import os
import numpy as np

from global_constants import income_const
from income_attributes import *
import utils


def read_raw_data(filepath,test=False):
    with open(filepath,'r') as f:
        data = f.readlines()

    last_idx = -1   # Get rid of '\n'
    if test==True:
        data = data[1:] # Get rid of first line
        last_idx = -2   # Get rid of '.\n'

    data = data[:-1] # Get rid of last line ''

    num_samples = len(data)
    for i in range(num_samples):
        data[i] = data[i][:last_idx].split(', ')

    return data


def discard_missing_data(data):
    num_samples = len(data)
    for i in range(num_samples):
        if '?' in data[i]:
            data[i] = None
        
    data = [sample for sample in data if sample is not None]
    return data


def convert_sample_to_feature_vector(sample):
    D = len(ATTRS)
    vec = np.zeros(D)
    for i,attr_type in enumerate(ATTR_TYPES):
        attr_value = sample[i]
        if attr_type in CONTINUOUS_ATTRS:
            vec_idx = ATTR_TO_IDX[CONTINUOUS_ATTRS[attr_type]]
            attr_value = float(attr_value)
        else:
            vec_idx = ATTR_TO_IDX[attr_value]
            attr_value = 1.0
        
        vec[vec_idx] = attr_value

    return vec


def convert_data_to_feature_vectors(data):
    feat = [convert_sample_to_feature_vector(sample) for sample in data]
    return convert_to_npy(feat)
    

def convert_to_npy(data):
    return np.array(data,dtype=np.float32)


def normalize(data,mean=None,std=None):
    if mean is None:
        mean = np.mean(data,0)
    
    if std is None:
        std = np.std(data,0)

    data = (data - mean) / (std + 1e-6)

    return data, mean, std


def get_labels(data):
    num_samples = len(data)
    labels = np.zeros(num_samples,dtype=np.float32)
    for i, sample in enumerate(data):
        labels[i] = LABELS[sample[-1]]
    
    return labels


def preprocess_train_val():
    print('Preprocess train-val data ...')
    filepath = os.path.join(
        income_const['download_dir'],
        income_const['urls']['data']['name'])
    
    print('\tRead raw data ...')
    data = read_raw_data(filepath)
    print('\t\tTotal samples:',len(data))
    
    print('\tDiscard samples with missing features ...')
    data = discard_missing_data(data)
    print('\t\tSamples without missing features:',len(data))
    
    print('\tConvert samples to feature vectors ...')
    feat = convert_data_to_feature_vectors(data)
    print('\t\tNormalize features ...')
    feat, feat_mean, feat_std = normalize(feat)
    print('\t\tFeature matrix dims:',feat.shape)
    
    print('\tSave features to npy file ...')
    proc_dir = income_const['proc_dir']
    filename = os.path.join(proc_dir,income_const['train_val_npy']['feat'])
    np.save(filename,feat)
    filename = os.path.join(proc_dir,income_const['train_val_npy']['feat_mean'])
    np.save(filename,feat_mean)
    filename = os.path.join(proc_dir,income_const['train_val_npy']['feat_std'])
    np.save(filename,feat_std)

    print('\tRead labels ...')
    labels = get_labels(data)
    print('\t\tLabels matrix dims:',labels.shape)
    num_pos = np.sum(labels)
    percent_pos = round(100*num_pos/labels.shape[0],2)
    print(f'\t\tPercentage of positives: {percent_pos}%')
    
    print('\tSave labels to npy file ...')
    filename = os.path.join(proc_dir,income_const['train_val_npy']['label'])
    np.save(filename,labels)
    
    print('\tCreate train-val split ...')
    num_train_val_samples = feat.shape[0]
    num_train_samples = int(
        income_const['train_val_split']*num_train_val_samples)
    sample_ids = np.arange(num_train_val_samples)
    np.random.shuffle(sample_ids)
    train_ids = sample_ids[:num_train_samples]
    val_ids = sample_ids[num_train_samples:]
    print('\t\tNum train samples:',len(train_ids))
    print('\t\tNum val samples:',len(val_ids))

    print('\tSave train and val sample ids ...')
    filename = os.path.join(proc_dir,income_const['sample_ids_npy']['train'])
    np.save(filename,train_ids)
    filename = os.path.join(proc_dir,income_const['sample_ids_npy']['val'])
    np.save(filename,val_ids)


def preprocess_test():
    print('Preprocess test data ...')
    filepath = os.path.join(
        income_const['download_dir'],
        income_const['urls']['test']['name'])
    
    print('\tRead raw data ...')
    data = read_raw_data(filepath,test=True)
    print('\t\tTotal samples:',len(data))
    
    print('\tDiscard samples with missing features ...')
    data = discard_missing_data(data)
    print('\t\tSamples without missing features:',len(data))
    
    print('\tConvert samples to feature vectors ...')
    feat = convert_data_to_feature_vectors(data)
    proc_dir = income_const['proc_dir']
    print('\t\tNormalize features ...')
    filename = os.path.join(proc_dir,income_const['train_val_npy']['feat_mean'])
    feat_mean = np.load(filename)
    filename = os.path.join(proc_dir,income_const['train_val_npy']['feat_std'])
    feat_std = np.load(filename)
    feat, _, _ = normalize(feat,feat_mean,feat_std)
    print('\t\tFeature matrix dims:',feat.shape)
    
    print('\tSave features to npy file ...')
    proc_dir = income_const['proc_dir']
    filename = os.path.join(proc_dir,income_const['test_npy']['feat'])
    np.save(filename,feat)

    print('\tRead labels ...')
    labels = get_labels(data)
    print('\t\tLabels matrix dims:',labels.shape)
    num_pos = np.sum(labels)
    percent_pos = round(100*num_pos/labels.shape[0],2)
    print(f'\t\tPercentage of positives: {percent_pos}%')
    
    print('\tSave labels to npy file ...')
    filename = os.path.join(proc_dir,income_const['test_npy']['label'])
    np.save(filename,labels)

    print('\tSave sample ids ...')
    num_samples = feat.shape[0]
    filename = os.path.join(proc_dir,income_const['sample_ids_npy']['test'])
    np.save(filename,np.arange(num_samples))


def main():
    np.random.seed(0)
    utils.mkdir_if_not_exist(income_const['proc_dir'])
    preprocess_train_val()
    preprocess_test()


if __name__=='__main__':
    main()