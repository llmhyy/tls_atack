import json
import math
import mmap
import numpy as np
from functools import partial
from random import shuffle
from keras.utils import Sequence
from keras.preprocessing.sequence import pad_sequences



def find_lines(data):
    for i, char in enumerate(data):
        if char == b'\n':
            yield i 

def get_mmapdata_and_byteoffset(feature_file):
    ########################################################################
    # Biggest saviour: shuffling a large file w/o loading in memory
    # >>> https://stackoverflow.com/questions/24492331/shuffle-a-large-list-of-items-without-loading-in-memory
    ########################################################################

    # Creating a list of byte offset for each line
    with open(feature_file, 'r') as f:
        mmap_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        start = 0
        byte_offset = []
        for end in find_lines(mmap_data):
            byte_offset.append((start, end))
            start = end + 1
    return mmap_data, byte_offset

def get_min_max(mmap_data, byte_offset):
    min_feature = None
    max_feature = None
    for start,end in byte_offset:
        dataline = mmap_data[start:end+1].decode('ascii').strip().rstrip(',')
        dataline = np.array(json.loads('['+dataline+']'))
        if min_feature is not None:
            dataline = np.vstack((min_feature, dataline))
        if max_feature is not None:
            dataline = np.vstack((max_feature, dataline))
        min_feature = np.min(dataline, axis=0)
        max_feature = np.max(dataline, axis=0)
    return (min_feature, max_feature)

def split_train_test(byte_offset, split_ratio, seed):
    # Shuffling the indices to give a random train test split
    indices = np.random.RandomState(seed=seed).permutation(len(byte_offset)) 
    split_idx = math.ceil((1-split_ratio)*len(byte_offset))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    train_byte_offset = [byte_offset[idx] for idx in train_idx]
    test_byte_offset = [byte_offset[idx] for idx in test_idx]

    return train_byte_offset, test_byte_offset

def normalize(option, min_max_feature=None):
    def l2_norm(batch_data):
        l2_norm = np.linalg.norm(batch_data, axis=2, keepdims=True)
        batch_data = np.divide(batch_data, l2_norm, out=np.zeros_like(batch_data), where=l2_norm!=0.0)
        return batch_data
    def min_max_norm(batch_data, min_max_feature):
        num = batch_data-min_max_feature[0]
        den = min_max_feature[1]-min_max_feature[0]
        batch_data = np.divide(num, den, out=np.zeros_like(num), where=den!=0.0)    
        return batch_data
    if option == 1:
        return l2_norm
    elif option == 2:
        if min_max_feature is not None:
            return partial(min_max_norm, min_max_feature=min_max_feature)
        else:
            print("Error: min-max range for feature is not provided")
            return

class BatchGenerator(Sequence):
    def __init__(self, mmap_data, byte_offset, batch_size, sequence_len, norm_fn, return_seq_len=False):
        self.mmap_data = mmap_data
        self.byte_offset = byte_offset
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.norm_fn = norm_fn
        # self.min_feature = min_max_feature[0]
        # self.max_feature = min_max_feature[1]
        self.return_seq_len = return_seq_len

    def __len__(self):
        return int(np.ceil(len(self.byte_offset)/float(self.batch_size)))

    def __getitem__(self, idx):
        batch_idx = self.byte_offset[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_data = []
        for start,end in batch_idx:
            dataline = self.mmap_data[start:end+1].decode('ascii').strip().rstrip(',')
            batch_data.append(json.loads('['+dataline+']'))
        
        if self.return_seq_len:
            batch_seq_len = [len(data) for data in batch_data]

        # Pad the sequence
        batch_data = pad_sequences(batch_data, maxlen=self.sequence_len, dtype='float32', padding='post',value=0.0)

        # Scale the features
        batch_data = self.norm_fn(batch_data)
        # (1)   L2 Normalization
        # l2_norm = np.linalg.norm(batch_data, axis=2, keepdims=True)
        # batch_data = np.divide(batch_data, l2_norm, out=np.zeros_like(batch_data), where=l2_norm!=0.0)
        # (2)   Min Max Normalization
        # batch_data = (batch_data - self.min_feature)/(self.max_feature - self.min_feature)
        # num = batch_data-self.min_feature
        # den = self.max_feature-self.min_feature
        # batch_data = np.divide(num, den, out=np.zeros_like(num), where=den!=0.0)

        # Append zero to the start of the sequence
        packet_zero = np.zeros((batch_data.shape[0],1,batch_data.shape[2]))
        batch_data = np.concatenate((packet_zero, batch_data), axis=1)

        # Split the data into inputs and targets
        batch_inputs = batch_data[:,:-1,:]
        batch_targets = batch_data[:,1:,:]

        if self.return_seq_len:
            return (batch_inputs, batch_targets, batch_seq_len)
        
        return (batch_inputs, batch_targets)
    
    def on_epoch_end(self):
        shuffle(self.byte_offset)