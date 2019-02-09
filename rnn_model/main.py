import os
import json
import math
import mmap
import argparse
import tracemalloc
import numpy as np
from sys import getsizeof
from datetime import datetime
from random import shuffle
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
import keras.backend as K
from keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

import parse_features as pf
import visualization as viz
import diagnostic as diag
import metric as mt

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--norm', help='Input normalization options for features', default=1, type=int, choices=[1,2,3])
parser.add_argument('-e', '--epoch', help='Input epoch for training', default=100, type=int)
parser.add_argument('-t', '--traffic', help='Input top-level directory of the traffic module containing extracted features', required=True)
parser.add_argument('-f', '--feature', help='Input filename of feature to be used', required=True)
parser.add_argument('-s', '--show', help='Flag for displaying plots', action='store_true', default=False)
parser.add_argument('-m', '--model', help='Input directory for existing model to be trained')
args = parser.parse_args()

# Config info
DATETIME_NOW = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
BATCH_SIZE = 64
SEQUENCE_LEN = 100
EPOCH = args.epoch
SAVE_EVERY_EPOCH = 5
SPLIT_RATIO = 0.3
SEED = 2019
feature_file = args.feature
existing_model = args.model

tracemalloc.start()

##########################################################################################

# DATA LOADING AND PREPROCESSING

##########################################################################################

# Search for extracted_features directory
extracted_features = os.path.join(args.traffic, 'extracted_features')
if not os.path.exists(extracted_features):
    raise FileNotFoundError("Directory extracted_features not found. Extract features first")

####################################################################
# OPTION 1: USING A GENERATOR TO LOAD DATA 
####################################################################

def find_lines(data):
    for i, char in enumerate(data):
        if char == b'\n':
            yield i 

class BatchGenerator(Sequence):
    def __init__(self, data, start_end_line, batch_size, sequence_len, return_seq_len=False):
        self.data = data
        self.start_end_line = start_end_line
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.return_seq_len = return_seq_len

    def __len__(self):
        return int(np.ceil(len(self.start_end_line)/float(self.batch_size)))

    def __getitem__(self, idx):
        batch_idx = self.start_end_line[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_data = []
        for start,end in batch_idx:
            dataline = self.data[start:end+1].decode('ascii').strip().rstrip(',')
            batch_data.append(json.loads('['+dataline+']'))
        
        if self.return_seq_len:
            batch_seq_len = [len(data) for data in batch_data]

        # Pad the sequence
        batch_data = pad_sequences(batch_data, maxlen=self.sequence_len, dtype='float32', padding='post',value=0.0)

        # Scale the features
        l2_norm = np.linalg.norm(batch_data, axis=2, keepdims=True)
        batch_data = np.divide(batch_data, l2_norm, out=np.zeros_like(batch_data), where=l2_norm!=0.0)

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
        shuffle(self.start_end_line)

# def data_batch_generator(data, start_end_line, batch_size, sequence_len):
#     # Shuffle the data before the start of every epoch such that each epoch is different
#     shuffle(start_end_line)
#     counter = 0
#     while True:

#         batch_data = []
#         for i in range(batch_size):
#             start, end = start_end_line[counter]
#             dataline = data[start:end+1].decode('ascii').strip().rstrip(',')
#             batch_data.append(json.loads('['+dataline+']'))
#             counter+=1

#         # Pad the sequence
#         batch_data = pad_sequences(batch_data, maxlen=sequence_len, dtype='float32', padding='post',value=0.0)

#         # Scale the features
#         l2_norm = np.linalg.norm(batch_data, axis=2, keepdims=True)
#         batch_data = np.divide(batch_data, l2_norm, out=np.zeros_like(batch_data), where=l2_norm!=0.0)

#         # Append zero to the start of the sequence
#         packet_zero = np.zeros((batch_data.shape[0],1,batch_data.shape[2]))
#         batch_data = np.concatenate((packet_zero, batch_data), axis=1)

#         # Split the data into inputs and targets
#         batch_inputs = batch_data[:,:-1,:]
#         batch_targets = batch_data[:,1:,:]

#         yield (batch_inputs, batch_targets)

# Creating a list of byte offset for each line
with open(os.path.join(extracted_features, feature_file), 'r') as f:
    data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    start = 0
    lines = []
    for end in find_lines(data):
        lines.append((start, end))
        start = end + 1

# Shuffling the indices to give a random train test split
indices = np.random.RandomState(seed=SEED).permutation(len(lines)) 
split_idx = math.ceil((1-SPLIT_RATIO)*len(lines))
train_idx, test_idx = indices[:split_idx], indices[split_idx:]
train_start_end_line = [lines[idx] for idx in train_idx]
test_start_end_line = [lines[idx] for idx in test_idx]

# Intializing constants
SAMPLE_SIZE = len(lines)
TRAIN_SIZE = len(train_start_end_line)
TEST_SIZE = len(test_start_end_line)
sample_traffic = json.loads('['+data[train_start_end_line[0][0]:train_start_end_line[0][1]+1].decode('ascii').strip().rstrip(',')+']')
INPUT_DIM = len(sample_traffic[0])

# Initialize the train and test generators
train_generator = BatchGenerator(data, train_start_end_line, BATCH_SIZE, SEQUENCE_LEN)
test_generator = BatchGenerator(data, test_start_end_line, BATCH_SIZE, SEQUENCE_LEN)

####################################################################
# OPTION 2: LOADING THE DATASET INTO MEMORY 
####################################################################

# # Load the dataset into memory 
# features = json.loads(pf.get_features(os.path.join(os.path.join(extracted_features, feature_file))))

# # Initialize useful constants and pad the sequences to have equal length
# batch_dim = len(features)
# sequence_dim = max(list(map(len, features)))
# feature_dim = features[0][0]
# traffic_length = np.asarray([len(sample) for sample in features])
# features_pad = pad_sequences(features, maxlen=SEQUENCE_LEN, dtype='float32', padding='post',value=0.0)
# #print(features_pad.shape)  # shape (3496,100,19)

# # Feature scaling
# # 1: Normalize each sample independently into unit vectors
# # 2: Standardize each feature across ALL traffic (http://cs231n.github.io/neural-networks-2/)
# # Shoud I standardize within 1 traffic only? Hmm wouldnt the padding affect the standardization as well
# # 3: Scale each feature between min and max of feature

# if args.norm == 1:
#     l2_norm = np.linalg.norm(features_pad, axis=2, keepdims=True)
#     features_scaled = np.divide(features_pad, l2_norm, out=np.zeros_like(features_pad), where=l2_norm!=0.0)

# elif args.norm == 2:
#     flattened = features_pad.reshape((features_pad.shape[0]*features_pad.shape[1],features_pad.shape[2]))
#     zero_centered = features_pad - np.mean(flattened, axis=0, keepdims=True)
#     std_dev = np.std(flattened, axis=0, keepdims=True)
#     features_scaled = np.divide(zero_centered, std_dev, out=np.zeros_like(features_pad), where=std_dev!=0.0) 

# elif args.norm == 3:
#     features_max = np.amax(features_pad, axis=(0,1))
#     features_min = np.amin(features_pad, axis=(0,1))
#     num = (features_pad-features_min)
#     dem = (features_max-features_min)
#     features_scaled = np.divide(num,dem,out=np.zeros_like(num), where=dem!=0.0)

# # Generate training input and output by lagging 1 timestep
# zero_features = np.zeros((features_scaled.shape[0],1,features_scaled.shape[2]))
# features_zero_appended = np.concatenate((zero_features, features_scaled), axis=1)
# X = features_zero_appended[:,:-1,:]
# Y = features_zero_appended[:,1:,:]
# # print(X.shape)     # shape (3496,100,19)
# # print(Y.shape)     # shape (3496,100,19)

# # Split dataset into train and test
# ### Using scipy's train_test_split ###
# # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=SPLIT_RATIO, random_state=SEED)
# # X_train_seqlen, X_test_seqlen, Y_train_seqlen, Y_test_seqlen = train_test_split(traffic_length, traffic_length, test_size=SPLIT_RATIO, random_state=SEED)
# ### Using raw numpy operations ###
# indices = np.random.RandomState(seed=SEED).permutation(X.shape[0])
# split_idx = math.ceil((1-SPLIT_RATIO)*X.shape[0])
# train_idx, test_idx = indices[:split_idx], indices[split_idx:]
# X_train, X_test, Y_train, Y_test = X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]
# X_train_seqlen, X_test_seqlen, Y_train_seqlen, Y_test_seqlen = traffic_length[train_idx], traffic_length[test_idx], traffic_length[train_idx], traffic_length[test_idx]

##########################################################################################
 
# MODEL BUILDING

##########################################################################################

# Build RNN model or Load existing RNN model
if existing_model:
    model = load_model(existing_model)
    model.summary()
else:
    model = Sequential()
    model.add(LSTM(INPUT_DIM, input_shape=(SEQUENCE_LEN,INPUT_DIM), return_sequences=True))
    model.add(Activation('relu'))
    model.summary()

    # Selecting optimizers 
    model.compile(loss='mean_squared_error',
                    optimizer='rmsprop')

class PredictEpoch(keras.callbacks.Callback):
    def __init__(self, train_generator, test_generator):
        self.train_generator = train_generator
        self.test_generator = test_generator

    def on_train_begin(self, logs={}):
        self.predict_train = []
        self.predict_test = []

    def on_epoch_end(self, epoch, logs={}):
        # At the end of every epoch, we make a prediction and evaluate its accuracy, instead of savings the prediction...too much MEM!
        if epoch%SAVE_EVERY_EPOCH==0:
            self.predict_train.append(self.model.predict_generator(self.train_generator))
            self.predict_test.append(self.model.predict_generator(self.test_generator))

# Training the RNN model
predictEpoch = PredictEpoch(train_generator, test_generator)
####################################################################
# OPTION 1: USING A GENERATOR TO LOAD DATA 
####################################################################
history = model.fit_generator(train_generator, steps_per_epoch=math.ceil(TRAIN_SIZE/BATCH_SIZE), 
                                                epochs=EPOCH, 
                                                callbacks=[predictEpoch], 
                                                validation_data=test_generator, 
                                                validation_steps=math.ceil(TEST_SIZE/BATCH_SIZE), 
                                                workers=4, 
                                                use_multiprocessing=True)

####################################################################
# OPTION 2: LOADING THE DATASET INTO MEMORY 
####################################################################
# history = model.fit(X_train,Y_train, validation_data = (X_test, Y_test),batch_size=BATCH_SIZE, epochs=EPOCH, callbacks=[predictEpoch])

# Predict with RNN model for a randomly chosen sample from the test dataset
# sample = X_test[[3]]
# Y_actual = Y_test[[3]]
# Y_predict = model.predict(sample)
# print('Actual: ')
# print(Y_actual[0])
# print('Actual shape: {}'.format(Y_actual[0].shape))
# print('Predicted: ')
# print(Y_predict[0])
# print('Predicted shape: {}'.format(Y_predict[0].shape))
# Y_actual = Y_actual.reshape(1, Y_actual.shape[1]*Y_actual.shape[2])
# Y_predict = Y_predict.reshape(1, Y_predict.shape[1]*Y_predict.shape[2])
# sample_score = cosine_similarity(Y_actual, Y_predict)
# print('Cosine similarity for sample: {}'.format(sample_score[0]))

# Calculate cosine similarity for ONE packet and across traffic
def cos_sim_onepacket(predict_data, true_data, packet_id=0):
    """
    Calculates the cosine similarity for one packet of every traffic (default first packet)

    Return a 2-tuple:
    (cos sim of one packet for final prediction, list of tuple (mean, median)) 
    where len(list) corresponds to #epoch
    """
    cos_sim_firstpacket_epoch = []
    true_data = [true_data] * len(predict_data)
    for epoch in range(len(predict_data)):
        dot = np.einsum('ij,ij->i', predict_data[epoch][:,packet_id,:], true_data[epoch][:,packet_id,:])
        vnorm = np.linalg.norm(predict_data[epoch][:,packet_id,:],axis=1)*np.linalg.norm(true_data[epoch][:,packet_id,:], axis=1)
        cos_sim = dot/vnorm
        
        # Append mean and median across all traffic
        mean_cos_sim = np.mean(cos_sim, axis=0)
        median_cos_sim = np.median(cos_sim, axis=0)
        cos_sim_firstpacket_epoch.append((mean_cos_sim, median_cos_sim))

    return (cos_sim, cos_sim_firstpacket_epoch)

def cos_sim_traffic(predict_data, true_data, first=None):
    """
    Calculates the cosine similarity (CS) for traffic. It first calculates the mean/median CS for each traffic
    then calculates the mean/median CS across traffic. User can specify the first ___ number of packets as arg
    
    Returns a 3-tuple: 
    (mean cos sim for final prediction, median cos sim for final prediction, list of tuple (mean, median)) 
    where len(list) corresponds to #epoch
    """
    cos_sim_epoch = []
    true_data = [true_data] * len(predict_data)
    for epoch in range(len(predict_data)):
        
        dot = np.einsum('ijk,ijk->ij', predict_data[epoch], true_data[epoch])
        vnorm = (np.linalg.norm(predict_data[epoch],axis=2)*np.linalg.norm(true_data[epoch],axis=2))
        cos_sim = np.divide(dot,vnorm,out=np.zeros_like(dot), where=vnorm!=0.0)
        if first:
            cos_sim = cos_sim[:,0:first]
        
        # Verify that dot product is calculated correctly
        #print(np.dot(predict_data[epoch][54,10,:],true_data[epoch][54,10,:]))
        #print(np.einsum('ijk,ijk->ij', predict_data[epoch], true_data[epoch])[54,10])
        # Verify that norm is calculated correctly
        #print(np.linalg.norm(predict_data[epoch][0,0,:])*np.linalg.norm(true_data[epoch][0,0,:]))
        #print((np.linalg.norm(predict_data[epoch],axis=2)*np.linalg.norm(true_data[epoch],axis=2))[0,0])
        # Verify that einsendot is calcaluted correctly
        #print(np.dot(predict_data[epoch][0,0,:],true_data[epoch][0,0,:])/(np.linalg.norm(predict_data[epoch][0,0,:])*np.linalg.norm(true_data[epoch][0,0,:])))
        #print(cos_sim[0,0])

        mean_cos_sim_traffic = np.mean(cos_sim, axis=1)
        median_cos_sim_traffic = np.median(cos_sim, axis=1)
        
        overall_mean = np.mean(mean_cos_sim_traffic)
        overall_median = np.median(median_cos_sim_traffic)

        cos_sim_epoch.append((overall_mean, overall_median))
    return (mean_cos_sim_traffic, median_cos_sim_traffic, cos_sim_epoch)

def cos_sim_truetraffic(predict_data, true_data, seq_len):
    """
    Calculates the cosine similarity (CS) for true traffic (non-padded packets). It is similar to cos_sim_traffic()
    Information on actual sequence length of each traffic must be known through seq_len. The length of seq_len
    should be the same as true_data

    Returns a 3-tuple:
    (mean cos sim for final prediction, median cos sim for final prediction, list of tuple (mean, median)) 
    where len(list) corresponds to #epoch
    """

    cos_sim_epoch = []
    true_data = [true_data] * len(predict_data)
    for epoch in range(len(predict_data)):
        dot = np.einsum('ijk,ijk->ij', predict_data[epoch], true_data[epoch])
        vnorm = (np.linalg.norm(predict_data[epoch],axis=2)*np.linalg.norm(true_data[epoch],axis=2))
        cos_sim = np.divide(dot,vnorm,out=np.zeros_like(dot), where=vnorm!=0.0)

        # iterate through traffic
        mean_cos_sim_traffic = []
        median_cos_sim_traffic = []
        for i in range(cos_sim.shape[0]):
            mean_cos_sim_traffic.append(np.mean(cos_sim[i,0:seq_len[i]]))
            median_cos_sim_traffic.append(np.median(cos_sim[i,0:seq_len[i]]))
        mean_cos_sim_traffic = np.asarray(mean_cos_sim_traffic)
        median_cos_sim_traffic = np.asarray(median_cos_sim_traffic)

        overall_mean = np.mean(mean_cos_sim_traffic)
        overall_median = np.median(median_cos_sim_traffic)
        cos_sim_epoch.append((overall_mean, overall_median))
    return (mean_cos_sim_traffic, median_cos_sim_traffic, cos_sim_epoch)

def generate_plot(results_train, results_test, first, save, show=False):
    """
    Accepts a train and test object, which are generated from the function cos_sim_traffic

    A function to generate the following plots for first __ number of packets:
    • Accuracy plot for n epoch of training
    • Distribution of cosine similarity for final prediction

    If save is not False, it must be a string specifying the location to save the image. Else, the plot will be showed
    """
    plt.subplots_adjust(hspace=0.7)

    plt.subplot(311)
    mean_history_train = [i[0] for i in results_train[-1]]
    median_history_train = [i[1] for i in results_train[-1]]
    mean_history_test = [i[0] for i in results_test[-1]]
    median_history_test = [i[1] for i in results_test[-1]]
    x_values = [i for i in range(0, EPOCH, SAVE_EVERY_EPOCH)]
    plt.plot(x_values, mean_history_train, alpha=0.7)
    plt.plot(x_values, median_history_train, alpha=0.7)
    plt.plot(x_values, mean_history_test, alpha=0.7)
    plt.plot(x_values, median_history_test, alpha=0.7)
    plt.title('Model cosine similarity for first {} pkts'.format(first))
    plt.ylabel('Cosine Similarity')
    plt.xlabel('Epoch')
    plt.legend(['Train(mean)', 'Train(median)' , 'Val(mean)', 'Val(median)'], loc='upper left')

    plt.subplot(312)
    meantraffic_train = results_train[0]
    plt.plot(meantraffic_train,'|')
    plt.title('Dist of mean cosine similarity for first {} pkts (train)'.format(first))
    plt.ylabel('Mean Cosine Similarity')
    plt.xlabel('Traffic #')

    plt.subplot(313)
    meantraffic_test = results_test[0]
    plt.plot(meantraffic_test,'|')
    plt.title('Dist of mean cosine similarity for first {} pkts (validation)'.format(first))
    plt.ylabel('Mean Cosine Similarity')
    plt.xlabel('Traffic #')

    acc_dist = os.path.join(save, 'acc_dist')
    if not os.path.exists(acc_dist):
        os.mkdir(acc_dist)
    plt.savefig(os.path.join(acc_dist,'acc_dist_{}pkts').format(first))
    if show:
        plt.show()
    plt.clf()

##########################################################################################

# MODEL EVALUATION

##########################################################################################

trained_rnn = os.path.join(args.traffic, 'trained_rnn')
trained_rnn_expt = os.path.join(trained_rnn, 'expt_{}'.format(DATETIME_NOW))
trained_rnn_results = os.path.join(trained_rnn_expt, 'results')
trained_rnn_model = os.path.join(trained_rnn_expt, 'model')

# Create a new directory 'trained_rnn' to store model and results for training rnn on this traffic
if not os.path.exists(trained_rnn):
    os.mkdir(trained_rnn)

# Create directories for current experiment
os.mkdir(trained_rnn_expt)
os.mkdir(trained_rnn_results)
os.mkdir(trained_rnn_model)

plt.rcParams['figure.figsize'] = (10,7)
plt.rcParams['legend.fontsize'] = 8

# Visualize the model prediction on a specified dimension (default:packet length) over epochs
viz.visualize_traffic(predictEpoch.predict_train, Y_train, predictEpoch.predict_test, Y_test, save_every_epoch=SAVE_EVERY_EPOCH, save=trained_rnn_results, show=args.show)

# Generate result plots for true traffic
acc_pkttrue_train = cos_sim_truetraffic(predictEpoch.predict_train, Y_train, Y_train_seqlen)
acc_pkttrue_test = cos_sim_truetraffic(predictEpoch.predict_test, Y_test, Y_test_seqlen)
print('Final cosine similarity for true traffic on train data')
print(acc_pkttrue_train[-1][-1])
print('Final cosine similarity for true traffic on test data')
print(acc_pkttrue_test[-1][-1])
generate_plot(acc_pkttrue_train, acc_pkttrue_test, first='True', save=trained_rnn_results, show=args.show)

# Generate result plots for first packet
acc_pkt1_train = cos_sim_onepacket(predictEpoch.predict_train, Y_train, packet_id=0)
acc_pkt1_test = cos_sim_onepacket(predictEpoch.predict_test, Y_test, packet_id=0)
print('Final cosine similarity of first packet on train data')
print(acc_pkt1_train[-1][-1])
print('Final cosine similarity of first packet on test data')
print(acc_pkt1_test[-1][-1])
generate_plot(acc_pkt1_train, acc_pkt1_test, first=1, save=trained_rnn_results, show=args.show)

# Generate result plots for first 10,20,...,90,100 packets
for pktlen in range(10,101,10):
    acc_pkt_train = cos_sim_traffic(predictEpoch.predict_train, Y_train, first=pktlen)
    acc_pkt_test = cos_sim_traffic(predictEpoch.predict_test, Y_test, first=pktlen)
    print('Final cosine similarity for the first {} packets on train data'.format(pktlen))
    print(acc_pkt_train[-1][-1])
    print('Final cosine similarity for the first {} packets on test data'.format(pktlen))
    print(acc_pkt_test[-1][-1])
    generate_plot(acc_pkt_train, acc_pkt_test, first=pktlen, save=trained_rnn_results, show=args.show)

# Generate plots for training & validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

plt.savefig(os.path.join(trained_rnn_results,'loss'))
if args.show:
    plt.show()
plt.clf()

# Save the model
model.save(os.path.join(trained_rnn_model,'rnnmodel_{}.h5'.format(DATETIME_NOW)))

# Save model config information
train_info = os.path.join(trained_rnn_model, 'train_info_{}.txt'.format(DATETIME_NOW))
with open(train_info, 'w') as f:
    # Datetime
    f.write('####################################\n\n')
    f.write('Training Date: {}\n'.format(DATETIME_NOW.split('_')[0]))
    f.write('Training Time: {}\n'.format(DATETIME_NOW.split('_')[1]))
    f.write('Batch Size: {}\n'.format(BATCH_SIZE))
    f.write('Epoch: {}\n'.format(EPOCH))
    f.write('Feature file used: {}\n'.format(feature_file))
    f.write("Existing model used: {}\n".format(existing_model))
    f.write("Split Ratio: {}\n".format(SPLIT_RATIO))
    f.write("Seed: {}\n\n".format(SEED))
    f.write('####################################')

##########################################################################################

# DIAGNOSTIC TESTS

##########################################################################################

snapshot = tracemalloc.take_snapshot()
# diag.display_top(snapshot)
# Pick the top 3 biggest memory blocks 
top_stats = snapshot.statistics('traceback')
for i in range(0,5):
    stat = top_stats[i]
    print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
    for line in stat.traceback.format():
        print(line)

