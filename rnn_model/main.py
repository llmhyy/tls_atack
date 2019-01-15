import json
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

import parse_features as pf
import visualization as viz

##########################################################################################

# DATA LOADING AND PREPROCESSING

##########################################################################################

# Load the dataset into memory 
features = json.loads(pf.get_features('data/features.csv'))

# Initialize useful constants and pad the sequences to have equal length
batch_dim = len(features)
sequence_dim = max(list(map(len, features)))
feature_dim = features[0][0]
features_pad = pad_sequences(features, maxlen=sequence_dim, dtype='float32', padding='post',value=0.0)
#print(features_pad.shape)  # shape (3496,100,19)

# Feature scaling
# 1: Normalize each sample independently into unit vectors
# 2: Standardize each feature across ALL traffic (http://cs231n.github.io/neural-networks-2/)
# Shoud I standardize within 1 traffic only? Hmm wouldnt the padding affect the standardization as well
# 3: Scale each feature between min and max of feature

scale_option = 3

if scale_option == 1:
    l2_norm = np.linalg.norm(features_pad, axis=2, keepdims=True)
    features_scaled = np.divide(features_pad, l2_norm, out=np.zeros_like(features_pad), where=l2_norm!=0.0)

elif scale_option == 2:
    flattened = features_pad.reshape((features_pad.shape[0]*features_pad.shape[1],features_pad.shape[2]))
    zero_centered = features_pad - np.mean(flattened, axis=0, keepdims=True)
    std_dev = np.std(flattened, axis=0, keepdims=True)
    features_scaled = np.divide(zero_centered, std_dev, out=np.zeros_like(features_pad), where=std_dev!=0.0) 

elif scale_option == 3:
    features_max = np.amax(features_pad, axis=(0,1))
    features_min = np.amin(features_pad, axis=(0,1))
    num = (features_pad-features_min)
    dem = (features_max-features_min)
    features_scaled = np.divide(num,dem,out=np.zeros_like(num), where=dem!=0.0)
    # Verify that scaling is calculated correctly
    #print(features_pad[200,4,:])
    #print(features_min)
    #print(features_max)
    #print(features_scaled[200,4,:])

# Generate training input and output by lagging 1 timestep
zero_features = np.zeros((features_scaled.shape[0],1,features_scaled.shape[2]))
features_zero_appended = np.concatenate((zero_features, features_scaled), axis=1)
X = features_zero_appended[:,:-1,:]
Y = features_zero_appended[:,1:,:]
# print(X.shape)     # shape (3496,100,19)
# print(Y.shape)     # shape (3496,100,19)

# Train test split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2019)

##########################################################################################

# MODEL BUILDING

##########################################################################################

# Build RNN model
model = Sequential()
model.add(LSTM(X.shape[2], input_shape=(X.shape[1],X.shape[2]), return_sequences=True))
model.add(Activation('relu'))
model.summary()
#model.add(Dropout(0.2))

# Defining cosine similarity
def cos_sim(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    # Flattening features in a time series from (n_sample, time seq, features) into (n_sample, n_features)
    #print(K.shape(y_true))
    #y_true = K.reshape(y_true, K.cast((y_true.shape[0],y_true.shape[1]*y_true.shape[2]), dtype='int32'))
    y_true = K.reshape(y_true, K.cast((K.shape(y_true)[0],K.shape(y_true)[1]*K.shape(y_true)[2]), dtype='int32'))
    #y_pred = K.reshape(y_pred, K.cast((y_pred.shape[0],y_pred.shape[1]*y_pred.shape[2]), dtype='int32'))
    y_pred = K.reshape(y_pred, K.cast((K.shape(y_pred)[0],K.shape(y_pred)[1]*K.shape(y_pred)[2]), dtype='int32'))
    return y_true*y_pred

# Selecting optimizers 
model.compile(loss='mean_squared_error',
                optimizer='rmsprop')

class PredictEpoch(keras.callbacks.Callback):
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def on_train_begin(self, logs={}):
        self.predict_train = []
        self.predict_test = []

    def on_epoch_end(self, epoch, logs={}):
        self.predict_train.append(self.model.predict(self.train))
        self.predict_test.append(self.model.predict(self.test))

# Training the RNN model
predictEpoch = PredictEpoch(X_train, X_test)
history = model.fit(X_train,Y_train, validation_data = (X_test, Y_test),batch_size=64, epochs=10, callbacks=[predictEpoch])

# Visualize the model prediction over epochs
#viz.visualize_traffic(predictEpoch.predict_train, Y_train, predictEpoch.predict_test, Y_test)

# Evaluate the RNN model on validation dataset
#score = model.evaluate(X_test, Y_test, batch_size=64)
#print("Mean Squared Error: {}   ||   Cosine Similarity: {}".format(score[0], score[1]))

# Predict with RNN model for a randomly chosen sample from the test dataset
sample = X_test[[3]]
Y_actual = Y_test[[3]]
Y_predict = model.predict(sample)
#np.set_printoptions(threshold=np.inf)
print('Actual: ')
print(Y_actual[0])
print('Actual shape: {}'.format(Y_actual[0].shape))
print('Predicted: ')
print(Y_predict[0])
print('Predicted shape: {}'.format(Y_predict[0].shape))
Y_actual = Y_actual.reshape(1, Y_actual.shape[1]*Y_actual.shape[2])
Y_predict = Y_predict.reshape(1, Y_predict.shape[1]*Y_predict.shape[2])
sample_score = cosine_similarity(Y_actual, Y_predict)
print('Cosine similarity for sample: {}'.format(sample_score[0]))

# Calculate cosine similarity for first packet and across traffic
def cos_sim_firstpacket(predict_data, true_data):
    """
    Calculates the cosine similarity for the first packet of every traffic and return a 
    list of tuple (mean, median) where len(list) corresponds to #epoch
    """
    cos_sim_firstpacket_epoch = []
    true_data = [true_data] * len(predict_data)
    for epoch in range(len(predict_data)):
        #print(predict_data[epoch][1,0,:])
        #print(true_data[epoch][1,0,:])
        #print(np.dot(predict_data[epoch][1,0,:], true_data[epoch][1,0,:]))
        #print (np.tensordot(predict_data[epoch][:,0,:], true_data[epoch][:,0,:],axes=(1,1)).diagonal()[0])
        #print(np.dot(predict_data[epoch][1,0,:], true_data[epoch][1,0,:])/(np.linalg.norm(predict_data[epoch][1,0,:]) * np.linalg.norm(true_data[epoch][1,0,:])))
        #print(np.tensordot(predict_data[epoch][:,0,:], true_data[epoch][:,0,:],axes=(1,1)).diagonal()[0]/(np.linalg.norm(predict_data[epoch][1,0,:],axis=0)*np.linalg.norm(true_data[epoch][1,0,:], axis=0)))
        cos_sim = np.tensordot(predict_data[epoch][:,0,:], true_data[epoch][:,0,:],axes=(1,1)).diagonal()/(np.linalg.norm(predict_data[epoch][:,0,:],axis=1)*np.linalg.norm(true_data[epoch][:,0,:], axis=1))
        # Append mean and median across all traffic
        mean_cos_sim = np.mean(cos_sim, axis=0)
        median_cos_sim = np.median(cos_sim, axis=0)
        cos_sim_firstpacket_epoch.append((mean_cos_sim, median_cos_sim))

    return cos_sim_firstpacket_epoch

def cos_sim_tenpackets(predict_data, true_data):
    cos_sim_tenpackets_epoch = []
    true_data = [true_data] * len(predict_data)
    for epoch in range(len(predict_data)):
        dot = np.einsum('ijk,ijk->ij', predict_data[epoch], true_data[epoch])
        vnorm = (np.linalg.norm(predict_data[epoch],axis=2)*np.linalg.norm(true_data[epoch],axis=2))
        cos_sim = np.divide(dot,vnorm,out=np.zeros_like(dot), where=vnorm!=0.0)
        #print(cos_sim.shape)
        cos_sim_10 = cos_sim[:,0:10]
        mean_cos_sim_10 = np.mean(cos_sim_10, axis=1)
        median_cos_sim_10 = np.median(cos_sim_10, axis=1)
        ten_mean = np.mean(mean_cos_sim_10)
        ten_median = np.median(median_cos_sim_10)
        cos_sim_tenpackets_epoch.append((ten_mean, ten_median))
    return (mean_cos_sim_10, median_cos_sim_10, cos_sim_tenpackets_epoch)

def cos_sim_traffic(predict_data, true_data):
    """
    Calculates the cosine similarity for the whole traffic. Returns a 3-tuple: 
    (mean cos sim for final prediction, median cos sim for final prediction, list of tuple (mean, median)) 
    where len(list) corresponds to #epoch
    """
    cos_sim_epoch = []
    true_data = [true_data] * len(predict_data)
    for epoch in range(len(predict_data)):
        # get mean/median cos sim within a traffic
        #print(predict_data[epoch].shape)
        #print(true_data[epoch].shape)
        #print(np.einsum('ijk,ijk->ij', predict_data[epoch], true_data[epoch]).shape)
        #print((np.linalg.norm(predict_data[epoch],axis=2)*np.linalg.norm(true_data[epoch],axis=2)).shape)
        
        dot = np.einsum('ijk,ijk->ij', predict_data[epoch], true_data[epoch])
        vnorm = (np.linalg.norm(predict_data[epoch],axis=2)*np.linalg.norm(true_data[epoch],axis=2))
        cos_sim = np.divide(dot,vnorm,out=np.zeros_like(dot), where=vnorm!=0.0)
        
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

print('Final cosine similarity of first packet on train dataset')
acc_pkt1_train = cos_sim_firstpacket(predictEpoch.predict_train, Y_train)
print(acc_pkt1_train[-1])

print('Final cosine similarity of first packet on test dataset')
acc_pkt1_test = cos_sim_firstpacket(predictEpoch.predict_test, Y_test)
print(acc_pkt1_test[-1])

print('Final cosine similarity of traffic on train dataset')
acc_pkttraffic_train = cos_sim_traffic(predictEpoch.predict_train, Y_train)
print(acc_pkttraffic_train[-1][-1])

print('Final cosine similarity of traffic on test dataset')
acc_pkttraffic_test = cos_sim_traffic(predictEpoch.predict_test, Y_test)
print(acc_pkttraffic_test[-1][-1])

print('Cosine similarity for the first 10 packets on train dataset')
acc_pkt10_train = cos_sim_tenpackets(predictEpoch.predict_train, Y_train)
print(acc_pkt10_train[-1])

print('Cosine similarity for the first 10 packets on test dataset')
acc_pkt10_test = cos_sim_tenpackets(predictEpoch.predict_test, Y_test)
print(acc_pkt10_test[-1])

# Plot training & validation cosine similarity for first packet
plt.subplots_adjust(hspace=0.8)
#plt.legend(loc=2, prop={'size': 2})

plt.subplot(411)
#plt.plot(history.history['cos_sim'])
#plt.plot(history.history['val_cos_sim'])
mean_history_train = [i[0] for i in acc_pkt1_train]
median_history_train = [i[1] for i in acc_pkt1_train]
mean_history_test = [i[0] for i in acc_pkt1_test]
median_history_test = [i[1] for i in acc_pkt1_test]
#print(mean_history_train)
#print(median_history_train)
#print(mean_history_test)
#print(median_history_test)
plt.plot(mean_history_train, alpha=0.7)
plt.plot(median_history_train, alpha=0.7)
plt.plot(mean_history_test, alpha=0.7)
plt.plot(median_history_test, alpha=0.7)
plt.title('Model cosine similarity on 1st packet')
plt.ylabel('Cosine similarity')
plt.xlabel('Epoch')
plt.legend(['Train(mean)', 'Train(median)' , 'Val(mean)', 'Val(median)'], loc='upper left')

# Plot training % validation cosine similarity for first 10 packets
plt.subplot(412)
meanten_history_train = [i[0] for i in acc_pkt10_train[-1]]
medianten_history_train = [i[1] for i in acc_pkt10_train[-1]]
meanten_history_test = [i[0] for i in acc_pkt10_test[-1]]
medianten_history_test = [i[1] for i in acc_pkt10_test[-1]]
plt.plot(meanten_history_train, alpha=0.7)
plt.plot(medianten_history_train, alpha=0.7)
plt.plot(meanten_history_test, alpha=0.7)
plt.plot(medianten_history_test, alpha=0.7)
plt.title('Model cosine similarity for first 10 packets')
plt.ylabel('Cosine similarity')
plt.xlabel('Epoch')
plt.legend(['Train(mean)', 'Train(median)' , 'Val(mean)', 'Val(median)'], loc='upper left')


# Plot training & validation cosine similarity for overall traffic
plt.subplot(413)
meantraffic_history_train = [i[0] for i in acc_pkttraffic_train[-1]]
mediantraffic_history_train = [i[1] for i in acc_pkttraffic_train[-1]]
meantraffic_history_test = [i[0] for i in acc_pkttraffic_test[-1]]
mediantraffic_history_test = [i[1] for i in acc_pkttraffic_test[-1]]
plt.plot(meantraffic_history_train, alpha=0.7)
plt.plot(mediantraffic_history_train, alpha=0.7)
plt.plot(meantraffic_history_test, alpha=0.7)
plt.plot(mediantraffic_history_test, alpha=0.7)
plt.title('Model cosine similarity on overall traffic')
plt.ylabel('Cosine similarity')
plt.xlabel('Epoch')
plt.legend(['Train(mean)', 'Train(median)' , 'Val(mean)', 'Val(median)'], loc='upper left')

# Plot training & validation loss
plt.subplot(414)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

plt.show()

# Plot distribution of mean osine similarity for final prediction on overall traffic
plt.subplot(221)
meantraffic_train = acc_pkttraffic_train[0]
plt.plot(meantraffic_train,'|')
plt.title('Distribution of mean cosine similarity across train traffic')
plt.ylabel('Mean Cosine Similarity')
plt.xlabel('Taffic #')

plt.subplot(223)
meantraffic_test = acc_pkttraffic_test[0]
plt.plot(meantraffic_test,'|')
plt.title('Distribution of mean cosine similarity across validation traffic')
plt.ylabel('Mean Cosine Similarity')
plt.xlabel('Taffic #')

# Plot distribution of mean cosine similarity for final prediction on first 10 packets
plt.subplot(222)
mean10pkt_train = acc_pkt10_train[0]
plt.plot(mean10pkt_train, '|')
plt.title('Distribution of mean cosine similarity across first 10 packets of train traffic')
plt.ylabel('Mean Cosine Similarity')
plt.xlabel('Traffic #')

plt.subplot(224)
mean10pkt_test = acc_pkt10_test[0]
plt.plot(mean10pkt_test, '|')
plt.title('Distribution of mean cosine similarity across first 10 packets of test traffic')
plt.ylabel('Mean Cosine Similarity')
plt.xlabel('Traffic #')

plt.show()

