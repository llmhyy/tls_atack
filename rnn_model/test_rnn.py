import os
import argparse
import numpy as np
from keras.models import load_model

from utils_datagen import get_mmapdata_and_byteoffset
from utils_datagen import get_min_max
from utils_datagen import normalize
from utils_datagen import BatchGenerator
from utils_plot import plot_distribution
import utils_metric as utilsMetric

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--traffic', help='Input top-level directory of the traffic module containing extracted features', required=True)
parser.add_argument('-f', '--feature', help='Input directory path of feature file to be used', required=True)
parser.add_argument('-m', '--model', help='Input directory for existing model to be used for testing', required=True)
args = parser.parse_args()

BATCH_SIZE = 64
SEQUENCE_LEN = 100

target_traffic_name = os.path.split(args.traffic.strip('/'))[-1]
savedir = os.path.join(os.path.split(args.model)[0],'..','test_results',target_traffic_name)
if not os.path.exists(savedir):
    os.makedirs(savedir)

# Start logging into file
logfile = open(os.path.join(savedir, 'test_log.txt'),'w')

model = load_model(args.model)
model.summary()

# Search for extracted_features directory
extracted_features = os.path.join(args.traffic, 'extracted_features')
if not os.path.exists(extracted_features):
    raise FileNotFoundError("Directory extracted_features not found. Extract features first")

# Load the mmap data and the byte offsets from the feature file
print('\nLoading features into memory...')
mmap_data, byte_offset = get_mmapdata_and_byteoffset(args.feature)
# Get min and max for each feature
min_max_feature = get_min_max(mmap_data, byte_offset)
# Initialize the normalization function 
norm_fn = normalize(2, min_max_feature)

data_generator = BatchGenerator(mmap_data, byte_offset, BATCH_SIZE, SEQUENCE_LEN, norm_fn, return_seq_len=True)

# Obtain the mean accuracy for each traffic and store in a array
print('Computing mean accuracy for traffic...')
mean_acc_for_all_traffic = np.array([])
for (batch_inputs, batch_true, batch_seq_len) in data_generator:
    batch_predict = model.predict_on_batch(batch_inputs)
    batch_acc = utilsMetric.calculate_acc_of_traffic(batch_predict, batch_true)

    for i, seq_len in enumerate(batch_seq_len):
        acc_spliced = batch_acc[i:i+1,0:seq_len]
        mean_acc = utilsMetric.calculate_mean_acc_of_traffic(acc_spliced)
        mean_acc_for_all_traffic = np.concatenate((mean_acc_for_all_traffic, mean_acc))

# Compute the overall mean accuracy across all traffic
overall_mean_acc = np.mean(mean_acc_for_all_traffic, keepdims=True)[0]

# Plot the distribution of cosine similarity
plot_distribution(mean_acc_for_all_traffic, overall_mean_acc, os.path.splitext(os.path.basename(args.feature))[0], savedir)
print('#### Overall Mean Consine Similarity for {}:\t{}\n'.format(target_traffic_name, overall_mean_acc))

# Write into log file
logfile.write("#####  MEAN ACCURACY  #####\n")
logfile.write('Overall Mean Accuracy for {}:\t{:.6f}\n'.format(target_traffic_name, overall_mean_acc))

logfile.close()