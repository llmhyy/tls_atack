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
parser.add_argument('-i', '--info', help='Input directory path of feature info to be used', required=True)
parser.add_argument('-m', '--model', help='Input directory for existing model to be used for testing', required=True)
args = parser.parse_args()

BATCH_SIZE = 64
SEQUENCE_LEN = 100

target_traffic_name = os.path.split(args.traffic.strip('/'))[-1]
savedir = os.path.join(os.path.split(args.model)[0],'..','test_results',target_traffic_name)
if not os.path.exists(savedir):
    os.makedirs(savedir)

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
# Initialize the batch generator
data_generator = BatchGenerator(mmap_data, byte_offset, BATCH_SIZE, SEQUENCE_LEN, norm_fn, return_seq_len=True)

print('Computing metrics for traffic...')
mean_acc_for_all_traffic = np.array([])
squared_error_for_features = None
num_traffic = 0
for (batch_inputs, batch_true, batch_seq_len) in data_generator:
    batch_predict = model.predict_on_batch(batch_inputs)
    
    # Compute mean accuracy for each traffic and store in a array
    batch_acc = utilsMetric.calculate_acc_of_traffic(batch_predict, batch_true)
    for i, seq_len in enumerate(batch_seq_len):
        acc_spliced = batch_acc[i:i+1,0:seq_len]
        mean_acc = utilsMetric.calculate_mean_acc_of_traffic(acc_spliced)
        mean_acc_for_all_traffic = np.concatenate((mean_acc_for_all_traffic, mean_acc))

    # Compute squared error for each dimension in batch of traffic
    squared_error_batch = utilsMetric.calculate_squared_error_of_traffic(batch_predict, batch_true)
    num_traffic += BATCH_SIZE
    if squared_error_for_features is not None:
        squared_error_for_features = squared_error_for_features + squared_error_batch
    else:
        squared_error_for_features = np.zeros_like(squared_error_batch) + squared_error_batch

# Compute the overall mean accuracy across all traffic
overall_mean_acc = np.mean(mean_acc_for_all_traffic, keepdims=True)[0]

# Compute the MSE for each feature
mean_squared_error_for_features = squared_error_for_features/num_traffic

# Plot the distribution of cosine similarity
plot_distribution(mean_acc_for_all_traffic, overall_mean_acc, os.path.splitext(os.path.basename(args.feature))[0], savedir)
# print('#### Overall Mean Consine Similarity for {}:\t{}\n'.format(target_traffic_name, overall_mean_acc))

# Write results into log file
with open(os.path.join(savedir, 'test_log.txt'),'w') as logfile:
    logfile.write("#####  MEAN ACCURACY  #####\n")
    logfile.write('Overall Mean Accuracy for {:60}:{:>10.6f}\n'.format(target_traffic_name, overall_mean_acc))
    logfile.write("\n#####  MEAN SQUARED ERROR FOR EACH DIMENSION  #####\n")
    with open(args.info, 'r') as f:
        features_info = f.readlines()[1:] # Ignore header
        for i,feature_mse in enumerate(mean_squared_error_for_features):
            split_features_info = features_info[i].split(',')
            network_layer, tls_protocol, feature_name, feature_type, feature_enum_value = split_features_info[0].strip(), split_features_info[1].strip(), split_features_info[2].strip(), split_features_info[3].strip(), split_features_info[4].strip()
            if 'Enum' in feature_type:
                feature_name = feature_name+'-'+feature_enum_value
            if 'TLS' in network_layer:
                feature_name = '('+tls_protocol+')'+feature_name
            line = 'Mean Squared Error for {:60}{:>10.6f}\n'.format(feature_name+':', feature_mse)
            logfile.write(line)
