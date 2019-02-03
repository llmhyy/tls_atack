import os
import logging
import argparse
import numpy as np
from datetime import datetime

import utils 

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--traffic', help='Input top-level directory of the traffic module containing pcap files', required=True)
args = parser.parse_args()

logging.basicConfig(filename='output.log', level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')

pcap_dir = args.traffic 
extracted_features = os.path.join(pcap_dir, 'extracted_features')
# Create a new directory 'extracted_features' to store extracted features
if not os.path.exists(extracted_features):
    os.mkdir(extracted_features)
datetime_now = datetime.now()
# File for storing extracted features
features_csv = os.path.join(extracted_features,'features_tls_{}.csv'.format(datetime_now.strftime('%Y-%m-%d_%H-%M-%S')))
# File for storing information about enums used
features_info = os.path.join(extracted_features, 'features_info_{}.txt'.format(datetime_now.strftime('%Y-%m-%d_%H-%M-%S')))

def search_and_extract(pcap_dir, features_csv, enums):
    success = 0
    failed = 0
    with open(features_csv, 'a', newline='') as csv:
        for root, dirs, files in os.walk(pcap_dir):
            for f in files:
                if f.endswith(".pcap"):
                    try:
                        #print("Extracting features from {}".format(f))
                        logging.info("Extracting features from {}".format(f))
                        # Generate TCP features
                        tcp_features = utils.extract_tcp_features(os.path.join(root, f))
                        # Generate TLS/SSL features
                        tls_features = utils.extract_tslssl_features(os.path.join(root, f), enums)
                        # Combine TCP and TLS/SSL features
                        traffic_features = (np.concatenate((np.array(tcp_features), np.array(tls_features)), axis=1)).tolist()
                        # Each packet in traffic features is a vector of 139 dimension

                        # Write into csv file
                        for traffic_feature in traffic_features:
                            csv.write(str(traffic_feature)+', ')
                        csv.write('\n')
                        success+=1
                        if success%1000==0:
                            print('{} pcap files has been parsed...'.format(success))

                    # Skip this pcap file 
                    except (KeyError, AttributeError):
                        logging.exception('Serious error in file {}. Traffic is skipped'.format(f))
                        failed+=1
                        continue

    print("{} pcap files have been successfully parsed from {} with features generated. {} pcap files have failed".format(success, pcap_dir, failed))


# Iterate through pcap files and identify all enums
if 'new_traffic' in pcap_dir:
    # Traffic contains both TLS and SSLv3 traffic and both have wide differences in enums
    enums_tls = utils.searchEnums(os.path.join(pcap_dir, 'output_TLS'), limit=1000)
    enums_sslv3 = utils.searchEnums(os.path.join(pcap_dir, 'outsslv3'), limit=1000)
    enums = {k:list(set(v+enums_sslv3[k])) for k,v in enums_tls.items()}
elif 'legitimate traffic' in pcap_dir:
    # Traffic contains both TLS and SSLv3 traffic and both have wide differences in enums
    enums_tls = utils.searchEnums(os.path.join(pcap_dir, 'output TLS'), limit=3)
    enums_sslv3 = utils.searchEnums(os.path.join(pcap_dir, 'output_SSLv3'), limit=3)
    enums = {k:list(set(v+enums_sslv3[k])) for k,v in enums_tls.items()}
else:
    enums = utils.searchEnums(pcap_dir, limit=2000)

with open(features_info, 'w') as f:
    for k,v in enums.items():
        print('Enum: {}'.format(k))
        f.write('Enum: {}\n'.format(k))
        print(v)
        f.write(str(v)+'\n')
        print('Length of enum: {}'.format(len(v)))
        f.write('Length of enum: {}\n\n'.format(len(v)))

# Iterate through pcap files and extract features
search_and_extract(pcap_dir, features_csv, enums)

