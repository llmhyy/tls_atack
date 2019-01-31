import os
import logging
# import argparse
import numpy as np
from datetime import datetime

import utils 

# parser = argparse.ArgumentParser()
# parser.add_argument('-p', '--pcap', help='Input directory where pcap files are stored', nargs='+', required=True)
# parser.add_argument('-s', '--save', help='Input directory to save the csv file', required=True)
# args = parser.parse_args()

logging.basicConfig(filename='output.log', level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')

pcap_dir = '/Users/YiLong/Desktop/SUTD/NUS-Singtel_Research/tls_atack/legitimate traffic/'
# pcap_tls_dir = '/Users/YiLong/Desktop/SUTD/NUS-Singtel_Research/tls_atack/legitimate traffic/output TLS/'
pcap_tls_dir = '/Users/YiLong/Desktop/SUTD/NUS-Singtel_Research/tls_atack/new_traffic/output_TLS/'
pcap_sslv3_dir = '/Users/YiLong/Desktop/SUTD/NUS-Singtel_Research/tls_atack/legitimate traffic/output_SSLv3/'
features_csv = 'features_tls_{}.csv'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
# features_csv = os.path.join(args.save, 'features_tls_{}.csv'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

def search_and_extract(pcap_dir, features_csv, enums):
    file_count = 0
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
                        file_count+=1
                        if file_count%1000==0:
                            print('{} pcap files has been parsed...'.format(file_count))

                    # Skip this pcap file 
                    except (KeyError, AttributeError):
                        logging.exception('Serious error in file {}. Traffic is skipped'.format(f))
                        continue

    print("{} pcap files have been successfully parsed from {} with features generated".format(file_count, pcap_dir))

# Iterate through pcap files and identify all enums
enums_tls = utils.searchEnums(pcap_tls_dir, limit=1000)
enum_sslv3 = utils.searchEnums(pcap_sslv3_dir, limit=1000)
# enums = tuple(list(set(i[0]+i[1])) for i in zip(enums_tls, enum_sslv3))
enums = {k:list(set(v+enum_sslv3[k])) for k,v in enums_tls.items()}
for k,v in enums.items():
    print('Enum: {}'.format(k))
    print(v)
    print('Length of enum: {}'.format(len(v)))

# Iterate through pcap files and extract features
search_and_extract(pcap_tls_dir, features_csv, enums)
search_and_extract(pcap_sslv3_dir, features_csv, enums)

