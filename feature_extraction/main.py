import os
import logging
import numpy as np
from datetime import datetime

import utils 

logging.basicConfig(filename='output.log', level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')

pcap_dir = '/Users/YiLong/Desktop/SUTD/NUS-Singtel_Research/tls_atack/legitimate traffic/'
pcap_tls_dir = '/Users/YiLong/Desktop/SUTD/NUS-Singtel_Research/tls_atack/legitimate traffic/output TLS/'
pcap_sslv3_dir = '/Users/YiLong/Desktop/SUTD/NUS-Singtel_Research/tls_atack/legitimate traffic/output_SSLv3/'
features_csv = 'features_tls_{}.csv'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

def search_and_extract(pcap_dir, features_csv, enums):
    with open(features_csv, 'a', newline='') as csv:
        count = 0
        for root, dirs, files in os.walk(pcap_dir):
            for f in files:
                if f.endswith(".pcap"):
                    #print("Extracting features from {}".format(f))
                    logging.info("Extracting features from {}".format(f))
                    # Generate TCP features
                    tcp_features = utils.extract_tcp_features(os.path.join(root, f))
                    # Generate TLS/SSL features
                    tls_features = utils.extract_tslssl_features(os.path.join(root, f), enums[0], enums[1], enums[2], enums[3], enums[4])
                    # Combine TCP and TLS/SSL features
                    traffic_features = (np.concatenate((np.array(tcp_features), np.array(tls_features)), axis=1)).tolist()
                    # Each packet in traffic features is a vector of 137 dimension

                    # Write into csv file
                    for traffic_feature in traffic_features:
                        csv.write(str(traffic_feature)+', ')
                    csv.write('\n')
                    count+=1
    print("{} pcap files have been successfully parsed from {} with features generated".format(count, pcap_dir))

# Iterate through pcap files and identify all enums
enums_tls = utils.searchEnums(pcap_tls_dir, limit=100)
enum_sslv3 = utils.searchEnums(pcap_sslv3_dir, limit=100)
enums = tuple(list(set(i[0]+i[1])) for i in zip(enums_tls, enum_sslv3))

# Test whether the enums are searched and generated correctly
# for enum in enums:
#   print(enum)
#   print('Length: {}'.format(len(enum)))

# Iterate through pcap files and extract features
search_and_extract(pcap_tls_dir, features_csv, enums)
search_and_extract(pcap_sslv3_dir, features_csv, enums)

