import os
import logging
import numpy as np

import utils 

logging.basicConfig(filename='output.log', level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')

pcap_dir = '/Users/YiLong/Desktop/SUTD/NUS-Singtel_Research/tls_atack/legitimate traffic'
features_csv = 'features_tls.csv'

# Iterate through pcap files and identify all enums
enums = utils.searchEnums(pcap_dir, limit=100)

# Test whether the enums are searched and generated correctly
# for enum in enums:
#   print(enum)
#   print('Length: {}'.format(len(enum)))

with open(features_csv, 'a', newline='') as csv:
    count = 0
    # Iterate through pcap files and extract features
    for root, dirs, files in os.walk(pcap_dir):
        for f in files:
            if f.endswith(".pcap"):
                print("Extracting features from {}".format(f))
                logging.info("Extracting features from {}".format(f))
                # Generate TCP features
                tcp_features = utils.extract_tcp_features(os.path.join(root, f))
                # Generate TLS/SSL features
                tls_features = utils.extract_tslssl_features(os.path.join(root, f), enums[0], enums[1], enums[2], enums[3], enums[4])
                
                # print('TCP')
                # print(tcp_features)
                # print('TLS')
                # print(tls_features)

                # Combine TCP and TLS/SSL features
                traffic_features = (np.concatenate((np.array(tcp_features), np.array(tls_features)), axis=1)).tolist()
                # Write into csv file
                for traffic_feature in traffic_features:
                    csv.write(str(traffic_feature)+', ')
                csv.write('\n')
                count+=1

                if count>=10:
                    break
        if count>=10:
            break

print("{} pcap files have been successfully parsed with features generated".format(count))