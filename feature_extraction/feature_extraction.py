import os
import csv
import json
import time
# from scapy.all import *
import pyshark
from pyshark.packet.layer import JsonLayer
import logging
import numpy as np
import ipaddress
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(filename='output.log', level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')

# load_layer("TLS")
# load_layer("SSL")


def extract_tcp_features(pcapfile):
    """
    ** NEED EDITS **
    Extract features from a pcap file and returns a feature vector. The feature vector is a 
    n row x m column 2D matrix, where n is the number of packets and m is the number of features

    Parameters:
    pcapfile (file): pcap file to be parsed

    Returns:
    list: returns a list of packet tcp features. E.g. if there are 8 packets in the pcapfile, it will be a list
    of 8 vectors of 20 features
    
    Key Considerations:
    * The index of the feature is fixed in the feature vector. E.g. come/leave will always occupy the first
    column, hence it would not make sense to have a strategy pattern where we can augment/subtract features
    * Hence, by extension, the number of features is FIXED. If the feature does not exist, we just zero them
    * Need to consider tls record layer as the most basic unit of traffic if frame contains ssl layer 
    since each record layer has a different goal
    """

    # Traffic features for storing features of packets
    traffic_features = []

    # Protocol version value to encode into one-hot vector
    # prot: ['TCP' (-), 'SSL2.0' (0x0200), 'SSL3.0' (0x0300), 'TLS1.0' (0x0301), 'TLS1.1' (0x0302), 'TLS1.2' (0x0303)]
    protcol_ver = [0, 512, 768, 769, 770, 771]

    ####################################################################################
    ## USING SCAPY ##

    if False:
        packets = rdpcap(pcapfile)
        # print(len(packets))
        # print(packets[0])
        # print(packets[0].show())
        # print(dir(packets))
        # print(type(packets))

        for i, packet in enumerate(packets):
            print('Packet  #' + str(i + 1))
            packet.show()
            features = []

            # 1: Come/Leave
            if packet.haslayer('IP'):
                if ipaddress.ip_address(unicode(packet['IP'].dst)).is_private:
                    features.append(1)
                else:
                    features.append(0)
            else:
                print("Error: packet does not contain IP layer")
                exit()

            # 2: Protocol
            # print(packet.haslayer('SSL'))

            # 3: Length
            # 4: Interval
            # 5: Flag
            # 6: Window Size

            # print(features)

    ####################################################################################
    ## USING PYSHARK ##
    # Pyshark seems to be unable to distinguish between record layers

    packets = pyshark.FileCapture(pcapfile)
    for i, packet in enumerate(packets):

        # TCP FEATURES
        ##################################################################
        features = []

        # 1: COME/LEAVE
        if ipaddress.ip_address(str(packet.ip.dst)).is_private:
            features.append(1)
        else:
            features.append(0)

        # 2: PROTOCOL
        protocol_id = 0
        protocol_onehot = [0] * len(protcol_ver)
        # Checks for SSL layer in packet. Bug in detecting SSL layer despite plain TCP packet
        if ('ssl' in packet) and (packet.ssl.get('record_version') != None):
            # Convert hex into integer and return the index in the ref list
            protocol_id = protcol_ver.index(int(packet.ssl.record_version, 16))
        protocol_onehot[protocol_id] = 1
        features.extend(protocol_onehot)

        # 3: LENGTH
        features.append(int(packet.length))

        # 4: INTERVAL
        features.append(float(packet.frame_info.time_delta) * 1000)

        # 5: FLAG
        num_of_flags = 9
        # Convert hex into binary and pad left with 0 to fill 9 flags
        flags = list(bin(int(packet.tcp.flags, 16))[2:].zfill(num_of_flags))
        features.extend(list(map(int, flags)))

        # 6: WINDOW SIZE
        # Append the calculated window size (window size value * scaling factor)
        features.append(int(packet.tcp.window_size))

        traffic_features.append(features)

    ####################################################################################
    # Write into csvfile as a row
    # with open(csvfile, 'a', newline='') as f:
    #     for traffic_feature in traffic_features:
    #         f.write(str(traffic_feature)+', ')
    #     f.write('\n')

    return traffic_features

def searchCipherSuites(rootdir):
    """
    Given a root directory containing all the pcap files, it will search for all possible cipher suites
    and return a list of all cipher suites
    """
    ciphersuites = []
    compressionmethods = []
    supportedgroups = []


    logging.info("Traversing through directory to find all cipher suites...")
    for root, dirs, files in os.walk(rootdir):
        for f in files:
            if f.endswith(".pcap"):
                print("Processing {}".format(f))
                logging.info("Processing {}".format(f))
                # Might need to close FileCapture somehow to prevent the another loop running
                packets_json = pyshark.FileCapture(os.path.join(root, f), use_json=True)

                starttime = time.time()
                totaltime = 0.0
                found = False # Variable for ending the packet loop if ClientHello is found
                #for i, packet_json in enumerate(packets_json):
                for packet_json in packets_json:
                    starttime3 = time.time()
                    try:
                        if type(packet_json.ssl.record.handshake) == list:
                            handshakes = packet_json.ssl.record.handshake
                            for handshake in handshakes:
                                try:
                                    assert not (type(handshake)==list)
                                except AssertionError as e:
                                    logging.exception("Assertion failed")
                                # ClientHello found!
                                if int(handshake.type)==1:
                                    # Cipher Suites
                                    ciphersuites.extend([int(j) for j in handshake.ciphersuites.ciphersuite])
                                    # Compression Methods
                                    # Supported Groups
                                    # Signature Hash Algorithm
                                    found = True
                        else:
                            handshake = packet_json.ssl.record.handshake
                            try:
                                assert not (type(handshake)==list)
                            except AssertionError as e:
                                logging.exception("Assertion failed")

                            # ClientHello found!
                            if int(handshake.type)==1:
                                ciphersuites.extend([int(j) for j in handshake.ciphersuites.ciphersuite])
                                found = True

                    except AttributeError:
                        pass

                    finally:
                        if found:
                            logging.debug("Time spent on packet: {}s".format(time.time()-starttime3))
                            totaltime = totaltime + (time.time()-starttime3)
                            break
                        else:
                            logging.debug("Time spent on packet: {}s".format(time.time()-starttime3))
                            totaltime = totaltime + (time.time()-starttime3)

                logging.debug("Time spent on traffic: {}s".format(time.time()-starttime))
                logging.debug("Total time accumulated on traffic: {}s".format(totaltime))

                # If ClientHello cannot be found in the traffic
                if not found:
                    logging.warning("File has no ClientHello: {}".format(os.path.join(root,f)))
                ciphersuites = list(set(ciphersuites))

    logging.info("Done!")
    return ciphersuites

def searchCompressionMethods(rootdir):
    pass

def searchSupportedGroups(rootdir):
    pass

def searchSignatureHash_ClientHello(rootdir):
    pass

def searchSignatureHash_Certificate(rootdir):
    pass

# For handling data structure of jsonlayer type
def find_handshake(obj, target_type):
    if type(obj) == list:
        final = None
        for a_obj in obj:
            temp = find_handshake(a_obj, target_type)
            if temp:
                final = temp
        return final

    elif type(obj) == JsonLayer:    
        if obj.layer_name=='ssl' and hasattr(obj, 'record'):
            return find_handshake(obj.record, target_type)
        # elif obj.layer_name=='ssl' and hasattr(obj, 'handshake'):
        #     return find_handshake(obj.handshake, target_type)
        elif obj.layer_name=='record' and hasattr(obj, 'handshake'):
            return find_handshake(obj.handshake, target_type)
        # If correct handshake is identified
        elif obj.layer_name=='handshake' and int(obj.type)==target_type:
            return obj
        else:
            return None

# For handling data structure of dict type
def find_handshake2(obj, target_type):
    if type(obj) == list:
        final = None
        for a_obj in obj:
            temp = find_handshake2(a_obj, target_type)
            if temp:
                final = temp
        return final
    elif type(obj) == dict:
        if 'ssl.record' in obj:
            return find_handshake2(obj['ssl.record'], target_type)
        elif 'ssl.handshake' in obj:
            return find_handshake2(obj['ssl.handshake'], target_type)
        elif 'ssl.handshake.type' in obj and int(obj['ssl.handshake.type'])==target_type:
            return obj
    else:
        return None

def extract_tslssl_features(pcapfile, enumCipherSuites=[], enumCompressionMethods=[], enumSupportedGroups=[], enumSignatureHashClient=[], enumSignatureHashCert=[]):

    # TODO: implement dynamic features
    # Each packet will have its own record layer. If the layer does not exist, the features in that layer
    # will be zero-ed. Hence, we still assume one packet/Eth frame as the most basic unit of traffic

    # Traffic features for storing features of packets
    traffic_features = []
    packets = pyshark.FileCapture(pcapfile)
    packets_json = pyshark.FileCapture(pcapfile, use_json=True)
    total_ssl = 0
    #for i, (packet,packet_json) in enumerate(zip(packets, packets_json)):
    for i, packet_json in enumerate(packets_json):
        print('Packet ID {}'.format(i + 1))

        # if 'ssl' in packet and hasattr(packet.ssl,'record'):
        features = []


        ########################################################################
        ########################################################################
        # FOR DEBUGGING

        # try:
        #     #print(packet_json)
        #     #if i==2 or i==5:
        #     if i==2:
        #         print(packet_json)
        #         print('*********************************************************')
        #         print(packet_json.ssl.record.handshake)
        #         print('*********************************************************')
        #         print(packet_json.ssl.handshake.extension.len)
        #         #print(packet_json.ssl)
        #         #print(type(packet_json.ssl.record.handshake))
        #     #print(packet.ssl.record.handshake.ciphersuites)
        #     #cipher_suites = packet.ssl.record.handshake.ciphersuites.ciphersuite
        #     #print(cipher_suites.field_names)
        #     #print(cipher_suites)
        #     #for cipher_suite in cipher_suites:
        #     #    print(cipher_suite)
        # except AttributeError:
        #     pass
        # continue

        # try:
        #     if i == 8:
        #         print(packet_json)
        #         print(packets_json.ssl)
        #         myone = packet_json.ssl[0]
        #         handshake = find_handshake(packet_json.ssl, target_type=11)
        #         if handshake:
        #             features.append(int(handshake.length))
        #         else:
        #             features.append(0)
        # except AttributeError:
        #     features.append(0)
        # print(features)
        # continue
        ########################################################################
        ########################################################################


        # HANDSHAKE PROTOCOL
        ##################################################################

        # 1: ClientHello - LENGTH
        try:
            handshake = find_handshake(packet_json.ssl, target_type=1)
            if handshake:
                features.append(int(handshake.length))
            else:
                features.append(0)
        except AttributeError:
            features.append(0)

        # 2: ClientHello - CIPHER SUITE
        ciphersuite_feature = np.zeros_like(enumCipherSuites) # enumCipherSuites is the ref list
        try: 
            handshake = find_handshake(packet_json.ssl, target_type = 1)
            if handshake:
                for ciphersuite in handshake.ciphersuites.ciphersuite:
                    ciphersuite_int = int(ciphersuite)
                    if ciphersuite_int in enumCipherSuites:
                        ciphersuite_feature[enumCipherSuites.index(ciphersuite_int)] = 1
                features.extend(ciphersuite_feature)
            else:
                features.extend(ciphersuite_feature)
        except:
            features.extend(ciphersuite_feature)

        # 3: ClientHello - CIPHER SUITE LENGTH
        try:
            handshake = find_handshake(packet_json.ssl, target_type=1)
            if handshake:
                features.append(int(handshake.cipher_suites_length))
            else:
                features.append(0)
        except AttributeError:
            features.append(0)        

        # 4: ClientHello - COMPRESSION METHOD

        # 5: ClientHello - SUPPORTED GROUP LENGTH

        # 6: ClientHello - SUPPORTED GROUPS

        # 7: ClientHello - ENCRYPT THEN MAC LENGTH

        # 8: ClientHello - EXTENDED MASTER SECRET

        # 9: ClientHello - SIGNATURE HASH ALGORITHM

        # 10: ServerHello - LENGTH
        try:
            handshake = find_handshake(packet_json.ssl, target_type=2)
            if handshake:
                features.append(int(handshake.length))
            else:
                features.append(0)
        except AttributeError:
            features.append(0) 

        # 11: ServerHello - EXTENDED MASTER SECRET

        # 12: ServerHello - RENEGOTIATION INFO LENGTH

        # 13,14,15,16: Certificate - NUM_CERT, AVERAGE, MIN, MAX CERTIFICATE LENGTH
        handshake = None
        handshake2 = None
        # Attempt 1: use find_handshake()
        try:
            
            handshake = find_handshake(packet_json.ssl, target_type=11)
        except AttributeError:
            pass
        # Attempt 2: certificate is more difficult to identify. Use hardcode
        try: 
            handshake2 = find_handshake2(packet_json.ssl.value, target_type=11)
        except AttributeError:
            pass

        if handshake:
            certificates_length = [int(i) for i in handshake.certificates.certificate_length]
            mean_cert_len = sum(certificates_length)/float(len(certificates_length))
            features.extend([len(certificates_length), mean_cert_len,max(certificates_length),min(certificates_length)])
        elif handshake2:
            certificates_length = handshake2['ssl.handshake.certificates']['ssl.handshake.certificate_length']
            certificates_length = [int(i) for i in certificates_length]
            mean_cert_len = sum(certificates_length)/float(len(certificates_length))
            features.extend([len(certificates_length), mean_cert_len,max(certificates_length),min(certificates_length)])
        else:
            features.extend([0,0,0,0])

        # 17: Certificate - SIGNATURE ALGORITHM
        sighash_features = np.zeros_like(enumSignatureHashCert) # enumSignatureHashCert is the ref list
        try: 
            handshake = find_handshake(packet_json.ssl, target_type = 11)
            if handshake:
                for sighash in handshake.certificates.certificate:
                    algo_id = sighash.signedCertificate_element.algorithmIdentifier_element.id
                    if algo_id in enumSignatureHashCert:
                        sighash_features[enumSignatureHashCert.index(algo_id)] = 1
                features.extend(sighash_features)
            else:
                features.extend(sighash_features)
        except:
            features.extend(sighash_features)

        # 18: ServerHelloDone - LENGTH
        try:
            handshake = find_handshake(packet_json.ssl, target_type=14)
            if handshake:
                features.append(int(handshake.length))
            else:
                features.append(0)
        except AttributeError:
            features.append(0) 

        # 19: ClientKeyExchange - LENGTH

        # 20: ClientKeyExchange - PUBKEY LENGTH

        # 21: EncryptedHandshakeMessage - LENGTH


        #  CHANGE CIPHER PROTOCOL
        ##################################################################
        #  22: ChangeCipherSpec - LENGTH


        #  APPLICATION DATA PROTOCOL
        ##################################################################
        #  23: ApplicationDataProtocol - LENGTH




        print(features)

    print('Total SSL packets: {}'.format(total_ssl))


def search_and_extract(rootdir, csvfile):
    """
    Search in directory for pcap files and generate features from each pcap file

    Parameters:
    directory (str): ??
    csvfile (str): ??

    Returns:
    boolean: returns True if successful
    """
    count = 0
    for root, dirs, files in os.walk(rootdir):
        for f in files:
            if f.endswith(".pcap"):
                print("Parsing {}..".format(f))
                tcp_features = extract_tcp_features(os.path.join(root, f))
                # TODO: write the tcp_features into csv file

                count += 1
    print("{} pcap files have been successfully parsed with features generated".format(count))
    return True


if __name__ == '__main__':
    testcsv = 'test_tcp_features2.csv'
    rootdir = '/Users/YiLong/Desktop/SUTD/NUS-Singtel_Research/tls_atack/legitimate traffic'

    # Test whether tcp features are extracted
    # extract_tcp_features('sample/ari.nus.edu.sg_2018-12-24_14-30-02.pcap')
    # extract_tcp_features('sample/australianmuseum.net.au_2018-12-21_16-15-59.pcap')
    # extract_tcp_features('sample/www.stripes.com_2018-12-21_16-20-12.pcap')
    # extract_tcp_features('sample/www.zeroaggressionproject.org_2018-12-21_16-19-03.pcap')

    # Test whether all enums are generated
    #enumCipherSuites = searchCipherSuites(rootdir)
    #exit()
    #enumCompressionMethods = searchCompressionMethods(rootdir)
    #enumSupportedGroups = searchSupportedGroups(rootdir)
    #enumSignatureHashClient = searchSignatureHash_ClientHello(rootdir)
    #enumSignatureHashCert = searchSignatureHash_Certificate(rootdir)
    #enumCipherSuites,enumCompressionMethods, enumSupportedGroups, enumSignatureHashClient, enumSignatureHashCert = [],[],[],[],[]

    # Test whether all features are extracted
    sample = 'sample/ari.nus.edu.sg_2018-12-24_14-30-02.pcap'
    # sample = 'sample/www.zeroaggressionproject.org_2018-12-21_16-19-03.pcap'
    # sample = 'sample/australianmuseum.net.au_2018-12-21_16-15-59.pcap'
    # sample = 'sample/www.stripes.com_2018-12-21_16-20-12.pcap'
    extract_tslssl_features(sample)
    exit()

    # Test whether directory is searched correctly with features extracted 

    search_and_extract(rootdir, testcsv)
