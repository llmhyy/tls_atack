import numpy as np

# Note: acc refers to cosine similarity

def calculate_acc_of_traffic(predict, true, seqlen):
    # Calculates the cosine similarity for each packet for a batch of traffic
    pass

def calculate_mean_acc(list_of_acc):
    # Calculates the mean cosine similarity across packets for a batch of traffic
    pass

def calculate_median_acc(list_of_acc):
    # Calculates the median cosine similarity across packets for a batch of traffic
    pass

#################################################################################
# OLD FUNCTIONS


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