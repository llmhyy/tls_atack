import numpy as np

# Note: acc refers to cosine similarity

def calculate_acc_of_traffic(predict, true):
    # Calculates the cosine similarity for each packet by comparing predict with true for a batch of traffic
    # and returns an array of cosine similarity for each packet
    dot = np.einsum('ijk,ijk->ij', predict, true)
    vnorm = (np.linalg.norm(predict,axis=2)*np.linalg.norm(true,axis=2))
    cos_sim = np.divide(dot,vnorm,out=np.zeros_like(dot), where=vnorm!=0.0)
    return cos_sim

def calculate_mean_acc_of_traffic(list_of_acc):
    # Calculates the mean cosine similarity across packets in a traffic. Works with batches
    return np.mean(list_of_acc, axis=1)

def calculate_median_acc_of_traffic(list_of_acc):
    # Calculates the median cosine similarity across packets in a traffic. Works with batches
    return np.median(list_of_acc, axis=1)