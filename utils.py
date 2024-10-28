import os
import numpy as np
import torch

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def np_acc(outputs, labels):
    """Return all acc

    Args:
        outputs (_type_): outputs
        labels (_type_): labels

    Returns:
        acc([1,]): acc
    """
    output = outputs.detach().cpu().numpy()
    total = (np.sign(output * labels) + 1) / 2   
    acc = np.sum(total) / len(output)    
    return acc

def ACC(outputs, labels, num_batch):
    """Calculate acc (tensor).

    Args:
        outputs (_type_): outputs
        labels (_type_): labels
        num_batch (_type_): Batch size.

    Returns:
        acc([]): _description_
    """
    # pdb.set_trace()
    num_test = len(outputs) // num_batch  
    acc = np.zeros(num_test + 1)

    # if outputs.device!= 'cpu':
    #     outputs = outputs.cpu()
    total = (torch.sign(outputs * labels) + 1) / 2   
    for i in range(num_test):       
        acc[i] = torch.sum(total[num_batch * i:num_batch * (i + 1)]) / num_batch
    acc[-1] = torch.sum(total) / len(outputs)
    return acc

def norma_rep(represtation):
    size_input = represtation.shape
    represtation = np.reshape(
        normalize(np.reshape(represtation, [size_input[0], -1])), size_input)
    return represtation


def normalize(x):
    mean = np.expand_dims(np.mean(x, axis=1), 1)
    sigma = np.expand_dims(np.std(x, axis=1), 1)
    sigma[sigma == 0] = 1
    nor_x = (x - mean) / sigma
    return nor_x