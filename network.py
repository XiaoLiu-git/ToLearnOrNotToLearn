import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tool_gabor_45 import Vernier, Gabor
import utils

class Net_sCC(nn.Module):

    def __init__(self):
        # an affine operation: y = Wx + b
        super(Net_sCC, self).__init__()
        self.layer1 = torch.nn.Conv2d(1, 6, 3)

        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.layer2 = nn.Conv2d(6, 10, 3)
        # self.conv2 = nn.Conv2d(16, 32, 3)
        self.readout = nn.Conv2d(10, 1, 3)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # pdb.set_trace()
        # x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.layer1(x))
        # Max pooling over a (2, 2) window
        x = F.relu(self.layer2(x))
        # If the size is a square, you can specify with a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = torch.flatten(x, 1)
        x = self.readout(x)
        # pdb.set_trace()
        x = torch.flatten(x, 1).unsqueeze(1) #需要添加一个channel维度给max_pool
        # x = F.max_pool1d(x, 408)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x
    
def representation_torch(img_dataset, num_x=40, num_theta=18):
    imgset = np.concatenate((img_dataset[:,:,-40:,:],img_dataset,img_dataset[:,:,:50,:]),axis=2)  #写死
    # generate gabor filter kernel
    [w, h] = img_dataset[0, 0].shape
    func_size = [100, h, num_theta]  #整除取下  把w//4都改成100
    basis_gabor = np.zeros((num_theta, 100, h))
    gb = Gabor(sigma=30, freq=.01) #二维矩阵
    for theta in range(num_theta): #生成所有theta的gabor filter.每个[w // 4, h]
        basis_gabor[theta,:, :] = gb.genGabor(func_size[:-1],
                                                theta * 180 / num_theta)    
    # representation
    conv = torch.nn.Conv2d(1, 18, (100, h), stride=10, padding=0, bias=False)
    conv.weight.data = torch.Tensor(np.expand_dims(basis_gabor,axis=1))
    # conv.bias.data = torch.Tensor(np.zeros([num_theta]))
    imgset = torch.Tensor(imgset)
    imgset_conv=conv(imgset)
    imgset_representation=np.flip(np.swapaxes(imgset_conv.detach().numpy(),1,3),axis=2)
    return imgset_representation

def test(net, inputs, num_batch, labels, prt=False):
    """test

    Args:
        net (_type_): net
        inputs (_type_): t_inputs[num_test(trials num) * 4(location), 1, 40, 18]
                        representation feature map size[40,18].
        num_batch (_type_): Batch size.
        labels (_type_): Corresponding label.
        prt (bool, optional): Print accuracy. Only print acc[0]-acc[4].Defaults to False.

    Returns:
        acc([5,num_batch]): acc[0]-acc[3] are batch accuracies, a[4]/a[-1] is all accuracy.
    """
    with torch.no_grad():
        # b_x = torch.tensor(inputs, dtype=torch.float32)
        outputs = net(inputs)
        outputs = outputs.squeeze(1)
        acc = utils.ACC(outputs, labels, num_batch)
        if prt:
            print("Accuracy:")
            print(np.around(acc,3) * 100)
    return acc

def feedforward(x, W):
    """
    :param x:size [num * 2 * num_ori, 1, 40, 18]
    :param W: size[1, 18]
    :return: Y: same size as x, Y=x*W
    """
    return x * W

def update_weight(w, inputs, l_lambda=.1,exposure=False):
    # pdb.set_trace()
    inputs=np.where((inputs > -.5)&(inputs < .5), 0, inputs)
    inputs = np.sign(inputs)
    dw = np.mean(np.mean(inputs[:-1, :, :, :] * inputs[1:, :, :, :], axis=0),
                 axis=-1)
    if exposure :
        w = np.concatenate((w[:22,:] - 100*l_lambda * dw.T[:22,:],w[22:-1,:] + l_lambda * dw.T[22:-1,:],w[-1:,:] - 100*l_lambda * dw.T[-1:,:]),axis=0) # loc2是前半
    else:
        w = w + l_lambda * dw.T
    # print(dw)
    w = np.maximum(w, 0)

    return w