from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import argparse

import utils
import network
import image

# Argument parser
parser = argparse.ArgumentParser(description='Random Rotating Script')
# save path
parser.add_argument('--save_path', type=str, default='./result/random_rotating/', help='Saving path')
# training procedure parameters
parser.add_argument('--num_repeat', type=int, default=100, help='Whole procedure repeat times')
parser.add_argument('--pre_epoch_size', type=int, default=10, help='Pre-training epoch size')         # default value different from run_double_training.py
parser.add_argument('--conventional_epoch', type=int, default=200, help='Conventional training epoch size default=10*20 10 sessions')
parser.add_argument('--begin_epoch', type=int, default=20, help='Hebb learning start epoch')
parser.add_argument('--training_mode', type=str, default='rotating', help='Training mode')
# network parameters
parser.add_argument('--net_name', type=str, default='s_cc3', help='Network name')
parser.add_argument('--l_lambda_pre', type=float, default=0.3, help='Pre-training learning rate for Hebbian')
parser.add_argument('--cnn_lr_pre', type=float, default=1e-3, help='Pre-training learning rate for CNN')
parser.add_argument('--l_lambda', type=float, default=0.3, help='Learning rate for Hebbian')
parser.add_argument('--cnn_lr', type=float, default=1e-3, help='Learning rate for CNN')
parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to use')
# stimulus parameters
parser.add_argument('--phase_8', type=int, default=1, help='Phase 8 value')
parser.add_argument('--freq', type=float, default=0.02, help='Frequency value')
parser.add_argument('--noise_cutout', type=int, default=0.6, help='Noise cutout value')
parser.add_argument('--contrast', type=float, default=0.6, help='Contrast value')
parser.add_argument('--diff', type=int, default=9, help='Diff value')
parser.add_argument('--label_reverse', type=bool, default=True, help='Label reverse flag')
# test stimulus parameters
parser.add_argument('--num_test', type=int, default=20, help='Number of tests')
parser.add_argument('--ori', type=float, default=22.5, help='Orientation value')
parser.add_argument('--loc', type=int, default=5, help='Location value')
parser.add_argument('--print_test', type=bool, default=False, help='Print test flag')

args = parser.parse_args()

# save path
day = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
name_head = args.save_path + day + '/'
utils.mkdir(name_head)

# training procedure parameters
num_repeat = args.num_repeat
pre_epoch_size = args.pre_epoch_size
conventional_epoch = args.conventional_epoch
begin_epoch = args.begin_epoch
training_mode = args.training_mode

# network parameters
net_name = args.net_name
l_lambda_pre = args.l_lambda_pre
cnn_lr_pre = args.cnn_lr_pre
l_lambda = args.l_lambda
cnn_lr = args.cnn_lr
device = args.device

# stimulus parameters
phase_8 = args.phase_8
phase = phase_8 * np.pi / 8
freq = args.freq
noise_cutout = args.noise_cutout
contrast = args.contrast
diff = args.diff
label_reverse = args.label_reverse

# test stimulus parameters
num_test = args.num_test
ori = args.ori
num_ori = int(180 / ori)
loc = args.loc
print_test = args.print_test

# for data saving (1:ori1, 2:ori2)
Acc_test = np.zeros([num_repeat, conventional_epoch//2, int(80 / loc), int(180 / ori), 10])
Acc_test_pre = np.zeros(
    [num_repeat, pre_epoch_size, int(80 / loc), int(180 / ori), 10])
Acc_10diff_train = np.zeros([num_repeat, conventional_epoch//2, 10])

def GenTestData_torch(_ori, num=10, loc="L", diff=0, phase=0, freq=0.02, noise_cutout=0.7, var_noise=1, contrast=0.6):
    num_ori = int(180 / _ori)
    img_dataset = np.zeros([num * 2 * num_ori * 10, 1, 800, 200])
    label = np.zeros([num * 2 * num_ori * 10])
    # generate test dataset
    for i in range(num_ori):
        ori_i = i * _ori
        for diff in range(10):
            Img =  image.GenImg(size=[800, 200], orient=ori_i, loc=loc, diff=diff, phase=phase, freq=freq,
                noise_cutout=noise_cutout, var_noise=var_noise, contrast=contrast) 
            if i == 6:
                if label_reverse:
                    Img =  image.GenImg(size=[800, 200], orient=ori_i, loc=loc, diff=diff, phase=phase, freq=freq,
                            noise_cutout=noise_cutout, var_noise=var_noise, contrast=contrast,label_reverse=label_reverse)
            for ii in range(num):
                img_p, img_n = Img.gen_test()
                label[i * 10 * num * 2 + diff * num * 2 + ii] = img_p[0]
                label[i * 10 * num * 2 + diff * num * 2 + ii + num] = img_n[0]                
                img_dataset[i * 10 * num * 2 + diff *
                            num * 2 + ii, :, :, :] = img_p[1]
                img_dataset[i * 10 * num * 2 + diff *
                            num * 2 + ii + num, :, :, :] = img_n[1]
            
    # generate test dataset representation
    output = network.representation_torch(img_dataset)
    return output, label


def ff_train(s, t_label, t_data, show_epc, bg_epoch=0, phase=0, freq=0.02, diff=0, noise_cutout=0.7, contrast=0.6):
    '''
    s: repeat index
    t_label: test label
    t_data: test data
    show_epc: show epoch
    bg_epoch: begin epoch
    phase: phase
    freq: frequency
    diff: diff
    noise_cutout: noise cutout
    contrast: contrast

    return: net, weight, optimizer
    '''
    running_loss = 0.0
    # network   
    net = network.Net_sCC().to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    # optimizer
    cnn_optimizer_pre = optim.Adam([
        {'params': net.layer2.parameters()},
        {'params': net.readout.parameters()},
        {'params': net.layer1.parameters(), 'lr': cnn_lr_pre}
    ], lr=cnn_lr_pre)

    cnn_optimizer = optim.Adam([
        {'params': net.layer2.parameters()},
        {'params': net.readout.parameters()},
        {'params': net.layer1.parameters(), 'lr': cnn_lr}
    ], lr=cnn_lr)

    # input settings
    Img_ori1loc1 = image.GenImg(size=[800, 200], orient='V', loc=1, diff=diff, phase=phase,
                      freq=freq, noise_cutout=noise_cutout, contrast=contrast)
    Img_ori1loc2 = image.GenImg(size=[800, 200], orient='V', loc=2, diff=diff, phase=phase,
                      freq=freq, noise_cutout=noise_cutout, contrast=contrast)
    Img_ori1loc3 = image.GenImg(size=[800, 200], orient='V', loc=3, diff=diff, phase=phase,
                      freq=freq, noise_cutout=noise_cutout, contrast=contrast)
    Img_ori1loc4 = image.GenImg(size=[800, 200], orient='V', loc=4, diff=diff, phase=phase,
                      freq=freq, noise_cutout=noise_cutout, contrast=contrast)
    Img_ori2loc1 = image.GenImg(size=[800, 200], orient='H', loc=1, diff=diff, phase=phase,
                      freq=freq, noise_cutout=noise_cutout, contrast=contrast, label_reverse=label_reverse)
    Img_ori2loc2 = image.GenImg(size=[800, 200], orient='H', loc=2, diff=diff, phase=phase,
                      freq=freq, noise_cutout=noise_cutout, contrast=contrast, label_reverse=label_reverse)
    Img_ori2loc3 = image.GenImg(size=[800, 200], orient='H', loc=3, diff=diff, phase=phase,
                      freq=freq, noise_cutout=noise_cutout, contrast=contrast, label_reverse=label_reverse)
    Img_ori2loc4 = image.GenImg(size=[800, 200], orient='H', loc=4, diff=diff, phase=phase,
                      freq=freq, noise_cutout=noise_cutout, contrast=contrast, label_reverse=label_reverse)
    
    # for training inputs
    num_batch = 16   # batch size
    inputs_img = np.zeros([num_batch, 1, 800, 200])  
    labels = np.zeros([num_batch])
    # for pre-training inputs
    num_batch_pre = num_batch * 4   # batch size
    inputs_img_pre = np.zeros(
        [num_batch_pre, 1, 800, 200]) 
    labels_pre = np.zeros([num_batch_pre])

    # ------------------------------Pre Training--------------------------------------------
    weight = np.ones([80, 1])        # hebbian weight initialize
    inputs_img_pre = np.zeros(
        [num_batch*8, 1, 800, 200]) 
    labels_pre = np.zeros([num_batch*8])
    for pre_epoch in range(pre_epoch_size):
        # generate pre-training data
        for i in range(num_batch):
            labels_pre[i], img_tg = Img_ori1loc1.gen_train()
            inputs_img_pre[i, :, :, :] = img_tg
            labels_pre[1*num_batch+i], img_tg = Img_ori1loc2.gen_train()
            inputs_img_pre[1*num_batch+i, :, :, :] = img_tg
            labels_pre[2*num_batch+i], img_tg = Img_ori1loc3.gen_train()
            inputs_img_pre[2*num_batch+i, :, :, :] = img_tg
            labels_pre[3*num_batch+i], img_tg = Img_ori1loc4.gen_train()
            inputs_img_pre[3*num_batch+i, :, :, :] = img_tg
            labels_pre[4*num_batch+i], img_tg = Img_ori2loc1.gen_train()
            inputs_img_pre[4*num_batch+i, :, :, :] = img_tg
            labels_pre[5*num_batch+i], img_tg = Img_ori2loc2.gen_train()
            inputs_img_pre[5*num_batch+i, :, :, :] = img_tg
            labels_pre[6*num_batch+i], img_tg = Img_ori2loc3.gen_train()
            inputs_img_pre[6*num_batch+i, :, :, :] = img_tg
            labels_pre[7*num_batch+i], img_tg = Img_ori2loc4.gen_train()
            inputs_img_pre[7*num_batch+i, :, :, :] = img_tg
        # all ori pre-train in one cnn network
        for pre_epoch_ in range(8):
            inputs = network.representation_torch(
                inputs_img_pre[pre_epoch_*num_batch:(pre_epoch_+1)*num_batch])
            inputs = utils.norma_rep(inputs)
            inputs = network.feedforward(inputs, weight)
            inputs = utils.norma_rep(inputs)
            weight = network.update_weight(weight, inputs, l_lambda=l_lambda_pre)
            cnn_optimizer_pre.zero_grad()
            b_x = torch.tensor(inputs, dtype=torch.float32)
            b_labels = torch.tensor(
                labels_pre[pre_epoch_*num_batch:(pre_epoch_+1)*num_batch])
            b_x, b_labels = b_x.to(device), b_labels.to(device)
            net.train()
            outputs = net(b_x)
            outputs = outputs.squeeze(1)   
            loss = criterion(outputs, (b_labels + 1) // 2)
            loss.backward()
            cnn_optimizer_pre.step()
        # testing for different location during pre-training
        net.eval()
        for i in range(80 // loc):  
            y = network.feedforward(t_data, weight)
            y = utils.norma_rep(y)
            y = torch.tensor(y, dtype=torch.float32)
            y = y.to(device)
            t_label = t_label.to(device)
            Acc_test_pre[s, pre_epoch, i, :, :] = network.test(net, y, num_test * 2, t_label,  
                                                          prt=False)[:-1].reshape(num_ori, -1)  
            t_data = np.roll(t_data, -5, axis=2)

    # ------------------------------Conventional Training--------------------------------------------
    _epc = 1
    for epoch in range(conventional_epoch):  # loop over the dataset multiple times
        # Generate conventional training data
        if training_mode == 'rotating':
            all_labels=np.zeros([3*7]) 
            all_inputs_img=np.zeros([3*7,1,800,200])
            for i in range(3):
                all_labels[i*7+0], img_tg = Img_ori2loc1.gen_train()  
                all_inputs_img[i*7+0, :, :, :] = img_tg
                all_labels[i*7+1], img_tg = Img_ori1loc2.gen_train()  
                all_inputs_img[i*7+1, :, :, :] = img_tg
                all_labels[i*7+2], img_tg = Img_ori2loc2.gen_train()  
                all_inputs_img[i*7+2, :, :, :] = img_tg
                all_labels[i*7+3], img_tg = Img_ori1loc3.gen_train()  
                all_inputs_img[i*7+3, :, :, :] = img_tg
                all_labels[i*7+4], img_tg = Img_ori2loc3.gen_train()  
                all_inputs_img[i*7+4, :, :, :] = img_tg
                all_labels[i*7+5], img_tg = Img_ori1loc4.gen_train()  
                all_inputs_img[i*7+5, :, :, :] = img_tg
                all_labels[i*7+6], img_tg = Img_ori2loc4.gen_train()  
                all_inputs_img[i*7+6, :, :, :] = img_tg
            shift = np.random.randint(0, 20)
            all_idx = np.arange(21)
            all_idx = np.roll(all_idx, shift)
            all_labels=all_labels[all_idx]
            all_inputs_img=all_inputs_img[all_idx]
            labels=all_labels[:num_batch]
            inputs_img=all_inputs_img[:num_batch, :, :, :]

        elif training_mode == 'random':
            all_labels=np.zeros([3*7]) 
            all_inputs_img=np.zeros([3*7,1,800,200])
            for i in range(3):
                all_labels[i*7+0], img_tg = Img_ori2loc1.gen_train()  
                all_inputs_img[i*7+0, :, :, :] = img_tg
                all_labels[i*7+1], img_tg = Img_ori1loc2.gen_train()  
                all_inputs_img[i*7+1, :, :, :] = img_tg
                all_labels[i*7+2], img_tg = Img_ori2loc2.gen_train()  
                all_inputs_img[i*7+2, :, :, :] = img_tg
                all_labels[i*7+3], img_tg = Img_ori1loc3.gen_train()  
                all_inputs_img[i*7+3, :, :, :] = img_tg
                all_labels[i*7+4], img_tg = Img_ori2loc3.gen_train()  
                all_inputs_img[i*7+4, :, :, :] = img_tg
                all_labels[i*7+5], img_tg = Img_ori1loc4.gen_train()  
                all_inputs_img[i*7+5, :, :, :] = img_tg
                all_labels[i*7+6], img_tg = Img_ori2loc4.gen_train()  
                all_inputs_img[i*7+6, :, :, :] = img_tg
            all_rand_idx = np.random.permutation(3*7)
            all_labels=all_labels[all_rand_idx]
            all_inputs_img=all_inputs_img[all_rand_idx]
            labels=all_labels[:num_batch]
            inputs_img=all_inputs_img[:num_batch, :, :, :]

        else:
            raise ValueError('Invalid training mode. Please choose from \'rotating\' or \'random\'.')
        # Get the representation
        inputs = network.representation_torch(inputs_img)
        inputs = utils.norma_rep(inputs)
        inputs = network.feedforward(inputs, weight)
        inputs = utils.norma_rep(inputs)
        # update the hebbian weight
        if epoch > bg_epoch:
            weight = network.update_weight(weight, inputs, l_lambda=l_lambda)
        # update the cnn weight
        cnn_optimizer.zero_grad()
        b_x = torch.tensor(inputs, dtype=torch.float32)
        b_labels = torch.tensor(labels)
        net.train()
        b_x, b_labels = b_x.to(device), b_labels.to(device)
        outputs = net(b_x)
        outputs = outputs.squeeze(1)   
        loss = criterion(outputs, (b_labels + 1) // 2)
        # pdb.set_trace()
        loss.backward()
        cnn_optimizer.step()
        # print statistics
        running_loss += loss.item()
        # test during conventional training
        if epoch % show_epc == show_epc - 1:  # save every show epoch
            # Testing for different location
            for i in range(80 // loc):  
                y = network.feedforward(t_data, weight)
                y = utils.norma_rep(y)
                y = torch.tensor(y, dtype=torch.float32)
                y = y.to(device)
                t_label = t_label.to(device)
                Acc_test[s, _epc - 1, i, :, :] = network.test(net, y, num_test * 2, t_label,  
                                                         print_test)[:-1].reshape(num_ori, -1) 
                t_data = np.roll(t_data, -5, axis=2)      # change the test data to the next location\

            # Testing for 10 different diff in train setting
            if training_mode == 'rotating':
                # 10 diff training test
                all_diff10_inputs_img = np.zeros([21,2,10, 1, 800, 200])  
                all_diff10_labels = np.zeros([21,2,10])
                for diff in range(10):
                    Img_ori1loc2_test = image.GenImg(size=[800, 200], orient='V', loc=2, diff=diff, phase=phase,
                                    freq=freq, noise_cutout=noise_cutout, contrast=contrast)
                    Img_ori1loc3_test = image.GenImg(size=[800, 200], orient='V', loc=3, diff=diff, phase=phase,
                                    freq=freq, noise_cutout=noise_cutout, contrast=contrast)
                    Img_ori1loc4_test = image.GenImg(size=[800, 200], orient='V', loc=4, diff=diff, phase=phase,
                                    freq=freq, noise_cutout=noise_cutout, contrast=contrast)
                    Img_ori2loc1_test = image.GenImg(size=[800, 200], orient='H', loc=1, diff=diff, phase=phase,
                                    freq=freq, noise_cutout=noise_cutout, contrast=contrast, label_reverse=label_reverse)
                    Img_ori2loc2_test = image.GenImg(size=[800, 200], orient='H', loc=2, diff=diff, phase=phase,
                                    freq=freq, noise_cutout=noise_cutout, contrast=contrast, label_reverse=label_reverse)
                    Img_ori2loc3_test = image.GenImg(size=[800, 200], orient='H', loc=3, diff=diff, phase=phase,
                                    freq=freq, noise_cutout=noise_cutout, contrast=contrast, label_reverse=label_reverse)
                    Img_ori2loc4_test = image.GenImg(size=[800, 200], orient='H', loc=4, diff=diff, phase=phase,
                                    freq=freq, noise_cutout=noise_cutout, contrast=contrast, label_reverse=label_reverse)
                    
                    for i in range(3):
                        img_p, img_n = Img_ori2loc1_test.gen_test()
                        all_diff10_labels[i*7+0,0,diff] = img_p[0]
                        all_diff10_labels[i*7+0,1,diff] = img_n[0]                
                        all_diff10_inputs_img[i*7+0,0,diff, :, :, :] = img_p[1]
                        all_diff10_inputs_img[i*7+0,1,diff, :, :, :] = img_n[1]

                        img_p, img_n = Img_ori1loc2_test.gen_test()
                        all_diff10_labels[i*7+1,0,diff] = img_p[0]
                        all_diff10_labels[i*7+1,1,diff] = img_n[0]                
                        all_diff10_inputs_img[i*7+1,0,diff, :, :, :] = img_p[1]
                        all_diff10_inputs_img[i*7+1,1,diff, :, :, :] = img_n[1]

                        img_p, img_n = Img_ori2loc2_test.gen_test()
                        all_diff10_labels[i*7+2,0,diff] = img_p[0]
                        all_diff10_labels[i*7+2,1,diff] = img_n[0]                
                        all_diff10_inputs_img[i*7+2,0,diff, :, :, :] = img_p[1]
                        all_diff10_inputs_img[i*7+2,1,diff, :, :, :] = img_n[1]

                        img_p, img_n = Img_ori1loc3_test.gen_test()
                        all_diff10_labels[i*7+3,0,diff] = img_p[0]
                        all_diff10_labels[i*7+3,1,diff] = img_n[0]                
                        all_diff10_inputs_img[i*7+3,0,diff, :, :, :] = img_p[1]
                        all_diff10_inputs_img[i*7+3,1,diff, :, :, :] = img_n[1]

                        img_p, img_n = Img_ori2loc3_test.gen_test()
                        all_diff10_labels[i*7+4,0,diff] = img_p[0]
                        all_diff10_labels[i*7+4,1,diff] = img_n[0]                
                        all_diff10_inputs_img[i*7+4,0,diff, :, :, :] = img_p[1]
                        all_diff10_inputs_img[i*7+4,1,diff, :, :, :] = img_n[1]

                        img_p, img_n = Img_ori1loc4_test.gen_test()
                        all_diff10_labels[i*7+5,0,diff] = img_p[0]
                        all_diff10_labels[i*7+5,1,diff] = img_n[0]                
                        all_diff10_inputs_img[i*7+5,0,diff, :, :, :] = img_p[1]
                        all_diff10_inputs_img[i*7+5,1,diff, :, :, :] = img_n[1]

                        img_p, img_n = Img_ori2loc4_test.gen_test()
                        all_diff10_labels[i*7+6,0,diff] = img_p[0]
                        all_diff10_labels[i*7+6,1,diff] = img_n[0]                
                        all_diff10_inputs_img[i*7+6,0,diff, :, :, :] = img_p[1]
                        all_diff10_inputs_img[i*7+6,1,diff, :, :, :] = img_n[1]
                    
                all_diff10_labels=all_diff10_labels[all_idx]
                all_diff10_inputs_img=all_diff10_inputs_img[all_idx]
                diff10_labels =all_diff10_labels[:num_batch,:,:]
                diff10_inputs_img =all_diff10_inputs_img[:num_batch,:,:,:,:,:]
                
                diff10_labels = np.transpose(diff10_labels,(2,0,1))
                diff10_labels = diff10_labels.reshape(-1)
                diff10_inputs_img = np.transpose(diff10_inputs_img,(2,0,1,3,4,5))
                diff10_inputs_img = diff10_inputs_img.reshape(-1,1,800,200)  
            
            elif training_mode == 'random':
                # 10 diff training test
                all_diff10_inputs_img = np.zeros([21,2,10, 1, 800, 200])  
                all_diff10_labels = np.zeros([21,2,10])
                for diff in range(10):
                    Img_ori1loc2_test = image.GenImg(size=[800, 200], orient='V', loc=2, diff=diff, phase=phase,
                                    freq=freq, noise_cutout=noise_cutout, contrast=contrast)
                    Img_ori1loc3_test = image.GenImg(size=[800, 200], orient='V', loc=3, diff=diff, phase=phase,
                                    freq=freq, noise_cutout=noise_cutout, contrast=contrast)
                    Img_ori1loc4_test = image.GenImg(size=[800, 200], orient='V', loc=4, diff=diff, phase=phase,
                                    freq=freq, noise_cutout=noise_cutout, contrast=contrast)
                    Img_ori2loc1_test = image.GenImg(size=[800, 200], orient='H', loc=1, diff=diff, phase=phase,
                                    freq=freq, noise_cutout=noise_cutout, contrast=contrast, label_reverse=label_reverse)
                    Img_ori2loc2_test = image.GenImg(size=[800, 200], orient='H', loc=2, diff=diff, phase=phase,
                                    freq=freq, noise_cutout=noise_cutout, contrast=contrast, label_reverse=label_reverse)
                    Img_ori2loc3_test = image.GenImg(size=[800, 200], orient='H', loc=3, diff=diff, phase=phase,
                                    freq=freq, noise_cutout=noise_cutout, contrast=contrast, label_reverse=label_reverse)
                    Img_ori2loc4_test = image.GenImg(size=[800, 200], orient='H', loc=4, diff=diff, phase=phase,
                                    freq=freq, noise_cutout=noise_cutout, contrast=contrast, label_reverse=label_reverse)
                    
                    for i in range(3):
                        img_p, img_n = Img_ori2loc1_test.gen_test()
                        all_diff10_labels[i*7+0,0,diff] = img_p[0]
                        all_diff10_labels[i*7+0,1,diff] = img_n[0]                
                        all_diff10_inputs_img[i*7+0,0,diff, :, :, :] = img_p[1]
                        all_diff10_inputs_img[i*7+0,1,diff, :, :, :] = img_n[1]

                        img_p, img_n = Img_ori1loc2_test.gen_test()
                        all_diff10_labels[i*7+1,0,diff] = img_p[0]
                        all_diff10_labels[i*7+1,1,diff] = img_n[0]                
                        all_diff10_inputs_img[i*7+1,0,diff, :, :, :] = img_p[1]
                        all_diff10_inputs_img[i*7+1,1,diff, :, :, :] = img_n[1]

                        img_p, img_n = Img_ori2loc2_test.gen_test()
                        all_diff10_labels[i*7+2,0,diff] = img_p[0]
                        all_diff10_labels[i*7+2,1,diff] = img_n[0]                
                        all_diff10_inputs_img[i*7+2,0,diff, :, :, :] = img_p[1]
                        all_diff10_inputs_img[i*7+2,1,diff, :, :, :] = img_n[1]

                        img_p, img_n = Img_ori1loc3_test.gen_test()
                        all_diff10_labels[i*7+3,0,diff] = img_p[0]
                        all_diff10_labels[i*7+3,1,diff] = img_n[0]                
                        all_diff10_inputs_img[i*7+3,0,diff, :, :, :] = img_p[1]
                        all_diff10_inputs_img[i*7+3,1,diff, :, :, :] = img_n[1]

                        img_p, img_n = Img_ori2loc3_test.gen_test()
                        all_diff10_labels[i*7+4,0,diff] = img_p[0]
                        all_diff10_labels[i*7+4,1,diff] = img_n[0]                
                        all_diff10_inputs_img[i*7+4,0,diff, :, :, :] = img_p[1]
                        all_diff10_inputs_img[i*7+4,1,diff, :, :, :] = img_n[1]

                        img_p, img_n = Img_ori1loc4_test.gen_test()
                        all_diff10_labels[i*7+5,0,diff] = img_p[0]
                        all_diff10_labels[i*7+5,1,diff] = img_n[0]                
                        all_diff10_inputs_img[i*7+5,0,diff, :, :, :] = img_p[1]
                        all_diff10_inputs_img[i*7+5,1,diff, :, :, :] = img_n[1]

                        img_p, img_n = Img_ori2loc4_test.gen_test()
                        all_diff10_labels[i*7+6,0,diff] = img_p[0]
                        all_diff10_labels[i*7+6,1,diff] = img_n[0]                
                        all_diff10_inputs_img[i*7+6,0,diff, :, :, :] = img_p[1]
                        all_diff10_inputs_img[i*7+6,1,diff, :, :, :] = img_n[1]
                    
                all_diff10_labels=all_diff10_labels[all_rand_idx]
                all_diff10_inputs_img=all_diff10_inputs_img[all_rand_idx]
                diff10_labels =all_diff10_labels[:num_batch,:,:]
                diff10_inputs_img =all_diff10_inputs_img[:num_batch,:,:,:,:,:]
                
                diff10_labels = np.transpose(diff10_labels,(2,0,1))
                diff10_labels = diff10_labels.reshape(-1)
                diff10_inputs_img = np.transpose(diff10_inputs_img,(2,0,1,3,4,5))
                diff10_inputs_img = diff10_inputs_img.reshape(-1,1,800,200)
            
            else:
                raise ValueError('Invalid training mode. Please choose from \'rotating\' or \'random\'.')
                        # rotating & random
            y = network.representation_torch(diff10_inputs_img)
            y = network.feedforward(y, weight)
            y = utils.norma_rep(y)
            y = torch.tensor(y, dtype=torch.float32)
            diff10_labels = torch.tensor(diff10_labels)
            y = y.to(device)
            diff10_labels = diff10_labels.to(device)
            Acc_10diff_train[s, _epc - 1, :] = network.test(net, y, num_batch*2, diff10_labels,  
                                                         print_test)[:-1] 
            
            _epc += 1

    # # save every repeat model
    # PATH = name_head + 'model/' + str(s) + '/'
    # utils.mkdir(PATH)
    # torch.save(best_net.state_dict(), PATH + 'net.pth')
    # np.save(PATH + 'weight.npy', best_weight)
    return net, weight, cnn_optimizer


# ------------------------------Training Procedure-------------------------------------------------
# generate test data and get the representation
t_data, t_label = GenTestData_torch(
    _ori=ori, num=num_test, phase=phase, freq=freq, diff=diff, noise_cutout=noise_cutout, contrast=contrast) 
t_data = utils.norma_rep(t_data)
t_label = torch.tensor(t_label)

# repeat the whole experiment for many times
for s in tqdm(range(num_repeat)): 

    # ------------------------------Conventional Training--------------------------------------------
    net, weight, cnn_optimizer = ff_train(
        s, t_label, t_data,
        show_epc=2,
        bg_epoch=begin_epoch,
        phase=phase, freq=freq,
        diff=diff, noise_cutout=noise_cutout, contrast=contrast
    )  

# save the results
np.save(name_head + 'Acc_test.npy', Acc_test)
np.save(name_head + 'Acc_testpre.npy', Acc_test_pre)
np.save(name_head + 'Acc_10diff_train.npy', Acc_10diff_train)
