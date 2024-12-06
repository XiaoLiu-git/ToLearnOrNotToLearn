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
parser = argparse.ArgumentParser(description='Double Training Script')
# save path
parser.add_argument('--save_path', type=str, default='./result/double_training/', help='Saving path')
# training procedure parameters
parser.add_argument('--num_repeat', type=int, default=100, help='Whole procedure repeat times')
parser.add_argument('--pre_epoch_size', type=int, default=20, help='Pre-training epoch size')
parser.add_argument('--conventional_epoch', type=int, default=10*20, help='Conventional training epoch size')
parser.add_argument('--begin_epoch', type=int, default=20, help='Hebb learning start epoch')
# network parameters
parser.add_argument('--net_name', type=str, default='s_cc3', help='Network name')
parser.add_argument('--l_lambda', type=float, default=0.3, help='Lambda value')
parser.add_argument('--cnn_lr', type=float, default=1e-3, help='Learning rate for CNN')
parser.add_argument('--cnn_lr_pre', type=float, default=1e-3, help='Pre-training learning rate for CNN')
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

# network parameters
net_name = args.net_name
l_lambda = args.l_lambda
cnn_lr = args.cnn_lr
cnn_lr_pre = args.cnn_lr_pre
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
# 1
AccALL_1 = np.zeros([num_repeat, 1+conventional_epoch//2, 5])
Acc_test_1 = np.zeros([num_repeat, conventional_epoch//2, int(40 / loc), int(180 / ori), 10])
Acc_test_db_1 = np.zeros([num_repeat, 200, int(40 / loc), int(180 / ori), 10])
Acc_test_pre_1 = np.zeros(
    [num_repeat, pre_epoch_size, int(40 / loc), int(180 / ori), 10])
Acc_traindb_1 = np.zeros([num_repeat, 200])
Loss_traindb = np.zeros([num_repeat, 200])
# 2
AccALL_2 = np.zeros([num_repeat, 1+conventional_epoch//2, 5])
Acc_test_2 = np.zeros([num_repeat, conventional_epoch//2, int(40 / loc), int(180 / ori), 10])
Acc_test_db_2 = np.zeros([num_repeat, 200, int(40 / loc), int(180 / ori), 10])
Acc_test_pre_2 = np.zeros(
    [num_repeat, pre_epoch_size, int(40 / loc), int(180 / ori), 10])
Acc_traindb_2 = np.zeros([num_repeat, 200])


def GenTestData_torch(_ori, num=10, loc="L", diff=0, phase=0, freq=0.02, noise_cutout=0.7, var_noise=1, contrast=0.6):
    num_ori = int(180 / _ori)
    img_dataset = np.zeros([num * 2 * num_ori * 10, 1, 400, 200])
    label = np.zeros([num * 2 * num_ori * 10])
    # generate test dataset
    for i in range(num_ori):
        ori_i = i * _ori
        for diff in range(10):
            Img = image.GenImg(orient=ori_i, loc=loc, diff=diff, phase=phase, freq=freq,
                noise_cutout=noise_cutout, var_noise=var_noise, contrast=contrast) 
            if i == 6:
                if label_reverse:
                    Img = image.GenImg(orient=ori_i, loc=loc, diff=diff, phase=phase, freq=freq,
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

    return: acc_list, loss_list, net, weight, optimizer, criterion
    '''
    running_loss = 0.0
    # network   
    net_1 = network.Net_sCC().to(device)
    net_2 = network.Net_sCC().to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    # optimizer
    cnn_1_optimizer_pre = optim.Adam([
        {'params': net_1.layer2.parameters()},
        {'params': net_1.readout.parameters()},
        {'params': net_1.layer1.parameters(), 'lr': cnn_lr_pre}
    ], lr=cnn_lr_pre)
    cnn_2_optimizer_pre = optim.Adam([
        {'params': net_2.layer2.parameters()},
        {'params': net_2.readout.parameters()},
        {'params': net_2.layer1.parameters(), 'lr': cnn_lr_pre}
    ], lr=cnn_lr_pre)

    cnn_1_optimizer = optim.Adam([
        {'params': net_1.layer2.parameters()},
        {'params': net_1.readout.parameters()},
        {'params': net_1.layer1.parameters(), 'lr': cnn_lr}
    ], lr=cnn_lr)
    cnn_2_optimizer = optim.Adam([
        {'params': net_2.layer2.parameters()},
        {'params': net_2.readout.parameters()},
        {'params': net_2.layer1.parameters(), 'lr': cnn_lr}
    ], lr=cnn_lr)

    # input settings
    Img_L = image.GenImg(orient='V', loc="L", diff=diff, phase=phase,
                      freq=freq, noise_cutout=noise_cutout, contrast=contrast)
    Img_R = image.GenImg(orient='V', loc="R", diff=diff, phase=phase,
                      freq=freq, noise_cutout=noise_cutout, contrast=contrast)
    Img_DT = image.GenImg(orient='H', loc="R", diff=diff, phase=phase,
                       freq=freq, noise_cutout=noise_cutout, contrast=contrast,label_reverse=label_reverse)
    Img_th = image.GenImg(orient='H', loc="L", diff=diff, phase=phase,
                       freq=freq, noise_cutout=noise_cutout, contrast=contrast,label_reverse=label_reverse)
    # for training inputs
    num_batch = 16   # batch size
    inputs_img = np.zeros([num_batch, 1, 400, 200])  
    labels = np.zeros([num_batch])
    # for pre-training inputs
    num_batch_pre = num_batch * 4   # batch size
    inputs_img_pre = np.zeros(
        [num_batch_pre, 1, 400, 200]) 
    labels_pre = np.zeros([num_batch_pre])

    # ------------------------------Pre Training--------------------------------------------
    weight = np.ones([40, 1])        # hebbian weight initialize
    inputs_img_pre = np.zeros(
        [num_batch*4, 1, 400, 200]) 
    labels_pre = np.zeros([num_batch*4])
    for pre_epoch in range(pre_epoch_size):
        # generate pre-training data
        for i in range(num_batch):
            labels_pre[i], img_tg = Img_L.gen_train()  # gen train
            inputs_img_pre[i, :, :, :] = img_tg
            labels_pre[1*num_batch+i], img_tg = Img_DT.gen_train()  # gen train
            inputs_img_pre[1*num_batch+i, :, :, :] = img_tg
            labels_pre[2*num_batch+i], img_tg = Img_R.gen_train()  # gen train
            inputs_img_pre[2*num_batch+i, :, :, :] = img_tg
            labels_pre[3*num_batch+i], img_tg = Img_th.gen_train()  # gen train
            inputs_img_pre[3*num_batch+i, :, :, :] = img_tg
        # cnn_1 only pre-train for ori1 data, cnn_2 only pre-train for ori2 data
        for pre_epoch_ in range(4):
            inputs = network.representation_torch(
                inputs_img_pre[pre_epoch_*num_batch:(pre_epoch_+1)*num_batch])
            inputs = utils.norma_rep(inputs)
            inputs = network.feedforward(inputs, weight)
            inputs = utils.norma_rep(inputs)
            weight = network.update_weight(weight, inputs, l_lambda=l_lambda)
            if pre_epoch_//2==0:
                cnn_1_optimizer_pre.zero_grad()
            else:
                cnn_2_optimizer_pre.zero_grad()
            b_x = torch.tensor(inputs, dtype=torch.float32)
            b_labels = torch.tensor(
                labels_pre[pre_epoch_*num_batch:(pre_epoch_+1)*num_batch])
            b_x, b_labels = b_x.to(device), b_labels.to(device)
            if pre_epoch_//2==0:
                net_1.train()
                outputs = net_1(b_x)
            else:
                net_2.train()
                outputs = net_2(b_x)
            outputs = outputs.squeeze(1)   # 
            loss = criterion(outputs, (b_labels + 1) // 2)
            loss.backward()
            if pre_epoch_//2==0:
                cnn_1_optimizer_pre.step()
            else:
                cnn_2_optimizer_pre.step()
        # testing for different location during pre-training
        net_1.eval()
        net_2.eval()
        for i in range(40 // loc):  
            y = network.feedforward(t_data, weight)
            y = utils.norma_rep(y)
            y = torch.tensor(y, dtype=torch.float32)
            y = y.to(device)
            t_label = t_label.to(device)
            #1
            Acc_test_pre_1[s, pre_epoch, i, :, :] = network.test(net_1, y, num_test * 2, t_label,  
                                                          prt=False)[:-1].reshape(num_ori, -1)  
            #2
            Acc_test_pre_2[s, pre_epoch, i, :, :] = network.test(net_2, y, num_test * 2, t_label, 
                                                          prt=False)[:-1].reshape(num_ori, -1) 
            t_data = np.roll(t_data, -5, axis=2)

    # ------------------------------Conventional Training--------------------------------------------
    _epc = 1
    for epoch in range(conventional_epoch):  # loop over the dataset multiple times
        # Generate conventional training data
        for i in range(num_batch):
            labels[i], img_tg = Img_L.gen_train()  # gen train
            inputs_img[i, :, :, :] = img_tg
        inputs = network.representation_torch(inputs_img)
        inputs = utils.norma_rep(inputs)
        inputs = network.feedforward(inputs, weight)
        inputs = utils.norma_rep(inputs)
        # update the hebbian weight
        if epoch > bg_epoch:
            weight = network.update_weight(weight, inputs, l_lambda=l_lambda)
        # update the cnn_1 weight (only the cnn_1, because conventional training is trained only on ori1 data)
        cnn_1_optimizer.zero_grad()
        b_x = torch.tensor(inputs, dtype=torch.float32)
        b_labels = torch.tensor(labels)
        net_1.train()
        b_x, b_labels = b_x.to(device), b_labels.to(device)
        outputs_1 = net_1(b_x)
        outputs_1 = outputs_1.squeeze(1)   
        loss = criterion(outputs_1, (b_labels + 1) // 2)
        # pdb.set_trace()
        loss.backward()
        cnn_1_optimizer.step()
        # print statistics
        running_loss += loss.item()
        # test during conventional training
        if epoch % show_epc == show_epc - 1:  # save every show epoch
            # Testing for different location
            for i in range(40 // loc):  
                y = network.feedforward(t_data, weight)
                y = utils.norma_rep(y)
                y = torch.tensor(y, dtype=torch.float32)
                y = y.to(device)
                t_label = t_label.to(device)
                #1
                Acc_test_1[s, _epc - 1, i, :, :] = network.test(net_1, y, num_test * 2, t_label,  
                                                         print_test)[:-1].reshape(num_ori, -1) 
                #2
                Acc_test_2[s, _epc - 1, i, :, :] = network.test(net_2, y, num_test * 2, t_label,  
                                                         print_test)[:-1].reshape(num_ori, -1)  
                t_data = np.roll(t_data, -5, axis=2)      # change the test data to the next location\
            _epc += 1

    # # save every repeat model
    # PATH = name_head + 'model/' + str(s) + '/'
    # utils.mkdir(PATH)
    # torch.save(best_net.state_dict(), PATH + 'net.pth')
    # np.save(PATH + 'weight.npy', best_weight)
    return net_1, net_2, weight, cnn_1_optimizer, cnn_2_optimizer, criterion


# ------------------------------Training Procedure-------------------------------------------------
# generate test data and get the representation
t_data, t_label = GenTestData_torch(
    _ori=ori, num=num_test, phase=phase, freq=freq, diff=diff, noise_cutout=noise_cutout, contrast=contrast) 
t_data = utils.norma_rep(t_data)
t_label = torch.tensor(t_label)

# repeat the whole experiment for many times
for s in tqdm(range(num_repeat)): 

    # ------------------------------Conventional Training--------------------------------------------
    net_1, net_2, weight, cnn_1_optimizer, cnn_2_optimizer, criterion = ff_train(
        s, t_label, t_data,
        show_epc=2,
        bg_epoch=begin_epoch,
        phase=phase, freq=freq,
        diff=diff, noise_cutout=noise_cutout, contrast=contrast
    )  

    # ------------------------------Dobule Training--------------------------------------------------
    # generate dobule training training data
    Img_DT = image.GenImg(orient='H', loc="R", diff=diff, phase=phase,
                        freq=freq, noise_cutout=noise_cutout, contrast=contrast,label_reverse=label_reverse)
    db_inputs_img = np.zeros([16, 1, 400, 200])
    db_labels = np.zeros([16])
    # dobule training for 200 times
    for db_i in range(200):
        # Testing for different location(though not used in the paper)
        for i in range(40 // loc): 
            # go through hebbian layer 
            y = network.feedforward(t_data, weight)
            # go through normalize layer
            y = utils.norma_rep(y)
            # go through the cnn network
            net_1.eval()
            net_2.eval()
            y = torch.tensor(
                y, dtype=torch.float32)
            y = y.to(device)
            t_label = t_label.to(device)
            #1
            Acc_test_db_1[s, db_i, i, :, :] = network.test(net_1, y, num_test * 2, t_label,
                                                    print_test)[:-1].reshape(num_ori, -1)
            #2
            Acc_test_db_2[s, db_i, i, :, :] = network.test(net_2, y, num_test * 2, t_label,
                                                    print_test)[:-1].reshape(num_ori, -1)
            # here change the test data to the next location
            t_data = np.roll(t_data, -5, axis=2)

        # dobule training learning procedure
        # generate dobule training training data
        for i in range(16):
            db_labels[i], img_tg = Img_DT.gen_train()
            db_inputs_img[i, :, :, :] = img_tg
        # get the representation
        activity_tg = network.representation_torch(
            db_inputs_img)
        # go through normalize layer
        inputs = utils.norma_rep(activity_tg)
        # go through hebbian layer
        inputs = network.feedforward(inputs, weight)
        # go through normalize layer
        inputs = utils.norma_rep(inputs)
        # update the hebbian weight
        weight = network.update_weight(
            weight, inputs, l_lambda=l_lambda)
        # go through the cnn network
        # here only update the cnn_2 weight (only the cnn_2, because dobule training is trained only on ori2 data)
        cnn_2_optimizer.zero_grad()
        b_x = torch.tensor(inputs, dtype=torch.float32)
        b_labels = torch.tensor(db_labels)
        net_2.train()
        b_x, b_labels = b_x.to(device), b_labels.to(device)
        outputs = net_2(b_x)
        outputs = outputs.squeeze(1)  
        # loss
        loss = criterion(outputs, (b_labels + 1) // 2)
        loss.backward()
        cnn_2_optimizer.step()
        # calculate db train acc
        db_acc = utils.np_acc(outputs, db_labels)
        Acc_traindb_2[s, db_i] = db_acc   
        Loss_traindb[s, db_i] = loss.item()

    # PATH = name_head + 'model/' + str(s) + '/'
    # utils.mkdir(PATH)
    # np.save(PATH + 'weightdb.npy', weight)

# save the results
#1
np.save(name_head + 'AccALL_1.npy', AccALL_1)
np.save(name_head + 'Acc_test_1.npy', Acc_test_1)
np.save(name_head + 'Acc_testdb_1.npy', Acc_test_db_1)
np.save(name_head + 'Acc_testpre_1.npy', Acc_test_pre_1)
np.save(name_head + 'Acc_traindb_1.npy', Acc_traindb_1)                            
np.save(name_head + 'Loss_traindb.npy', Loss_traindb)
#2
np.save(name_head + 'AccALL_2.npy', AccALL_2)
np.save(name_head + 'Acc_test_2.npy', Acc_test_2)
np.save(name_head + 'Acc_testdb_2.npy', Acc_test_db_2)
np.save(name_head + 'Acc_testpre_2.npy', Acc_test_pre_2)
np.save(name_head + 'Acc_traindb_2.npy', Acc_traindb_2)
