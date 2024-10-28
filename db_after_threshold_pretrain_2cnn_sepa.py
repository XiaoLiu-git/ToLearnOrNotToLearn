from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
# import offline_Hebb_23_4 as ob
import utils
import network
import image

# global Acc_test
day = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
foldername = './' + day
utils.mkdir(foldername)
num_section = 100
pre_epoch_size = 20
num_test = 20
ori = 22.5
num_ori = int(180 / ori)
loc = 5
label_reverse = True
weight_decay_pre = 0
weight_decay = 0


conventional_epoch = 10*20
AccALL_1 = np.zeros([num_section, 1+conventional_epoch//2, 5])
LossALL = np.zeros([num_section, conventional_epoch//2, 1])
Acc_test_1 = np.zeros([num_section, conventional_epoch//2, int(40 / loc), int(180 / ori), 10])

Acc_test_db_1 = np.zeros([num_section, 200, int(40 / loc), int(180 / ori), 10])
Acc_test_pre_1 = np.zeros(
    [num_section, pre_epoch_size, int(40 / loc), int(180 / ori), 10])
Acc_traindb_1 = np.zeros([num_section, 200])
Loss_traindb = np.zeros([num_section, 200])
print_test = False
l_lambda = .3
cnn_lr = 1e-3
cnn_lr_pre = 1e-3
#2
AccALL_2 = np.zeros([num_section, 1+conventional_epoch//2, 5])
Acc_test_2 = np.zeros([num_section, conventional_epoch//2, int(40 / loc), int(180 / ori), 10])
Acc_test_db_2 = np.zeros([num_section, 200, int(40 / loc), int(180 / ori), 10])
Acc_test_pre_2 = np.zeros(
    [num_section, pre_epoch_size, int(40 / loc), int(180 / ori), 10])
Acc_traindb_2 = np.zeros([num_section, 200])



# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

# 在每个位置每种角度都测试了
# 每种难度（10种）
# 在正常训练和db后都有

def GenTestData_torch(_ori, num=10, loc="L", diff=0, phase=0, freq=0.02, noise_cutout=0.7, var_noise=1, contrast=0.6):
    num_ori = int(180 / _ori)
    img_dataset = np.zeros([num * 2 * num_ori * 10, 1, 400, 200])
    label = np.zeros([num * 2 * num_ori * 10])
    # generate test dataset
    for i in range(num_ori):
        ori_i = i * _ori
        for diff in range(10):
            Img = image.GenImg(orient=ori_i, loc=loc, diff=diff, phase=phase, freq=freq,
                noise_cutout=noise_cutout, var_noise=var_noise, contrast=contrast)  # 类实例化
            if i == 6:
                if label_reverse:
                    Img = image.GenImg(orient=ori_i, loc=loc, diff=diff, phase=phase, freq=freq,
                            noise_cutout=noise_cutout, var_noise=var_noise, contrast=contrast,label_reverse=label_reverse)  # 类实例化
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


def ff_train(s, t_label, t_data, show_epc, net_name, slow_learning=True, bg_epoch=0, phase=0, freq=0.02, diff=0, noise_cutout=0.7, contrast=0.6):
    """training

    Args:
        show_epc (int): Every what epoch to record acc & loss
        net_name (string): Which net
        slow_learning (bool, optional): Here no use. Defaults to True.
        bg_epoch (int, optional): Hebb learning start epoch (to change weight). Defaults to 0.
        phase (int, optional): The gabor phase ([radians]). Defaults to 0.
        freq (float, optional): The gabor frequency. Defaults to 0.02.
        noise_cutout (float, optional): Noise of all images. Defaults to 0.7.

    Returns:
        acc_list([200 // show_epc + 1, 5]): Acc of every show epoch and the first epoch of training and other 4 sets of validation set.
        loss_list([200 // show_epc, 1]): Training loss of every show epoch.
        net: The lastest net, not the best net. 
        weight([40, 18]): Lastest Hebb weight. 

    """
    running_loss = 0.0
    # network    在train里面新建net
    net_1 = network.Net_sCC().to(device)
    net_2 = network.Net_sCC().to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam([
    #     {'params': net.layer2_1.parameters()},
    #     {'params': net.readout_1.parameters()},
    #     {'params': net.layer1_1.parameters(), 'lr': cnn_lr},
    #     {'params': net.layer2_2.parameters()},
    #     {'params': net.readout_2.parameters()},
    #     {'params': net.layer1_2.parameters(), 'lr': cnn_lr}

    # ], lr=1e-3)
    cnn_1_optimizer_pre = optim.Adam([
        {'params': net_1.layer2.parameters()},
        {'params': net_1.readout.parameters()},
        {'params': net_1.layer1.parameters(), 'lr': cnn_lr_pre}
    ], lr=cnn_lr_pre,weight_decay=weight_decay_pre)
    cnn_2_optimizer_pre = optim.Adam([
        {'params': net_2.layer2.parameters()},
        {'params': net_2.readout.parameters()},
        {'params': net_2.layer1.parameters(), 'lr': cnn_lr_pre}
    ], lr=cnn_lr_pre,weight_decay=weight_decay_pre)

    cnn_1_optimizer = optim.Adam([
        {'params': net_1.layer2.parameters()},
        {'params': net_1.readout.parameters()},
        {'params': net_1.layer1.parameters(), 'lr': cnn_lr}
    ], lr=cnn_lr,weight_decay=weight_decay)
    cnn_2_optimizer = optim.Adam([
        {'params': net_2.layer2.parameters()},
        {'params': net_2.readout.parameters()},
        {'params': net_2.layer1.parameters(), 'lr': cnn_lr}
    ], lr=cnn_lr,weight_decay=weight_decay)

    # input
    num_batch = 16
    Img_L = image.GenImg(orient='V', loc="L", diff=diff, phase=phase,
                      freq=freq, noise_cutout=noise_cutout, contrast=contrast)
    Img_R = image.GenImg(orient='V', loc="R", diff=diff, phase=phase,
                      freq=freq, noise_cutout=noise_cutout, contrast=contrast)
    Img_DT = image.GenImg(orient='H', loc="R", diff=diff, phase=phase,
                       freq=freq, noise_cutout=noise_cutout, contrast=contrast,label_reverse=label_reverse)
    Img_th = image.GenImg(orient='H', loc="L", diff=diff, phase=phase,
                       freq=freq, noise_cutout=noise_cutout, contrast=contrast,label_reverse=label_reverse)
    inputs_img = np.zeros([num_batch, 1, 400, 200])  # training inputs 仍然有1
    labels = np.zeros([num_batch])

    num_batch_pre = num_batch*4
    inputs_img_pre = np.zeros(
        [num_batch_pre, 1, 400, 200])  # training inputs 仍然有1
    labels_pre = np.zeros([num_batch_pre])

    # generate testing data, select 40 < L,R,TH < 60
    # 因为有起始，所以+1;5是一个train加4个loc的test
    acc_list_1 = np.zeros([conventional_epoch // show_epc + 1, 5])
    acc_list_2 = np.zeros([conventional_epoch // show_epc + 1, 5])
    loss_list = np.zeros([conventional_epoch // show_epc, 1])  # 没有加起始，training loss
    num_test = 20  # 每个位置10个pair,20个trial

    # 所有t_input,第一维对应input的batch index
    t_inputs_img = np.zeros([num_test * 4, 1, 400, 200])
    t_labels = np.zeros([num_test * 4])
    for i in range(num_test // 2):  # 按照pair生成，trial//2
        # loc1        
        img_tg_p, img_tg_n = Img_L.gen_test()
        t_labels[2 * i] = img_tg_p[0]
        t_labels[2 * i] = img_tg_n[0]                
        t_inputs_img[2 * i, :, :, :] = img_tg_p[1]
        t_inputs_img[2 * i + 1, :, :, :] = img_tg_n[1]
        # loc2
        img_tg_p, img_tg_n = Img_DT.gen_test()
        t_labels[1 * num_test + 2 * i ] = img_tg_p[0]
        t_labels[1 * num_test + 2 * i + 1] = img_tg_n[0]                
        t_inputs_img[1 * num_test + 2 * i, :, :, :] = img_tg_p[1]
        t_inputs_img[1 * num_test + 2 * i + 1, :, :, :] = img_tg_n[1]
        # loc3
        img_tg_p, img_tg_n = Img_R.gen_test()
        t_labels[2 * num_test + 2 * i ] = img_tg_p[0]
        t_labels[2 * num_test + 2 * i + 1] = img_tg_n[0]                
        t_inputs_img[2 * num_test + 2 * i, :, :, :] = img_tg_p[1]
        t_inputs_img[2 * num_test + 2 * i + 1, :, :, :] = img_tg_n[1]
        # loc4
        img_tg_p, img_tg_n = Img_th.gen_test()
        t_labels[3 * num_test + 2 * i ] = img_tg_p[0]
        t_labels[3 * num_test + 2 * i + 1] = img_tg_n[0]                
        t_inputs_img[3 * num_test + 2 * i, :, :, :] = img_tg_p[1]
        t_inputs_img[3 * num_test + 2 * i + 1, :, :, :] = img_tg_n[1]
    t_inputs = network.representation_torch(t_inputs_img)
    _epc = 1
    t_inputs = utils.norma_rep(t_inputs)

    # # Test initial   #这里简省了feedforward，因为初始都是1
    # net_1.eval()
    # net_2.eval()
    # #1
    # t_inputs_ff = torch.tensor(t_inputs, dtype=torch.float32)
    # t_labels = torch.tensor(t_labels)
    # t_inputs_ff, t_labels = t_inputs_ff.to(device), t_labels.to(device)
    # acc_list_1[0, 1:] = network.test(net_1, t_inputs_ff, num_test, t_labels)[
    #     :-1]  # 这里test
    # acc_list_1[0, 0] = acc_list_1[0, 1]  # 前两维一样。train和第一个位置的test。注意非初始epoch是否一样
    # best_acc = np.zeros(np.size(acc_list_1[0, :]))  # [5]train和4个test的best
    # #2
    # acc_list_2[0, 1:] = network.test(net_2, t_inputs_ff, num_test, t_labels)[
    # :-1]  # 这里test
    # acc_list_2[0, 0] = acc_list_2[0, 1]  # 前两维一样。train和第一个位置的test。注意非初始epoch是否一样
    # best_acc = np.zeros(np.size(acc_list_2[0, :]))  # [5]train和4个test的best

    # Pre train
    weight = np.ones([40, 1])
    inputs_img_pre = np.zeros(
        [num_batch*4, 1, 400, 200])  # training inputs 仍然有1
    labels_pre = np.zeros([num_batch*4])
    for pre_epoch in range(pre_epoch_size):
        # 一起batch
        for i in range(num_batch):
            labels_pre[i], img_tg = Img_L.gen_train()  # gen train
            inputs_img_pre[i, :, :, :] = img_tg
            labels_pre[1*num_batch+i], img_tg = Img_DT.gen_train()  # gen train
            inputs_img_pre[1*num_batch+i, :, :, :] = img_tg
            labels_pre[2*num_batch+i], img_tg = Img_R.gen_train()  # gen train
            inputs_img_pre[2*num_batch+i, :, :, :] = img_tg
            labels_pre[3*num_batch+i], img_tg = Img_th.gen_train()  # gen train
            inputs_img_pre[3*num_batch+i, :, :, :] = img_tg
        # # 打乱batch
        # for i in range(num_batch//2):
        #     labels_pre[2*i], img_tg = Img_L.gen_train()  # gen train
        #     inputs_img_pre[2*i, :, :, :] = img_tg
        #     labels_pre[1+2*i], img_tg = Img_R.gen_train()  # gen train
        #     inputs_img_pre[1+2*i, :, :, :] = img_tg

        #     labels_pre[1*num_batch+2*i], img_tg = Img_DT.gen_train()  # gen train
        #     inputs_img_pre[1*num_batch+2*i, :, :, :] = img_tg
        #     labels_pre[1*num_batch+2*i+1], img_tg = Img_th.gen_train()  # gen train
        #     inputs_img_pre[1*num_batch+2*i+1, :, :, :] = img_tg

        #     labels_pre[2*num_batch+2*i], img_tg = Img_L.gen_train()  # gen train
        #     inputs_img_pre[2*num_batch+2*i, :, :, :] = img_tg
        #     labels_pre[2*num_batch+2*i+1], img_tg = Img_R.gen_train()  # gen train
        #     inputs_img_pre[2*num_batch+2*i+1, :, :, :] = img_tg

        #     labels_pre[3*num_batch+2*i], img_tg = Img_DT.gen_train()  # gen train
        #     inputs_img_pre[3*num_batch+2*i, :, :, :] = img_tg
        #     labels_pre[3*num_batch+2*i+1], img_tg = Img_th.gen_train()  # gen train
        #     inputs_img_pre[3*num_batch+2*i+1, :, :, :] = img_tg

        # 每个batch分别bp
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
            # optimizer.zero_grad()

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
            outputs = outputs.squeeze(1)   # 这里squeeze掉了1维 [16, 1]->[16]
            loss = criterion(outputs, (b_labels + 1) // 2)
            # pdb.set_trace()
            loss.backward()
            if pre_epoch_//2==0:
                cnn_1_optimizer_pre.step()
            else:
                cnn_2_optimizer_pre.step()
            # optimizer.step()

        # # 一起bp
        # inputs = ob.representation_torch(
        #     inputs_img_pre)
        # inputs = utils.norma_rep(inputs)
        # inputs = network.feedforward(inputs, weight)
        # inputs = utils.norma_rep(inputs)
        # weight = network.update_weight(weight, inputs, l_lambda=l_lambda)

        # optimizer.zero_grad()
        # b_x = torch.tensor(inputs, dtype=torch.float32)
        # b_labels = torch.tensor(
        #     labels_pre)
        # net.train()
        # b_x, b_labels = b_x.to(device), b_labels.to(device)
        # outputs = net(b_x)
        # outputs = outputs.squeeze(1)   # 这里squeeze掉了1维 [16, 1]->[16]
        # loss = criterion(outputs, (b_labels + 1) // 2)
        # # pdb.set_trace()
        # loss.backward()
        # optimizer.step()

        # Pre testing for different location
        net_1.eval()
        net_2.eval()
        t_label = torch.tensor(t_label)
        for i in range(40 // loc):  # 因为t_data里是所有ori，但是是默认L loc的  #这有8个
            y = network.feedforward(t_data, weight)
            y = utils.norma_rep(y)
            y = torch.tensor(y, dtype=torch.float32)
            # t_label = torch.tensor(t_label)
            y = y.to(device)
            t_label = t_label.to(device)
            #1
            Acc_test_pre_1[s, pre_epoch, i, :, :] = network.test(net_1, y, num_test * 2, t_label,  # 同位置按角度分batch也是10*2，ff_train中按set分也是10*2  #(不好，这里numtest写死了)
                                                          prt=False)[:-1].reshape(num_ori, -1)  # 输入是原始representation经过hebb weight得到的feature map
            #2
            Acc_test_pre_2[s, pre_epoch, i, :, :] = network.test(net_2, y, num_test * 2, t_label,  # 同位置按角度分batch也是10*2，ff_train中按set分也是10*2  #(不好，这里numtest写死了)
                                                          prt=False)[:-1].reshape(num_ori, -1)  # 输入是原始representation经过hebb weight得到的feature map
            # 更改t_data到下一个loc。但这个loc不能等同于location，而是feature上L的5个单位
            t_data = np.roll(t_data, -5, axis=2)

    # Test initial   #这里简省了feedforward，因为初始都是1
    net_1.eval()
    net_2.eval()
    #1
    t_inputs_ff = torch.tensor(t_inputs, dtype=torch.float32)
    t_labels = torch.tensor(t_labels)
    t_inputs_ff, t_labels = t_inputs_ff.to(device), t_labels.to(device)
    acc_list_1[0, 1:] = network.test(net_1, t_inputs_ff, num_test, t_labels)[
        :-1]  # 这里test
    acc_list_1[0, 0] = acc_list_1[0, 1]  # 前两维一样。train和第一个位置的test。注意非初始epoch是否一样
    best_acc = np.zeros(np.size(acc_list_1[0, :]))  # [5]train和4个test的best
    #2
    acc_list_2[0, 1:] = network.test(net_2, t_inputs_ff, num_test, t_labels)[
    :-1]  # 这里test
    acc_list_2[0, 0] = acc_list_2[0, 1]  # 前两维一样。train和第一个位置的test。注意非初始epoch是否一样
    best_acc = np.zeros(np.size(acc_list_2[0, :]))  # [5]train和4个test的best
    
    # Training
    for epoch in range(conventional_epoch):  # loop over the dataset multiple times
        # Generate a batch representation
        for i in range(num_batch):
            labels[i], img_tg = Img_L.gen_train()  # gen train
            # inputs_img[i, :, :, :] = ob.representation(img_tg)
            inputs_img[i, :, :, :] = img_tg
        inputs = network.representation_torch(inputs_img)
        inputs = utils.norma_rep(inputs)
        inputs = network.feedforward(inputs, weight)
        inputs = utils.norma_rep(inputs)

        if epoch > bg_epoch:
            weight = network.update_weight(weight, inputs, l_lambda=l_lambda)

        cnn_1_optimizer.zero_grad()
        b_x = torch.tensor(inputs, dtype=torch.float32)
        b_labels = torch.tensor(labels)
        net_1.train()
        b_x, b_labels = b_x.to(device), b_labels.to(device)
        outputs_1 = net_1(b_x)
        outputs_1 = outputs_1.squeeze(1)   # 这里squeeze掉了1维 [16, 1]->[16]
        loss = criterion(outputs_1, (b_labels + 1) // 2)
        # pdb.set_trace()
        loss.backward()
        cnn_1_optimizer.step()
        #2
        outputs_2 = net_2(b_x)

        # print statistics
        running_loss += loss.item()
        if epoch % show_epc == show_epc - 1:  # save every show epoch
            print_test = False
            acc_1 = utils.np_acc(outputs_1, labels)
            acc_list_1[_epc, 0] = acc_1   # train acc
            loss_list[_epc - 1, 0] = running_loss / \
                show_epc  # show epoch interval中产生所有loss的均值
            #2
            acc_2 = utils.np_acc(outputs_2, labels)
            acc_list_2[_epc, 0] = acc_2  # train acc
            # if epoch % (show_epc * 10) == show_epc * 10 - 1:   # print every 10* show epoch
            #     print('[%d] loss: %.6f' %
            #           (epoch + 1, running_loss / show_epc))
            #     print('train acc: %.2f %%' % (acc * 100))
            #     print_test = True
            # PATH = './net.pth'
            # torch.save(net.state_dict(), PATH)
            # validation
            running_loss = 0.0
            t_input_ff = network.feedforward(t_inputs, weight)
            t_input_ff = utils.norma_rep(t_input_ff)

            net_1.eval()
            net_2.eval()
            t_input_ff = torch.tensor(t_input_ff, dtype=torch.float32)
            t_labels = torch.tensor(t_labels)
            t_input_ff, t_labels = t_input_ff.to(device), t_labels.to(device)
            #1
            acc_test = network.test(net_1, t_input_ff, num_test,
                               t_labels, print_test)  # best for what

            acc_list_1[_epc, 1:] = acc_test[:-1]
            if acc_test[-1] > best_acc[-1]:  # 这里validation用了四个loc的acc
                best_net = net_1
                best_weight = weight
                best_acc = acc_test
            #2
            acc_test = network.test(net_2, t_input_ff, num_test,
                    t_labels, print_test)  # best for what

            acc_list_2[_epc, 1:] = acc_test[:-1]

            # Testing for different location

            for i in range(40 // loc):  # 因为t_data里是所有ori，但是是默认L loc的  #这有8个
                y = network.feedforward(t_data, weight)
                y = utils.norma_rep(y)
                y = torch.tensor(y, dtype=torch.float32)
                t_label = torch.tensor(t_label)
                y = y.to(device)
                t_label = t_label.to(device)
                #1
                Acc_test_1[s, _epc - 1, i, :, :] = network.test(net_1, y, num_test * 2, t_label,  # 同位置按角度分batch也是10*2，ff_train中按set分也是10*2  #(不好，这里numtest写死了)
                                                         print_test)[:-1].reshape(num_ori, -1)  # 输入是原始representation经过hebb weight得到的feature map
                #2
                Acc_test_2[s, _epc - 1, i, :, :] = network.test(net_2, y, num_test * 2, t_label,  # 同位置按角度分batch也是10*2，ff_train中按set分也是10*2  #(不好，这里numtest写死了)
                                                         print_test)[:-1].reshape(num_ori, -1)  # 输入是原始representation经过hebb weight得到的feature map
                # 更改t_data到下一个loc。但这个loc不能等同于location，而是feature上L的5个单位
                t_data = np.roll(t_data, -5, axis=2)

            _epc += 1

    PATH = name_head + 'model/' + str(s) + '/'
    utils.mkdir(PATH)
    torch.save(best_net.state_dict(), PATH + 'net.pth')
    np.save(PATH + 'weight.npy', best_weight)

    return acc_list_1,acc_list_2, loss_list, net_1,net_2, weight,cnn_1_optimizer,cnn_2_optimizer,criterion


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

for net_name in ['s_cc3']:
    # for diff in [1,3,5,7,9]:       #,1,2]:
    for phase_8 in range(1):  # 8
        phase = phase_8*np.pi/8
        for freq in [0.02]:
            # for diff in [0,1,2]:
            for noise_cutout in [2]:
                for contrast in [0.6]:
                    for diff in [9]:
                        np.random.seed(2021)
                        t_data, t_label = GenTestData_torch(
                            _ori=ori, num=num_test, phase=phase, freq=freq, diff=diff, noise_cutout=noise_cutout, contrast=contrast)  # 改动
                        np.random.seed(None)
                        # 原代码 (160, 1, 40, 18)先reshape为(160, 720)，再整体normalize，再reshape回(160, 1, 40, 18)
                        t_data = utils.norma_rep(t_data)
                        for begin_epoch in [20]:
                            name_head = foldername + '/db_af/05/' + net_name + '/' + str(phase_8) \
                                + '/' + str(freq) + '/' + str(diff) + '/' + str(noise_cutout) \
                                + '/' + str(contrast) + '/' + \
                                str(begin_epoch) + '/'
                            utils.mkdir(name_head)
                            for s in range(num_section):  # 一组参数训练的模型数量
                                # Training
                                AccALL_1[s, :, :],AccALL_2[s, :, :], LossALL[s, :, :], net_1,net_2, weight ,cnn_1_optimizer,cnn_2_optimizer,criterion = ff_train(
                                    s, t_label, t_data,
                                    show_epc=2,
                                    net_name=net_name,
                                    slow_learning=False,
                                    bg_epoch=begin_epoch,
                                    phase=phase, freq=freq,
                                    diff=diff, noise_cutout=noise_cutout, contrast=contrast
                                )  # 改动
                                # net = net.cpu()

                                # Dobule training
                                Img_DT = image.GenImg(orient='H', loc="R", diff=diff, phase=phase,
                                                   freq=freq, noise_cutout=noise_cutout, contrast=contrast,label_reverse=label_reverse)
                                db_inputs_img = np.zeros([16, 1, 400, 200])
                                db_labels = np.zeros([16])
                                for db_i in range(200):
                                    for i in range(40 // loc):  # 先test再更新，可以和前面的相连
                                        y = network.feedforward(t_data, weight)
                                        y = utils.norma_rep(y)
                                        net_1.eval()
                                        net_2.eval()
                                        y = torch.tensor(
                                            y, dtype=torch.float32)
                                        y = y.to(device)
                                        t_label = torch.tensor(
                                            t_label, dtype=torch.float32)
                                        t_label = t_label.to(device)
                                        #1
                                        Acc_test_db_1[s, db_i, i, :, :] = network.test(net_1, y, num_test * 2, t_label,
                                                                                print_test)[:-1].reshape(num_ori, -1)
                                        #2
                                        Acc_test_db_2[s, db_i, i, :, :] = network.test(net_2, y, num_test * 2, t_label,
                                                                                print_test)[:-1].reshape(num_ori, -1)
                                        t_data = np.roll(t_data, -5, axis=2)

                                    for i in range(16):
                                        db_labels[i], img_tg = Img_DT.gen_train()
                                        db_inputs_img[i, :, :, :] = img_tg
                                    activity_tg = network.representation_torch(
                                        db_inputs_img)
                                    inputs = utils.norma_rep(activity_tg)
                                    inputs = network.feedforward(inputs, weight)
                                    inputs = utils.norma_rep(inputs)
                                    weight = network.update_weight(
                                        weight, inputs, l_lambda=l_lambda)
                                    cnn_2_optimizer.zero_grad()
                                    b_x = torch.tensor(inputs, dtype=torch.float32)
                                    b_labels = torch.tensor(db_labels)
                                    net_2.train()
                                    b_x, b_labels = b_x.to(device), b_labels.to(device)
                                    outputs = net_2(b_x)
                                    outputs = outputs.squeeze(1)   # 这里squeeze掉了1维 [16, 1]->[16]
                                    loss = criterion(outputs, (b_labels + 1) // 2)
                                    # pdb.set_trace()
                                    loss.backward()
                                    cnn_2_optimizer.step()
                                    # db train acc
                                    db_acc = utils.np_acc(outputs, db_labels)
                                    Acc_traindb_2[s, db_i] = db_acc   
                                    Loss_traindb[s, db_i] = loss.item()

                                PATH = name_head + 'model/' + str(s) + '/'
                                utils.mkdir(PATH)
                                np.save(PATH + 'weightdb.npy', weight)

                            # train中的training+4组validation
                            np.save(name_head + 'AccALL_1.npy', AccALL_1)
                            # training loss，无validation loss
                            np.save(name_head + 'LossALL.npy', LossALL)
                            # 全面的所有的test acc，包括上面的4组
                            np.save(name_head + 'Acc_test_1.npy', Acc_test_1)
                            # db时全面的所有的test acc，包括上面的4组
                            np.save(name_head + 'Acc_testdb_1.npy', Acc_test_db_1)
                            # pretrain时全面的所有的test acc，包括上面的4组
                            np.save(name_head + 'Acc_testpre_1.npy', Acc_test_pre_1)
                            # db时train acc
                            np.save(name_head + 'Acc_traindb_1.npy', Acc_traindb_1)                            
                            # db时train loss
                            np.save(name_head + 'Loss_traindb.npy', Loss_traindb)

                            #2
                            # train中的training+4组validation
                            np.save(name_head + 'AccALL_2.npy', AccALL_2)
                            # 全面的所有的test acc，包括上面的4组
                            np.save(name_head + 'Acc_test_2.npy', Acc_test_2)
                            # db时全面的所有的test acc，包括上面的4组
                            np.save(name_head + 'Acc_testdb_2.npy', Acc_test_db_2)
                            # pretrain时全面的所有的test acc，包括上面的4组
                            np.save(name_head + 'Acc_testpre_2.npy', Acc_test_pre_2)
                            # db时train acc
                            np.save(name_head + 'Acc_traindb_2.npy', Acc_traindb_2)
