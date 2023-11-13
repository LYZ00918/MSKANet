import json
import torch
from torch import nn
import numpy as np
from toolbox.datasets.vaihingen import Vaihingen
from toolbox.datasets.potsdam import Potsdam
from torch.utils.data import DataLoader
from torch import optim
from datetime import datetime
from torch.autograd import Variable
from toolbox.loss.loss import MscCrossEntropyLoss, FocalLossbyothers, MscLovaszSoftmaxLoss
from toolbox.loss.sp import SP
from toolbox.loss.vid import VID
from toolbox.loss.pkt import PKT
from toolbox.loss.at import AT
from toolbox.loss.kd_losses1.fsp import FSP
from toolbox.loss.kd_losses1.ofd import OFD
from toolbox.loss.kd_losses1.TFFD import feature_transfer
from toolbox.loss.IOU import IOU
from toolbox.loss.My_loss.structure_transfer import ST_3
from toolbox.loss.My_loss.CWD import ChannelWiseDivergence
from toolbox.loss.My_loss.Att import SelfA
from toolbox.loss.My_loss.structure_transfer import transfer_AT
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.nn.functional as F
# teacher model
from toolbox.models.lyz.Module2.PPNet13 import SFAFMA_T
from toolbox.models.lyz.Module2.PPNet13_S import SFAFMA_S


'''
just test loss
'''


# DATASET = "Potsdam"
DATASET = "Vaihingen"
batch_size = 8
import argparse
parser = argparse.ArgumentParser(description="config")
parser.add_argument(
    "--config",
    nargs="?",
    type=str,
    default="configs/{}.json".format(DATASET),
    help="Configuration file to use",
)
args = parser.parse_args()
with open(args.config, 'r') as fp:
    cfg = json.load(fp)
if DATASET == "Potsdam":
    train_dataloader = DataLoader(Potsdam(cfg, mode='train'), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(Potsdam(cfg, mode='test'), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
elif DATASET == "Vaihingen":
    train_dataloader = DataLoader(Vaihingen(cfg, mode='train'), batch_size=batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
    test_dataloader = DataLoader(Vaihingen(cfg, mode='test'), batch_size=batch_size, shuffle=True, num_workers=4,
                                 pin_memory=True)
# weight = ClassWeight('median_freq_balancing')
criterion = nn.CrossEntropyLoss().cuda()
criterion1 = nn.KLDivLoss().cuda()
criterion2 = SP().cuda()
criterion3 = PKT().cuda()
criterion4 = AT(2).cuda()
criterion5 = feature_transfer().cuda()

criterion_st1 = ST_3(16,448)
criterion_st2 = ST_3(24,448)
criterion_st3 = ST_3(32,448)
criterion_st4 = ST_3(32,1280)
criterion_st5 = ST_3(160,1280)
criterion_st6 = ST_3(320,1280)

criterion_FSP = FSP().cuda()

criterion_SelfA1 = SelfA(3,3,[16,32,48],[32,64,128]).cuda()
criterion_SelfA2 = SelfA(2,2,[64,160],[256,512]).cuda()

criterion_trans_1 = transfer_AT(32,48,64,64,128,256).cuda()
criterion_trans_2 = transfer_AT(64,160,320,256,512,512).cuda()

iou = IOU()
criterion_without = MscCrossEntropyLoss().cuda()
# criterion1 = nn.CrossEntropyLoss(weight=weight.get_weight(test_dataloader, num_classes=5)).cuda()

criterion_focal1 = FocalLossbyothers().cuda()
criterion_Lovasz = MscLovaszSoftmaxLoss().cuda()
criterion_bce = nn.BCELoss().cuda()  # 边界监督
net_s = SFAFMA_S().cuda()
net_T = SFAFMA_T().cuda()
net_T.load_state_dict(torch.load('./weight/PPNet_12-Vaihingen-loss.pth'))
optimizer = optim.Adam(net_s.parameters(), lr=1e-4, weight_decay=5e-4)



def accuary(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size
best = [0.0]
size = (56, 56)
numloss = 0
nummae = 0
trainlosslist_nju = []
vallosslist_nju = []
iter_num = len(train_dataloader)
epochs = 200
# schduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # setting the learning rate desend starage
for epoch in range(epochs):
    if epoch % 20 == 0 and epoch != 0:  # setting the learning rate desend starage
        for group in optimizer.param_groups:
            group['lr'] = 0.1 * group['lr']
    # for group in optimizer.param_groups:
    # 	group['lr'] *= 0.99
    # 	print(group['lr'])
    train_loss = 0
    net = net_s.train()
    prec_time = datetime.now()
    alpha = 0.90
    for i, sample in enumerate(train_dataloader):
        image = Variable(sample['image'].cuda())  # [2, 3, 256, 256]
        ndsm = Variable(sample['dsm'].cuda())  # [2, 1, 256, 256]
        label = Variable(sample['label'].long().cuda())  # [2, 256, 256]
        # edge = Variable(sample['edge'].float().cuda())  # 边界监督 [12, 256, 256]

        # image = F.interpolate(image, (320, 320), mode="bilinear", align_corners=True)
        # ndsm = F.interpolate(ndsm, (320, 320), mode="bilinear", align_corners=True)
        # label = F.interpolate(label.unsqueeze(1).float(), (320, 320), mode="bilinear", align_corners=True).squeeze(1).long()
        # edge = F.interpolate(edge.unsqueeze(1), (224, 224), mode="bilinear", align_corners=True)

        # [12, 1, 256, 256]
        # print(edge.shape)
        # label_student = F.interpolate(label.unsqueeze(1).float(), size=size).squeeze(1).long().cuda()
        # teacher, student = net(image, ndsm)
        # ndsm = torch.repeat_interleave(ndsm, 3, dim=1)
        out = net(image, ndsm)
        out1 = net_T(image,ndsm)
        # with teacher transformer to label
        teacher_out = out1[0].data.cpu().numpy()
        teacher_out = np.argmax(teacher_out,axis=1)
        teacher_out = torch.from_numpy(teacher_out)
        teacher_out = Variable(teacher_out.long().cuda())
        loss0 = criterion_without(out[0],teacher_out)
        # loss calculate
        loss_avg = criterion_without(out[0:5],label)

        # loss mid
        # s1 = nn.Sequential(nn.Conv2d(32,224, kernel_size=1)).cuda()
        # s2 = nn.Sequential(nn.Conv2d(48,224, kernel_size=1)).cuda()
        # s3 = nn.Sequential(nn.Conv2d(64,224, kernel_size=1)).cuda()
        # loss1 = criterion_st_1(out1[10],out1[11],out[10])
        # loss2 = criterion_st_2(out1[10],out1[11],out1[12],out[11])
        # loss3 = criterion_st_3(out1[11],out1[12],out[12])
        # loss_cos = (loss3 + loss2 + loss1) / 3
        # print('loss_cos',loss_cos)

        # FSP loss
        # s_4 = nn.Sequential(nn.Conv2d(160, 512, kernel_size=1)).cuda()
        # s_5 = nn.Sequential(nn.Conv2d(320,512, kernel_size=1)).cuda()
        # S_4 = s_4(out1[13])
        # S_5 = s_5(out1[14])
        # loss_fsp = criterion_FSP(S_4,S_5,out[13],out[14])
        loss4 = criterion4(out[0],out1[0])
        loss5 = criterion4(out[1],out1[1])
        loss6 = criterion4(out[2],out1[2])
        loss7 = criterion4(out[3],out1[3])
        loss8 = criterion4(out[4],out1[4])
        loss_at = (loss4 + loss5 + loss6 + loss7 + loss8) / 5
        # print(loss_sp)
        t = 2
        # tout1 = F.softmax(out1[0]/t,dim=1)
        tout2 = F.softmax(out1[1]/t,dim=1)
        tout3 = F.softmax(out1[2]/t,dim=1)
        tout4 = F.softmax(out1[3]/t,dim=1)
        tout5 = F.softmax(out1[4]/t,dim=1)
        # sout1 = F.softmax(out[0]/t, dim=1)
        sout2 = F.softmax(out[1]/t, dim=1)
        sout3 = F.softmax(out[2]/t, dim=1)
        sout4 = F.softmax(out[3]/t, dim=1)
        sout5 = F.softmax(out[4]/t, dim=1)
        # loss1 = criterion1(sout1, tout1) * t * t  # 没有边界监督
        loss2 = criterion1(sout2, tout2) * t * t  # 没有边界监督
        loss3 = criterion1(sout3, tout3) * t * t  # 没有边界监督
        loss4 = criterion1(sout4, tout4) * t * t  # 没有边界监督
        loss5 = criterion1(sout5, tout5) * t * t  # 没有边界监督
        loss_KD = (loss5 + loss4 + loss3 + loss2)/4

        # loss9 = criterion_trans_1(out[10],out[11],out[12],out1[10],out1[11],out1[12])
        # loss10 = criterion_trans_2(out[12],out[13],out[14],out1[12],out1[13],out1[14])
        # loss_trans = (loss10 + loss9) / 2
        # print(loss_trans)
        # loss9 = criterion_SelfA1(out[5:8],out1[5:8])
        # loss10 = criterion_SelfA2(out[8:10],out1[8:10])
        # loss_selfA = (loss9 + loss10) / 2
        # print(loss_selfA)

        # s_1 = nn.Sequential(nn.Conv2d(16,32, kernel_size=1)).cuda()
        # # s_1_1 = nn.Sequential(nn.Conv2d(32,64, kernel_size=1)).cuda()
        # s_2 = nn.Sequential(nn.Conv2d(32,64, kernel_size=1)).cuda()
        # # s_2_2 = nn.Sequential(nn.Conv2d(48,128, kernel_size=1)).cuda()
        # s_3 = nn.Sequential(nn.Conv2d(48,128, kernel_size=1)).cuda()
        # # s_3_3 = nn.Sequential(nn.Conv2d(64,256, kernel_size=1)).cuda()
        # s_4 = nn.Sequential(nn.Conv2d(64,256, kernel_size=1)).cuda()
        # # s_4_4 = nn.Sequential(nn.Conv2d(160,512, kernel_size=1)).cuda()
        # s_5 = nn.Sequential(nn.Conv2d(160,512, kernel_size=1)).cuda()
        # # s_5_5 = nn.Sequential(nn.Conv2d(320,512, kernel_size=1)).cuda()
        # #
        # a1 = s_1(out[5])
        # a2 = s_2(out[6])
        # a3 = s_3(out[7])
        # a4 = s_4(out[8])
        # a5 = s_5(out[9])
        # #
        # loss7 = criterion5(a1,out1[5])
        # loss8 = criterion5(a2,out1[6])
        # loss9 = criterion5(a3,out1[7])
        # loss10 = criterion5(a4,out1[8])
        # loss11 = criterion5(a5,out1[9])
        # loss_all = (loss7 + loss8 + loss9 + loss10 + loss11)/5
        # a1_1 = s_1_1(out[10])
        # a2_2 = s_2_2(out[11])
        # a3_3 = s_3_3(out[12])
        # a4_4 = s_4_4(out[13])
        # a5_5 = s_5_5(out[14])
        # loss12 = criterion5_1(a1_1,out1[10])
        # loss13 = criterion5_2(a2_2,out1[11])
        # loss14 = criterion5_3(a3_3,out1[12])
        # loss15 = criterion5_4(a4_4,out1[13])
        # loss16 = criterion5_5(a5_5,out1[14])
        # loss_vid = (loss16 + loss15 + loss14 + loss13 + loss12)/5

        loss = loss_avg + loss0 + loss_KD * 0.1 + loss_at


        # 边界监督
        # loss1 = criterion_without(out[0], label)
        # loss2 = criterion_bce(nn.Sigmoid()(out[1]), edge)
        # loss = (loss2 + loss1) / 2
        # 边界监督

        print('Training: Iteration {:4}'.format(i), 'Loss:', loss.item())
        if (i+1) % 100 == 0:
            print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %5.4f' % (
                epoch+1, epochs, i+1, iter_num, train_loss / 100))
            train_loss = 0

        optimizer.zero_grad()

        loss.backward()  # backpropagation to get gradient
        # qichuangaaaxuexi
        optimizer.step()  # update the weight

        train_loss = loss.item() + train_loss

    net = net_s.eval()
    eval_loss = 0
    acc = 0
    with torch.no_grad():
        for j, sampleTest in enumerate(test_dataloader):
            imageVal = Variable(sampleTest['image'].float().cuda())
            ndsmVal = Variable(sampleTest['dsm'].float().cuda())
            labelVal = Variable(sampleTest['label'].long().cuda())
            # imageVal = F.interpolate(imageVal, (320, 320), mode="bilinear", align_corners=True)
            # ndsmVal = F.interpolate(ndsmVal, (320, 320), mode="bilinear", align_corners=True)
            # labelVal = F.interpolate(labelVal.unsqueeze(1).float(), (320, 320),
            #                          mode="bilinear", align_corners=True).squeeze(1).long()
            # ndsmVal = torch.repeat_interleave(ndsmVal, 3, dim=1)
            # teacherVal, studentVal = net(imageVal, ndsmVal)
            # outVal = net(imageVal)
            outVal = net(imageVal, ndsmVal)
            loss = criterion_without(outVal[0:5], labelVal)
            outVal = outVal[0].max(dim=1)[1].data.cpu().numpy()
            # outVal = outVal[1].max(dim=1)[1].data.cpu().numpy()
            labelVal = labelVal.data.cpu().numpy()
            accval = accuary(outVal, labelVal)
            # print('accVal:', accval)
            print('Valid:    Iteration {:4}'.format(j), 'Loss:', loss.item())
            eval_loss = loss.item() + eval_loss
            acc = acc + accval

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    epoch_str = ('Epoch: {}, Train Loss: {:.5f},Valid Loss: {:.5f},Valid Acc: {:.5f}'.format(
        epoch, train_loss / len(train_dataloader), eval_loss / len(test_dataloader), acc / len(test_dataloader)))
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print(epoch_str + time_str)

    trainlosslist_nju.append(train_loss / len(train_dataloader))
    vallosslist_nju.append(eval_loss / len(test_dataloader))

    if acc / len(test_dataloader) >= max(best):
        best.append(acc / len(test_dataloader))
        numloss = epoch
        torch.save(net.state_dict(), './weight/PPNet_S_KD(CE[S,T]_AT_KL))-{}-loss.pth'.format(DATASET))


    print(max(best), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   Accuracy', numloss)

# PPNet_S_KD(CE[S,T]_AT_KL+selfA))
# PPNet_S_KD(CE[S,T]_KL))