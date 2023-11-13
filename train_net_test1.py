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

from toolbox.loss.IOU import IOU
from toolbox.loss.My_loss.Att import SelfA
from toolbox.loss.My_loss.structure_transfer import ST_3
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.nn.functional as F
# teacher model
from toolbox.models.lyz.Module2.PPNet13_T import SFAFMA_T
from toolbox.models.lyz.Module2.PPNet13_S import SFAFMA_S


'''
just test loss
'''


# DATASET = "Potsdam"
DATASET = "Vaihingen"
batch_size = 10
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


criterion_st_1 = ST_3(10,224).cuda()
criterion_st_2 = ST_3(16,224).cuda()
criterion_st_3 = ST_3(24,224).cuda()
criterion_st_4 = ST_3(32,896).cuda()
criterion_st_5 = ST_3(60,896).cuda()

criterion_SelfA1 = SelfA(3,3,[16,24,32],[64,128,256]).cuda()
criterion_SelfA2 = SelfA(3,3,[32,60,100],[256,512,512]).cuda()


iou = IOU()
criterion_without = MscCrossEntropyLoss().cuda()
# criterion1 = nn.CrossEntropyLoss(weight=weight.get_weight(test_dataloader, num_classes=5)).cuda()

criterion_focal1 = FocalLossbyothers().cuda()
criterion_Lovasz = MscLovaszSoftmaxLoss().cuda()
criterion_bce = nn.BCELoss().cuda()  # 边界监督
net_s = SFAFMA_S().cuda()
net_T = SFAFMA_T().cuda()
# net_T.load_state_dict(torch.load('./weight/SFAFMA_T(ablation_hlg)-Potsdam-loss.pth'))
# net_T.load_state_dict(torch.load('./weight/SFAFMA_T-Potsdam-loss.pth'))
net_T.load_state_dict(torch.load('./weight/SFAFMA_T_3-Vaihingen-loss.pth'))
# net_T.load_state_dict(torch.load('./weight/PPNet_S_KD-Potsdam-loss.pth'))
optimizer = optim.Adam(net_s.parameters(), lr=1e-4, weight_decay=5e-4)

# resume = True
# if resume:
#     net_s.load_state_dict(torch.load('./weight/PPNet_S_KD(ablation_AT)_last-{}-loss.pth'.format(DATASET)))

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
        ndsm = torch.repeat_interleave(ndsm, 3, dim=1)
        out = net(image, ndsm)
        with torch.no_grad():
            # ndsm = torch.repeat_interleave(ndsm, 3, dim=1)
            out1 = net_T(image,ndsm)
        # with teacher transformer to label
        teacher_out = out1[0].data.cpu().numpy()
        teacher_out = np.argmax(teacher_out,axis=1)
        teacher_out = torch.from_numpy(teacher_out)
        teacher_out = Variable(teacher_out.long().cuda())
        loss0 = criterion_without(out[0],teacher_out)
        # # loss calculate
        loss_avg = criterion_without(out[0:5],label)


        t = 2
        tout1 = F.softmax(out1[0]/t,dim=1)
        tout2 = F.softmax(out1[1]/t,dim=1)
        tout3 = F.softmax(out1[2]/t,dim=1)
        tout4 = F.softmax(out1[3]/t,dim=1)
        tout5 = F.softmax(out1[4]/t,dim=1)
        sout1 = F.softmax(out[0]/t, dim=1)
        sout2 = F.softmax(out[1]/t, dim=1)
        sout3 = F.softmax(out[2]/t, dim=1)
        sout4 = F.softmax(out[3]/t, dim=1)
        sout5 = F.softmax(out[4]/t, dim=1)
        loss1 = criterion1(sout1, tout1) * t * t  # 没有边界监督
        loss2 = criterion1(sout2, tout2) * t * t  # 没有边界监督
        loss3 = criterion1(sout3, tout3) * t * t  # 没有边界监督
        loss4 = criterion1(sout4, tout4) * t * t  # 没有边界监督
        loss5 = criterion1(sout5, tout5) * t * t  # 没有边界监督
        loss_KD = (loss5 + loss4 + loss3 + loss2 + loss1)/5


        # print(loss_trans)
        loss9 = criterion_SelfA1(out[10:13],out1[10:13])
        loss10 = criterion_SelfA2(out[12:15],out1[12:15])
        loss_selfA = (loss9 + loss10) / 2
        # print(loss_selfA)

        loss11 = criterion_st_1(out1[5],out1[6],out1[7],out[5])
        loss12 = criterion_st_2(out1[5],out1[6],out1[7],out[6])
        loss13 = criterion_st_3(out1[5],out1[6],out1[7],out[7])
        loss14 = criterion_st_4(out1[7],out1[8],out1[9],out[8])
        loss15 = criterion_st_5(out1[7],out1[8],out1[9],out[9])
        loss_At = (loss11 + loss12 + loss13 + loss14 + loss15) / 5





        loss =  loss_avg + loss_selfA + loss_At + loss0 + loss_KD * 0.1


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
            ndsmVal = torch.repeat_interleave(ndsmVal, 3, dim=1)
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
        torch.save(net.state_dict(), './weight/PPNet_S_KD(paper2)_1-{}-loss.pth'.format(DATASET))
    torch.save(net.state_dict(), './weight/tem1/PPNet_S_KD(paper2)_{}_last-{}-loss.pth'.format(epoch,DATASET))


    print(max(best), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   Accuracy', numloss)

# PPNet_S_KD(CE[S,T]_AT_KL+selfA))
# PPNet_S_KD(CE[S,T]_KL))