import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./model')
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from model.sodnet import sodnet

from data import get_loader,test_dataset
from utils import clip_gradient
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt
from loss.ssim import SSIM




from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cudnn.benchmark = True
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#build the model
model = sodnet()


# path = '/home/cjb123321/projects/SFCNet/res/.pth'
# model.load_state_dict(torch.load(path))
# print("Successfully loaded:", path)
model = model.cuda()


base, body = [], []
for name, param in model.named_parameters():
        # if 'backbone' in name or 'swin_thermal' in name:
        if 'backbone' in name :
            # print('base')
            # print(name)
            base.append(param)
        else:
            # print('body')
            # print(name)
            body.append(param)
optimizer = torch.optim.SGD([{'params': base,'lr': opt.lr}, {'params': body,'lr': opt.lr}],  momentum=opt.momentum,
                                weight_decay=opt.decay_rate, nesterov=True)


#set the path
train_root = opt.train_data_root
test_root = opt.val_data_root

save_path=opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

#load data
print('load data...')
num_gpus = torch.cuda.device_count()
print(f"========>num_gpus:{num_gpus}==========")
train_loader = get_loader(train_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_root, opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path+'log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
# logging.info("SwinMCNet-Train")
logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{}'.format(opt.epoch,opt.lr,opt.batchsize,opt.trainsize,opt.clip,opt.decay_rate,opt.load,save_path))

# cl_loss = ContrastiveLossWithConv(k=3, temperature=0.07, threshold=0.4)
# loss
def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()
ssim_loss = SSIM(window_size=11, size_average=True)
# loss_fn = ContrastAndMSELoss(lambda_con=0.04, lambda_mse=1.0, alpha=5.0)

def flat(mask, h):
    batch_size = mask.shape[0]
    # h = 24     12,24,48
    mask = F.interpolate(mask, size=(int(h), int(h)), mode='bilinear')  # mask's size to 24x24
    x = mask.view(batch_size, 1, -1).permute(0, 2, 1)
    # print(x.shape) -> b 24*24 1
    g = x @ x.transpose(-2, -1)  # b 24*24 24*24
    g = g.unsqueeze(1)  # b 1 24*24 24*24
    return g

def att_loss(pred, mask, p4, p5, h): # (attmap,mask,wr,ws)
    g = flat(mask, h) # (4, 1, 576, 576)
    np4 = torch.sigmoid(p4.detach())
    np5 = torch.sigmoid(p5.detach())
    p4 = flat(np4, h)  # p4 and p5s' size to 24x24
    p5 = flat(np5, h)  # p4 and p5s' size to 24x24
    w1 = torch.abs(g - p4)
    w2 = torch.abs(g - p5)
    w = (w1 + w2) * 0.5 + 1
    # inter = ((pred * g) * w).sum(dim=(2, 3))  # wiou
    # union = ((pred + g) * w).sum(dim=(2, 3))
    # wiou = 1 - (inter + 1)/(union - inter + 1)
    attbce = F.binary_cross_entropy_with_logits(pred, g, weight=w * 1.0,
                                                reduction='mean')  # (input, target, weight, reduction)
    return attbce



step=0
writer = SummaryWriter(save_path+'summary')
best_mae=1
best_epoch=1
loss_kl = nn.KLDivLoss(reduction='batchmean')
#train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all=0
    epoch_step=0
    #SLSLOSS = SLSIoULoss()
    model_num = 4
    try:
        for i, (images, ts, gts, bodys, details) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            
            image, t, gt, body, detail = images.cuda(), ts.cuda(), gts.cuda(), bodys.cuda(), details.cuda()

            outb1, outd1, out1, outb2, outd2, out2 = model(image, t)
            # cl_total = cl_loss(t4cl, r4cl)
            loss0 = F.binary_cross_entropy_with_logits(outb2, body) + ssim_loss(outb2, body)
            loss1 = F.binary_cross_entropy_with_logits(outd2, detail) + ssim_loss(outd2, detail)

            loss2 = F.binary_cross_entropy_with_logits(out2, gt) + iou_loss(out2, gt) + ssim_loss(out2, gt)
            # loss3 = F.binary_cross_entropy_with_logits(e3, gt) + iou_loss(e3, gt) + ssim_loss(e3, gt)
            # loss2 = F.binary_cross_entropy_with_logits(e2, gt) + iou_loss(e2, gt) + ssim_loss(e2, gt)
            # loss1 = F.binary_cross_entropy_with_logits(e1, gt) + iou_loss(e1, gt) + ssim_loss(e1, gt)
            # loss4 = loss_fn(t1,rgb1,out2)
            # att_loss_1 = att_loss(cam1, gt, wr1, wd1, 12)  # attention guided loss    #J
            # att_loss_2 = att_loss(cam2, gt, wr2, wd2, 24)  # attention guided loss    #J
            # att_loss_3 = att_loss(cam3, gt, wr3, wd3, 48)  # attention guided loss    #J


            loss = loss2 + 0.5*loss0 + 0.5*loss1
            # loss = loss2
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step+=1
            epoch_step+=1
            loss_all+=loss.data
            if i % 50 == 0 or i == total_step or i==1:
                print('%s | epoch:%d/%d | step:%d/%d | lr=%.6f | lossr=%.6f | losst=%.6f | loss=%.6f '
                    %(datetime.now(),  epoch, opt.epoch, i, total_step, optimizer.param_groups[1]['lr'], loss.item(),
                    loss2.item(), loss2.item()))

                logging.info('##TRAIN##:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], lr_bk: {:.6f}, Loss1: {:.4f} '.
                    format( epoch, opt.epoch, i, total_step, optimizer.param_groups[1]['lr'], loss.data))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res=out2[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('out', torch.tensor(res), step,dataformats='HW')
                res=out2[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('out2', torch.tensor(res), step,dataformats='HW')
        
        loss_all/=epoch_step
        logging.info('##TRAIN##:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format( epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 10 == 0:
            torch.save(model.state_dict(), save_path+'SFC_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt: 
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path+'SFC_epoch_{}.pth'.format(epoch))
        print('save checkpoints successfully!')
        raise
        
#test function
@torch.no_grad()
def test(test_loader,model,epoch,save_path):
    global best_mae,best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum=0
        #for i in range(1000):
        for i in range(test_loader.size):
            image, t, gt, (H, W), name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            t = t.cuda()
            #shape = (W,H)
            outb1, outd1, out1, outb2, outd2, out2 = model(image,t)
            res = out2
            res = F.interpolate(res, size=gt.shape, mode='bilinear')
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])
        mae=mae_sum/test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('\n')
        print('##TEST##:Epoch: {}   MAE: {}'.format(epoch,mae))
        
        if epoch==1:
            best_mae=mae
        else:
            if mae<best_mae:
                best_mae=mae
                best_epoch=epoch
                torch.save(model.state_dict(), save_path+'CDFF_epoch_best.pth')
        print('##SAVE##:bestEpoch: {}   bestMAE: {}'.format(best_epoch,best_mae))
        print('\n')
        logging.info('##TEST##:Epoch:{}   MAE:{}   bestEpoch:{}   bestMAE:{}'.format(epoch,mae,best_epoch,best_mae))


if __name__ == '__main__':
    print("Start train...")
    decay_rate = 0.05  # 学习率衰减比例
    decay_epoch = 10  # 每隔多少个epoch衰减一次
    for epoch in range(1, opt.epoch + 1):
        # if 10 <= epoch <= 80 and epoch % 10 == 0:  # epoch处于20-80时衰减
        #     optimizer.param_groups[0]['lr'] *= decay_rate
        #     optimizer.param_groups[1]['lr'] *= decay_rate
        #     writer.add_scalar('lr', optimizer.param_groups[1]['lr'], global_step=epoch)
        #
        #     print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[1]['lr']}")
        if epoch < 60:
            # lr = (1 - abs((epoch) / 60 * 2 - 1)) * opt.lr
            optimizer.param_groups[0]['lr'] = (1 - abs((epoch) / 60 * 2 - 1)) * opt.lr * 0.1
            optimizer.param_groups[1]['lr'] = (1 - abs((epoch) / 60 * 2 - 1)) * opt.lr
        else:
            # lr = opt.lr * (1 - abs((59) / 60 * 2 - 1))
            optimizer.param_groups[0]['lr'] = opt.lr * (1 - abs((59) / 60 * 2 - 1)) * 0.1
            optimizer.param_groups[1]['lr'] = opt.lr * (1 - abs((59) / 60 * 2 - 1))
        # optimizer.param_groups[0]['lr'] = (1 - abs((epoch) / (opt.epoch) * 2 - 1)) * opt.lr * 0.1
        # optimizer.param_groups[1]['lr'] = (1 - abs((epoch) / (opt.epoch) * 2 - 1)) * opt.lr
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)
        train(train_loader, model, optimizer, epoch,save_path)
        test(test_loader,model,epoch,save_path)