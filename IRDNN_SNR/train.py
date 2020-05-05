import os,torch,cv2,yaml,util,lmdb,argparse
import numpy as np
import torchvision
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from model import *
from data_loader import *
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='set necessary argument...')
parser.add_argument('--gpu_num', type=int, default=0, help='gpu_num: 0 or 1')
parser.add_argument('--epoch', type=int, default=40, help='how many epoch you want to run')
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--flag', type=str, default='SNR', help='model name') 
parser.add_argument('--qp', type=int, default=38, help='set qp begin.')
parser.add_argument('--ref_num', type=int, default=2, help='set reference frame number begin.')
parser.add_argument('--mode', type=int, default=0, help='train=0')
parser.add_argument('--resume', type=str, default='None', help='[None] will not load model, default is model_epoch_50.pth ')
args = parser.parse_args()

gpu_num = args.gpu_num
flag = args.flag
bs = args.batch_size
lr = 0.0001
Epoch = args.epoch
mode = args.mode
ref_num = args.ref_num

qp = args.qp
# train = True
resume = args.resume
writer = SummaryWriter('tensorboard/record-QP{}_{}_RA'.format(qp,flag))
with open("/data/disk2/Datasets/Vimeo/part1/Vimeo_packed/lmdb/GT_info.yaml", "r") as yaml_part1:
            yaml_obj1 = yaml.load(yaml_part1.read())
with open("/data/disk2/Datasets/Vimeo/part2/Vimeo_packed/lmdb/GT_info.yaml", "r") as yaml_part2:
            yaml_obj2 = yaml.load(yaml_part2.read())
with open("/data/disk2/Datasets/Vimeo/part3/Vimeo_packed/lmdb/GT_info.yaml", "r") as yaml_part3:
            yaml_obj3 = yaml.load(yaml_part3.read())

def main():
    print("============> Setting GPU")

    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_num)
    device = torch.device("cuda:{}".format(torch.cuda.current_device()))

    print("============> Loading datasets ")
    train_data = Data_codec_isolate(ref_num,qp,yaml_obj1,yaml_obj2,yaml_obj3)
    train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=6)

    print("============> building model")

    model = IRDNN_SNR()
    model = model.to(device)

    print_network(model)

    epoch = 1

    print("============> setting optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    if resume != 'None':
        print("============> load checkpoint {}".format(resume))
        epoch = int(args.resume.split('_')[-5]) + 1
        state_path = os.path.join("checkpoint", resume)
        state = torch.load(state_path)

        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])

    print("============> training begin QP={}".format(qp))

    for i in range(epoch, Epoch+1):
        train(train_dataloader, model, optimizer, i, lr, device=device)
        if i % 1 == 0:
            save_checkpoint(model, optimizer, i)
            print("Checkpoint of epoch {} saved ".format(i))


def train(data_loader, model, opt, epoch, lr, device):
    learning_rate = adjust_learning_rate(epoch,lr)
    print("Epoch = {}, lr = {}".format(epoch, learning_rate))
    for param_group in opt.param_groups:
        param_group["lr"] = learning_rate

    model.train()
    psnr_decodec = 0
    for i, (img_GT, img_Rec_high, img_Rec_list_high, img_Rec_list_low) in enumerate(data_loader):
        img_GT = img_GT.to(device)
        img_Rec_high = img_Rec_high.to(device)
        img_Rec_list_high = img_Rec_list_high.to(device)
        img_Rec_list_low = img_Rec_list_low.to(device)

        opt.zero_grad()

        # model forward
        out = model(img_Rec_high,img_Rec_list_high,img_Rec_list_low)
        loss = F.mse_loss(out,img_GT)
        loss.backward()
        opt.step()

        with torch.no_grad():
            # compute psnr
            psnr_codec = -20 * ((img_Rec_high - img_GT).pow(2).mean().pow(0.5)).log10()
            psnr_decodec = -20 * ((out - img_GT).pow(2).mean().pow(0.5)).log10()
            psnr_gain = psnr_decodec - psnr_codec

            writer.add_scalar("train_QP{}_psnr_decodec".format(qp), psnr_decodec, len(data_loader)*(epoch-1)+i)
            writer.add_scalar("train_QP{}_psnr_codec".format(qp), psnr_codec, len(data_loader)*(epoch-1)+i)
            writer.add_scalar("train_QP{}_psnr_gain".format(qp), psnr_gain, len(data_loader)*(epoch-1)+i)
            writer.add_scalar("train_QP{}_loss".format(qp), loss, len(data_loader)*(epoch-1)+i)

            print("train({}:{}):> Epoch[{}]({}/{})== Loss {:.4f}({:.4f}), psnr_decodec: {:.4f} ({:.4f}) ,psnr_gain: {:.4f}" \
                  .format(flag,qp,epoch, i+1, len(data_loader), loss,lr, psnr_decodec, psnr_codec, psnr_gain))

def save_checkpoint(model, optimizer, epoch):
    model_out_path = os.path.join("checkpoint", "model_QP{}-QP{}_epoch_{}_{}.pth".format(qp, qp-6, epoch, flag))
    state = {"model":  model.state_dict(), 
             "optimizer": optimizer.state_dict()}
    if not os.path.exists("checkpoint"):
        os.makedirs("checkpoint")
    torch.save(state, model_out_path)
    print("Checkpoint of epoch {} saved to {}".format(epoch, model_out_path))


def adjust_learning_rate(epoch,lr):
    """Sets the learning rate to the initial LR decayed by 0.1 every 20 epochs"""
    learning_rate = lr * (0.1 ** ((epoch) // 20))
    return learning_rate


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

if __name__ == "__main__":
    main()
    
