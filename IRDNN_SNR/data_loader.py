
import os,argparse,torch,lmdb,cv2,random,yaml,util
# import util
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as data
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import DataLoader
from model import *
from PIL import Image
from torchvision.transforms import ToTensor

class Data_codec_isolate(data.Dataset):
    def __init__(self, ref_num, qp, yaml_obj1, yaml_obj2, yaml_obj3, train=True,  transform=transforms.ToTensor()):
        self.ref_num = ref_num
        self.qp = qp
        self.transform = transform
        self.yaml_obj1 = yaml_obj1
        self.yaml_obj2 = yaml_obj2
        self.yaml_obj3 = yaml_obj3
        self.len = 57307
        # self.len = 36968
        self.GT_env = None
        self.Rec_env = None

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx < 15727:
            name_GT = '/data/disk2/Datasets/Vimeo/part1/Vimeo_packed/lmdb/GT'
            name_high = '/data/disk2/Datasets/Vimeo/part1/Vimeo_packed/lmdb/ra_QP{}'.format(self.qp)
            name_low = '/data/disk2/Datasets/Vimeo/part1/Vimeo_packed/lmdb/ra_QP{}'.format(self.qp-6)
            key_idx = self.yaml_obj1[idx]
        elif idx > 15726 and idx < 36067 :
            name_GT = '/data/disk2/Datasets/Vimeo/part2/Vimeo_packed/lmdb/GT'
            name_high = '/data/disk2/Datasets/Vimeo/part2/Vimeo_packed/lmdb/ra_QP{}'.format(self.qp)
            name_low = '/data/disk2/Datasets/Vimeo/part2/Vimeo_packed/lmdb/ra_QP{}'.format(self.qp-6)
            key_idx = self.yaml_obj2[idx-15727]
        else :
            name_GT = '/data/disk2/Datasets/Vimeo/part3/Vimeo_packed/lmdb/GT'
            name_high = '/data/disk2/Datasets/Vimeo/part3/Vimeo_packed/lmdb/ra_QP{}'.format(self.qp)
            name_low = '/data/disk2/Datasets/Vimeo/part3/Vimeo_packed/lmdb/ra_QP{}'.format(self.qp-6)
            key_idx = self.yaml_obj3[idx-36067]

        GT_env = lmdb.open(name_GT, readonly=True, lock=False, readahead=False,
                                    meminit=False)
        High_env = lmdb.open(name_high, readonly=True, lock=False, readahead=False,
                                    meminit=False)
        Low_env = lmdb.open(name_low, readonly=True, lock=False, readahead=False,
                                    meminit=False)      
        ref_num = self.ref_num
        n_frame = random.randint(0,16)
        GT_key = 'GT_'+key_idx+'_'+ str(n_frame).rjust(3, '0')
        Rec_key_high = 'Rec_'+key_idx+'_'+ str(n_frame).rjust(3, '0')
        nh = 144
        nw = 144

        #### determine the neighbor frames
        poc = int(n_frame)
        video_len = 16
        neighbor_list =[]
        if (poc % 2) == 1:
            if (poc+1) < video_len:
                neighbor_list = [poc-1,poc+1]
            else:
                neighbor_list = [poc-1,poc-1]
        elif (poc % 8 ) == 0:
                neighbor_list = [poc,poc]
        elif (poc % 8 ) == 4:
            if (poc+4) < video_len:
                neighbor_list = [poc-4,poc+4]
            else:
                neighbor_list = [poc-4,poc-4]
        else:
            if (poc+2) < video_len:
                neighbor_list = [poc-2,poc+2]
            else:
                neighbor_list = [poc-2,poc-2]

        #### get the GT image (as the center frame)
        img_GT = util._read_img_lmdb(GT_env, GT_key, (1, int(nw), int(nh)))
        img_Rec_high = util._read_img_lmdb(High_env, Rec_key_high, (1, int(nw), int(nh)))

        #### get Rec images
        img_Rec_list_high = []
        img_Rec_list_low =[]

        #img_Rec_list_high
        for i in range(0,ref_num):
            Rec_key_list_high = 'Rec_' + key_idx + '_' + str(neighbor_list[i]).rjust(3, '0') 
            img_Rec_high_ = util._read_img_lmdb(High_env, Rec_key_list_high, (1, int(nw), int(nh)))
            img_Rec_list_high.append(img_Rec_high_)
 
        #img_Rec_list_low
        for i in range(0,ref_num):
            Rec_key_list_low = 'Rec_' + key_idx + '_' + str(neighbor_list[i]).rjust(3, '0') 
            if neighbor_list[i] == poc :
                img_Rec_low_ = util._read_img_lmdb(High_env, Rec_key_list_low, (1, int(nw), int(nh)))
            else:
                img_Rec_low_ = util._read_img_lmdb(Low_env, Rec_key_list_low, (1, int(nw), int(nh)))
            img_Rec_list_low.append(img_Rec_low_)

        High_env.close()
        Low_env.close()

        # numpy to tensor
        img_Rec_list_high = np.stack(img_Rec_list_high, axis=0)  # NHWC
        img_Rec_list_low = np.stack(img_Rec_list_low, axis=0)  # NHWC

        # [HWC] to [CHW]   [NHWC] to [NCHW]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT/255.0, (2, 0, 1)))).float()
        img_Rec_high = torch.from_numpy(np.ascontiguousarray(np.transpose(img_Rec_high/255.0, (2, 0, 1)))).float()
        img_Rec_list_high = torch.from_numpy(np.ascontiguousarray(np.transpose(img_Rec_list_high/255.0, (0, 3, 1, 2)))).float()
        img_Rec_list_low = torch.from_numpy(np.ascontiguousarray(np.transpose(img_Rec_list_low/255.0, (0, 3, 1, 2)))).float()

        return img_GT, img_Rec_high, img_Rec_list_high, img_Rec_list_low
