import torch,cv2,os
import torchvision
from model import *
from yuv_io import *
import numpy as np
import argparse
from upsample_SHM import upsample
from torchvision.utils import save_image
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr

parser = argparse.ArgumentParser(description='set necessary argument...')
parser.add_argument('--gpu_num', type=int, default=0, help='gpu_num: 0 or 1')
parser.add_argument('--qp', type=int, default=38, help='set qp begin. choose form 26,30,34,38')
parser.add_argument('--delta_qp', type=int, default=6, help='set qp begin. choose form 4,6')
parser.add_argument('--width', type=int, default=1920, help='1920, 2560')
parser.add_argument('--height', type=int, default=1080, help='1080,  1600')
parser.add_argument('--w_count', type=int, default=12, help='12, 8')
parser.add_argument('--h_count', type=int, default=6, help='6,  5')
parser.add_argument('--start', type=int, default=0, help='just used when test basketball video, start frame in test')
parser.add_argument('--resume', type=str, default='None', help='[None] will not load model, default is model_epoch_50.pth ')
parser.add_argument('--sequence', type=str, default='Cactus', help='which sequence to be deal with')
parser.add_argument('--fps', type=int, default=50, help='fps of the sequence, 24, 50, 60')
parser.add_argument('--count', type=int, default=48, help='all frames to be deal, 24, 48, 64')
args = parser.parse_args()

"""
--sequence:
    PeopleOnStreet（30）
    Traffic
    ParkScene（24）
    Kimonol
    Cactus（50）
    BQTerrace（60）
    BasketballDrive（50）
--net:
    IRDNN_SRN
    IRDNN_SVC
"""

gpu_num = args.gpu_num
qp = args.qp
delta_qp = args.delta_qp
resume = args.resume
start_frame = args.start
sequence = args.sequence
fps = args.fps          # sequence's fps 50, 24, 60 ....
Count = args.count
width = args.width
height = args.height
w_count = args.w_count
h_count = args.h_count

ori_file = os.path.join('/data/disk1/lilei/SHVC/data', 'sequence', '{}_{}x{}_{}.yuv'.format(sequence, width, height,fps))
rec_file_high = os.path.join('/data/disk1/lilei/SHVC/data', 'SNR_RA_Sequence', 'SNR_{}_qp{}_qp{}_RA_l0_rec.yuv'.format(sequence, qp, qp-delta_qp))
rec_file_low = os.path.join('/data/disk1/lilei/SHVC/data', 'SNR_RA_Sequence', 'SNR_{}_qp{}_qp{}_RA_l1_rec.yuv'.format(sequence, qp, qp-delta_qp))

print(rec_file_high)
Save_path = os.path.join('./result/{}_{}/'.format(sequence,qp))
if not os.path.exists(Save_path):
    os.makedirs(Save_path)

w_block = int(width/w_count)
h_block = int(height/h_count)

def main():

    print("============> Setting GPU")
    os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(gpu_num)
    device = torch.device("cuda:{}".format(torch.cuda.current_device()))
    model = IRDNN_SNR()
    model = model.to(device)
    model.eval()

    if resume != 'None':
        print("============> load checkpoint {}".format(resume))
        state_path = os.path.join("checkpoint", resume)
        state = torch.load(state_path)
        model.load_state_dict(state["model"])

    f_ori = open(ori_file, 'rb')
    f_rec = open(rec_file_high, 'rb')

    f_ori.seek(int(start_frame*height*width/2*3))
    f_rec.seek(int(start_frame*height*width/2*3))

    y_out = torch.zeros([1, 1, height , width], dtype=torch.float32)

    psnrs_rec = []
    psnrs_out = []
    psnrs_gain = []     
    ssims_rec = []
    ssims_out = []
    ssims_gain = []         

    [Y_frame_high,U_frame_high,V_frame_high] = YUVread(rec_file_high, [width,height], frame_num=Count, start_frame=0, mode='420')
    [Y_frame_low,U_frame_low,V_frame_low] = YUVread(rec_file_low, [width,height], frame_num=Count, start_frame=0, mode='420')
    with torch.no_grad():
        
        for i in range(start_frame, start_frame+Count):
            Save_path_yuv_repair = os.path.join(Save_path, '{}_SNR_repair_{}.yuv'.format(sequence,i))
            Save_path_yuv_rec = os.path.join(Save_path, '{}_SNR_rec_{}.yuv'.format(sequence,i))
            f_Save_repair = open(Save_path_yuv_repair, 'wb')
            f_Save_rec = open(Save_path_yuv_rec, 'wb')
            
            img_Rec_list_high = []
            img_Rec_list_low = []

            Y_frame_high = np.reshape(Y_frame_high,[Count,height, width,1,1])
            Y_frame_low = np.reshape(Y_frame_low,[Count,height, width,1,1])
            Y_frame_high_single = Y_frame_high[i,:,:]
            if (i % 2)==1 :
                if (i+1) < Count :
                    img_Rec_list_high.append(Y_frame_high[i-1,:,:])
                    img_Rec_list_high.append(Y_frame_high[i+1,:,:])
                    img_Rec_list_low.append(Y_frame_low[i-1,:,:])
                    img_Rec_list_low.append(Y_frame_low[i+1,:,:])
                else:
                    img_Rec_list_high.append(Y_frame_high[i-1,:,:])
                    img_Rec_list_high.append(Y_frame_high[i-1,:,:])
                    img_Rec_list_low.append(Y_frame_low[i-1,:,:])
                    img_Rec_list_low.append(Y_frame_low[i-1,:,:])
            elif (i % 8)==0 :
                    img_Rec_list_high.append(Y_frame_high[i,:,:])
                    img_Rec_list_high.append(Y_frame_high[i,:,:])
                    img_Rec_list_low.append(Y_frame_high[i,:,:])
                    img_Rec_list_low.append(Y_frame_high[i,:,:])
            elif (i % 8) ==4 :
                if (i+4) < Count :
                    img_Rec_list_high.append(Y_frame_high[i-4,:,:])
                    img_Rec_list_high.append(Y_frame_high[i+4,:,:])
                    img_Rec_list_low.append(Y_frame_low[i-4,:,:])
                    img_Rec_list_low.append(Y_frame_low[i+4,:,:])
                else:
                    img_Rec_list_high.append(Y_frame_high[i-4,:,:])
                    img_Rec_list_high.append(Y_frame_high[i-4,:,:])
                    img_Rec_list_low.append(Y_frame_low[i-4,:,:])
                    img_Rec_list_low.append(Y_frame_low[i-4,:,:])
            else:
                if (i+2) < Count :
                    img_Rec_list_high.append(Y_frame_high[i-2,:,:])
                    img_Rec_list_high.append(Y_frame_high[i+2,:,:])
                    img_Rec_list_low.append(Y_frame_low[i-2,:,:])
                    img_Rec_list_low.append(Y_frame_low[i+2,:,:])
                else:
                    img_Rec_list_high.append(Y_frame_high[i-2,:,:])
                    img_Rec_list_high.append(Y_frame_high[i-2,:,:])
                    img_Rec_list_low.append(Y_frame_low[i-2,:,:])
                    img_Rec_list_low.append(Y_frame_low[i-2,:,:])

            img_Rec_list_high = np.stack(img_Rec_list_high, axis=0)  # NHWCB  BNCHW
            img_Rec_list_low = np.stack(img_Rec_list_low, axis=0)  # NHWCB  BNCHW

            Y_frame_high_single = torch.from_numpy(np.ascontiguousarray(np.transpose(Y_frame_high_single/255.0,(3, 2, 0, 1)))).float() #HWCB  BCHW
            img_Rec_list_high = torch.from_numpy(np.ascontiguousarray(np.transpose(img_Rec_list_high/255.0, (4, 0, 3, 1, 2)))).float() # NHWCB  BNCHW
            img_Rec_list_low = torch.from_numpy(np.ascontiguousarray(np.transpose(img_Rec_list_low/255.0, (4, 0, 3, 1, 2)))).float()

            y_rec = torch.tensor(np.reshape(np.uint8(list(f_rec.read(height * width))), [1, 1, height, width]) / 255).float()
            u_pad = np.uint8(list(f_rec.read(height * width >> 2)))
            v_pad = np.uint8(list(f_rec.read(height * width >> 2)))

            y_ori = torch.tensor(np.reshape(np.uint8(list(f_ori.read(height * width))), [1, 1, height, width]) / 255).float()
            u = np.uint8(list(f_ori.read(height * width >> 2)))
            v = np.uint8(list(f_ori.read(height * width >> 2)))

            # deal a block
            for w in range(w_count):
                for h in range(h_count):
                    Y_frame_high_single_patch = Y_frame_high_single[:, :, h*h_block:(h+1)*h_block, w*w_block:(w+1)*w_block].to(device)
                    img_Rec_list_high_patch = img_Rec_list_high[:, :, :, h*h_block:(h+1)*h_block, w*w_block:(w+1)*w_block].to(device)
                    img_Rec_list_low_patch = img_Rec_list_low[:, :, :, h*h_block:(h+1)*h_block, w*w_block:(w+1)*w_block].to(device)
                    out = model(Y_frame_high_single_patch,img_Rec_list_high_patch,img_Rec_list_low_patch)
                    y_out[ :,:, h*h_block:(h+1)*h_block, w*w_block:(w+1)*w_block] = out.detach().to(torch.device("cpu"))
            
            psnr_out = compare_psnr(np.float32(y_ori), np.float32(y_out), data_range=1.0)
            psnr_rec = compare_psnr(np.float32(y_ori), np.float32(y_rec), data_range=1.0)
            ssim_out = compare_ssim(np.reshape(np.float32(y_ori),[height, width]), np.reshape(np.float32(y_out),[height, width]),data_range=1.0)
            ssim_rec = compare_ssim(np.reshape(np.float32(y_ori),[height, width]), np.reshape(np.float32(y_rec),[height, width]), data_range=1.0)
            psnr_gain = psnr_out - psnr_rec
            ssim_gain = ssim_out - ssim_rec

            psnrs_out.append(psnr_out)
            psnrs_rec.append(psnr_rec)
            psnrs_gain.append(psnr_gain)
            ssims_out.append(ssim_out)
            ssims_rec.append(ssim_rec)
            ssims_gain.append(ssim_gain)

            print("{}:> ({}/{})===psnr_out: {:.3f}({:.3f}), psnr_gain: {:.3f}, ssim_out: {:.3f}({:.3f}), ssim_gain: {:.3f} "\
                .format(sequence, i+1, Count, psnr_out,psnr_rec,psnr_gain,ssim_out,ssim_rec,ssim_gain ))

            f_Save_repair.write(np.uint8(y_out.detach()*255).tobytes())
            f_Save_repair.write(u_pad.tobytes())
            f_Save_repair.write(v_pad.tobytes())
            f_Save_repair.close()

            f_Save_rec.write(np.uint8(y_rec.detach()*255).tobytes())
            f_Save_rec.write(u_pad.tobytes())
            f_Save_rec.write(v_pad.tobytes())
            f_Save_rec.close()

    psnr_out_avg = np.mean(np.array(psnrs_out))
    psnr_rec_avg = np.mean(np.array(psnrs_rec))
    psnr_gain_avg = np.mean(np.array(psnrs_gain))
    ssim_out_avg = np.mean(np.array(ssims_out))
    ssim_rec_avg = np.mean(np.array(ssims_rec))
    ssim_gain_avg = np.mean(np.array(ssims_gain))
    print("=psnr_ori_avg: {:.3f}({:.3f}) psnr_gain_avg: {:.3f}, ssim_ori_avg: {:.3f}({:.3f}) ssim_gain_avg: {:.3f}"\
        .format(psnr_out_avg,psnr_rec_avg,psnr_gain_avg,ssim_out_avg,ssim_rec_avg,ssim_gain_avg))

    f_ori.close()
    f_rec.close()
    

if __name__ == '__main__':
    main()


