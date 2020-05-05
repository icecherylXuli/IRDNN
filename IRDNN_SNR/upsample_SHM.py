import os
import sys
# import cv2
import torch
import numpy as np


def progress_bar(num, total, width=40):
    rate = num / total
    rate_num = int(rate * width)
    r = '\r[%s%s] %d%%%s%d' % ("=" * rate_num, " " * (width - rate_num), int(rate * 100), ' done of ', total)
    sys.stdout.write(r)
    sys.stdout.flush()


def upsample(input_tensor, factor=2): #output_path, input_width, input_height, factor):
    b, c, h_ori, w_ori  = input_tensor.shape
    # print(b,c,h_ori, w_ori)
    Y_out = np.zeros([b, c, 2*h_ori, 2*w_ori], np.int32)

    # widths and heights
    # w_ori = input_width
    # h_ori = input_height
    w_pad = w_ori + 8
    h_pad = h_ori
    w_16_tmp = round(w_ori * factor)
    h_16_tmp = h_ori + 8
    w_out = round(w_ori * factor)
    h_out = round(h_ori * factor)

    filters = np.array([
        [0, 0, 0, 64, 0, 0, 0, 0],
        [0, 1, -3, 63, 4, -2, 1, 0],
        [-1, 2, -5, 62, 8, -3, 1, 0],
        [-1, 3, -8, 60, 13, -4, 1, 0],
        [-1, 4, -10, 58, 17, -5, 1, 0],
        [-1, 4, -11, 52, 26, -8, 3, -1],
        [-1, 3, -9, 47, 31, -10, 4, -1],
        [-1, 4, -11, 45, 34, -10, 4, -1],
        [-1, 4, -11, 40, 40, -11, 4, -1],
        [-1, 4, -10, 34, 45, -11, 4, -1],
        [-1, 4, -10, 31, 47, -9, 3, -1],
        [-1, 3, -8, 26, 52, -11, 4, -1],
        [0, 1, -5, 17, 58, -10, 4, -1],
        [0, 1, -4, 13, 60, -8, 3, -1],
        [0, 1, -3, 8, 62, -5, 2, -1],
        [0, 1, -2, 4, 63, -3, 1, 0]
    ])

    # filters = np.array([
    #     [  0,    0,    0,   64,    0,    0,    0,    0 ],
    #     [  0,    1,   -3,   63,    4,   -2,    1,    0 ],
    #     [ -1,    2,   -5,   62,    8,   -3,    1,    0 ],
    #     [ -1,    3,   -8,   60,   13,   -4,    1,    0 ],
    #     [ -1,    4,  -10,   58,   17,   -5,    1,    0 ],
    #     [ -1,    4,  -11,   52,   26,   -8,    3,   -1 ],
    #     [ -1,    3,   -9,   47,   31,  -10,    4,   -1 ],
    #     [ -1,    4,  -11,   45,   34,  -10,    4,   -1 ],
    #     [ -1,    4,  -11,   40,   40,  -11,    4,   -1 ],
    #     [ -1,    4,  -10,   34,   45,  -11,    4,   -1 ],
    #     [ -1,    4,  -10,   31,   47,   -9,    3,   -1 ],
    #     [ -1,    3,   -8,   26,   52,  -11,    4,   -1 ],
    #     [  0,    1,   -5,   17,   58,  -10,    4,   -1 ],
    #     [  0,    1,   -4,   13,   60,   -8,    3,   -1 ],
    #     [  0,    1,   -3,    8,   62,   -5,    2,   -1 ],
    #     [  0,    1,   -2,    4,   63,   -3,    1,    0 ]
    # ])

    # only Y
    # y_ori, u_ori, v_ori = YUVread(input_tensor, [h_ori, w_ori], 1)
    # b, c, w_ori, h_ori = input_tensor.shape
    # y_ori = np.reshape(y_ori, [h_ori, w_ori])
    # y_ori = cv2.imread(input_tensor, 0)
    input_tensor = (255 * input_tensor).int()
    # b, c, w_ori, h_ori = input_tensor.shape
    for i in range(len(input_tensor)):
        y_ori = input_tensor[i,0,:,:]

        y_pad = np.zeros([h_pad, w_pad], np.int32)
        y_pad[0:h_ori, 4:4 + w_ori] = y_ori[:, :]
        # pad left
        y_pad[0:h_ori, 0] = y_ori[:, 0]
        y_pad[0:h_ori, 1] = y_ori[:, 0]
        y_pad[0:h_ori, 2] = y_ori[:, 0]
        y_pad[0:h_ori, 3] = y_ori[:, 0]
        # pad right
        y_pad[0:h_ori, -4] = y_ori[:, -1]
        y_pad[0:h_ori, -3] = y_ori[:, -1]
        y_pad[0:h_ori, -2] = y_ori[:, -1]
        y_pad[0:h_ori, -1] = y_ori[:, -1]

        y_16_tmp = np.zeros([h_16_tmp, w_16_tmp], np.int32)
        # y_out = np.zeros([h_out, w_out], np.int32)

        # horizontal
        for h in range(h_ori):
            for w in range(w_out):
                w_in_16 = round(w * 16 / factor)
                y_16_tmp[h + 4, w] = np.sum(y_pad[h, w_in_16 // 16 + 1:w_in_16 // 16 + 9] * filters[w_in_16 % 16])

        # pad top
        y_16_tmp[0, :] = y_16_tmp[4, :]
        y_16_tmp[1, :] = y_16_tmp[4, :]
        y_16_tmp[2, :] = y_16_tmp[4, :]
        y_16_tmp[3, :] = y_16_tmp[4, :]
        # pad bottom
        y_16_tmp[-4, :] = y_16_tmp[-5, :]
        y_16_tmp[-3, :] = y_16_tmp[-5, :]
        y_16_tmp[-2, :] = y_16_tmp[-5, :]
        y_16_tmp[-1, :] = y_16_tmp[-5, :]

        # y_16_tmpo = np.uint8(y_16_tmp/64)
        # Ywrite(y_16_tmpo, 'test00.yuv')

        # vertical
        for w in range(w_out):
            for h in range(h_out):
                h_in_16 = round(h * 16 / factor)
                value = np.sum(y_16_tmp[h_in_16 // 16 + 1:h_in_16 // 16 + 9, w] * filters[h_in_16 % 16])
                if value % (4096) >= 2048:
                    Y_out[i, 0, h, w] = (value >> 12) + 1
                else:
                    Y_out[i, 0, h, w] = (value >> 12)

    # clip & output
    Y_out = torch.from_numpy(np.clip(Y_out, 0, 255)/255).float()
    return Y_out


if __name__ == '__main__':
    
    # in_path = 'train_data_psnr/t37_214_120.yuv'
    # out_path = 'train_data_psnr/t37_214_120_pyupsample.yuv'
    in_path = 'train_data_psnr/t25_208_196_104_98_2_0_0_0_22_rec.yuv'
    out_path = 'train_data_psnr/t25_208_196_104_98_2_0_0_0_22_rec_SHM_UP.yuv'
    # upsample(in_path, out_path, 960, 540, 2)
    upsample(in_path, out_path, 42, 42, 2)
    
