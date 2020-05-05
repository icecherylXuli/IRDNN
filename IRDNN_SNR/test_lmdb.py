import lmdb
import cv2
import numpy as np
import argparse
 



def test_lmdb(name='train_QP30'):
        # env_db = lmdb.Environment(name)
        dir = '/data/disk2/Datasets/derf_packed_whole/lmdb/train_QP38'   # /train_QP36_QP40 dir
        env_db = lmdb.Environment(dir)
        # env_db = lmdb.open("./trainC")
        txn = env_db.begin()
        # get函数通过键值查询数据,如果要查询的键值没有对应数据，则输出None
        buf = txn.get(str('Rec_FourPeople_001_1280_720_601').encode('ascii'))
        # print('Rec_akiyo_012_352_288_300')
        # print(buf)
        print(type(buf))
        
        # value = np.frombuffer(buf, dtype=np.uint8)
        # value = value.reshape(720, 1280)
        # cv2.imwrite('./test.png', value)

        # k = 1
        # for key, value in txn.cursor():  #遍历
        #         print (key)
        #         value = np.frombuffer(value, dtype=np.uint8)
        #         value = value.reshape(720, 1280)
        #         # print(value.shape)
        #         # cv2.imshow(str(k), value)
        #         # cv2.waitKey()
        #         # cv2.imwrite('./test_'+ str(k) + '.png', value)
        #         k+=1
        #         if k >= 10:
        #                 break

        env_db.close()

if __name__ == '__main__':
        # parser = argparse.ArgumentParser()
        # parser.add_argument('--name', type=str, default='train_QP36_QP40', help='lmdb name')
        # args = parser.parse_args()
        # name = args.name
        
        # test_lmdb(name=name)
        test_lmdb()

        # python test_lmdb.py --name train_QP36_QP40