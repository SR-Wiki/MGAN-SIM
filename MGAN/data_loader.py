import scipy
from glob import glob
import numpy as np
import random
import tifffile
class DataLoader():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def load_data(self, start=1, end=6, is_testing=False):
        data_type = "train" if not is_testing else "test"

        imgs_As = []
        imgs_Bs = []

        for num in range(start, end):
            batch_images = ('./%s/%d.tif' % (self.dataset_name, num))
            img = self.imread(batch_images)
            h, w = img.shape
            _w = int(w / 2)
            _h = int(h / 2)
            img_A, img_B = img[:, :_w], img[:, _w:]
            img_A = img_A - np.min(img_A)
            img_A = img_A / np.max(img_A)
            img_B = img_B - np.min(img_B)
            img_B = img_B / np.max(img_B)
            img_A = img_A.astype('float32')
            img_B = img_B.astype('float32')

            img_A = img_A.reshape(1, h, _w)
            img_B = img_B.reshape(1, h, _w)

            imgs_As.append(img_A)
            imgs_Bs.append(img_B)

        imgs_As = np.array(imgs_As) / 255.
        imgs_Bs = np.array(imgs_Bs) / 255.


        imgs_As = np.array(imgs_As)
        imgs_Bs = np.array(imgs_Bs)

        yield imgs_As, imgs_Bs


    def load_batch(self, train_batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path = glob('./%s/*' % (self.dataset_name))

        self.batch_num = int(np.floor(len(path) / train_batch_size))
        for i in range(self.batch_num):
            location = np.array(np.random.randint(low=0, high=len(path), size=(train_batch_size, 1), dtype='int'))
            for batchsize in range(train_batch_size):
                # location = location.tolist()
                batch = []
                batch_tem = path[int(location[batchsize, :])]
                batch.append(batch_tem)
                imgs_As = []
                imgs_Bs = []
                for img in batch:
                    img = self.imread(img)
                    h, w = img.shape
                    half_w = int(w / 2)
                    img_data = img[:, :half_w]
                    img_label = img[:, half_w:]
                    if not is_testing and np.random.random() < 0.5:
                        img_data = np.fliplr(img_data)
                        img_label = np.fliplr(img_label)
                    if not is_testing and np.random.random() > 0.5:
                        img_data = np.flipud(img_data)
                        img_label = np.flipud(img_label)
                    ran = np.random.random()
                    if not is_testing and ran < 0.5:
                        img_data = np.rot90(img_data, 1)
                        img_label = np.rot90(img_label, 1)
                    if not is_testing and ran > 0.5:
                        img_data = np.rot90(img_data, -1)
                        img_label = np.rot90(img_label, -1)
                    img_label = img_label - np.min(img_label)
                    img_label = img_label / np.max(img_label)
                    img_data = img_data - np.min(img_data)
                    img_data = img_data / np.max(img_data)
                    img_data = img_data.astype('float32')
                    img_label = img_label.astype('float32')
                    img_data = img_data.reshape(1, h, half_w)
                    img_label = img_label.reshape(1, h, half_w)
                    imgs_As.append(img_data)
                    imgs_Bs.append(img_label)
            imgs_As = np.array(imgs_As)
            imgs_Bs = np.array(imgs_Bs)
            yield imgs_As, imgs_Bs



    def imread(self, path):
        #return scipy.misc.imread(path, mode='RGB').astype(np.float)[:,:,0]
        return tifffile.imread(path).astype(float)