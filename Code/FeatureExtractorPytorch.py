import os
import torch
from torchvision import models
import numpy as np
from keras.preprocessing import image
import multiprocessing as mp
os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def get_img(path):
    img = image.load_img(path)
    x = image.img_to_array(img, data_format='channels_first')
    channel, s1, s2 = x.shape
    if s1 < 224 or s2 < 224:
        target_size = (max(224, s1), max(224, s2))
        img = image.load_img(path, target_size=target_size)
        x = image.img_to_array(img, data_format='channels_first')
    if len(x.shape) != 3:
        print('[Image`s shape gets wrong. FileName: {}, Shape: {}]'.format(path, x.shape))
        return False, x
    else:
        return True, x


class ResNet152Extractor:
    def __init__(self):
        base_model = models.resnet152(pretrained=True)
        modules = list(base_model.children())[:-1]
        self.base_model = torch.nn.Sequential(*modules)
        for p in self.base_model.parameters():
            p.requires_grad = False
        if torch.cuda.is_available():
            self.base_model = self.base_model.cuda()
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.base_model.eval()

    def get_feature(self, img_path_sequence):
        self.base_model.eval()
        sequence_size = len(img_path_sequence)
        cpu_num = mp.cpu_count() - 1
        chunk_size = 8
        index = 0
        feature_sequence = []
        while True:
            start_idx, end_idx = 512 * index, 512 * (index + 1)
            index += 1
            if start_idx >= sequence_size:
                break
            img_paths = img_path_sequence[start_idx: end_idx]
            img_sequence = []
            with mp.Pool(processes=cpu_num) as pool:
                records = pool.map(get_img, img_paths, chunk_size)
            pool.close()
            for i, record in enumerate(records):
                correct_or, img = record
                if correct_or:
                    img_sequence.append(img)
                else:
                    raise Exception('LoadImgError')
            for img in img_sequence:
                x = np.expand_dims(img, 0)
                x = x / 255.
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                x[:, 0, :, :] -= mean[0]
                x[:, 1, :, :] -= mean[1]
                x[:, 2, :, :] -= mean[2]
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
                x = torch.from_numpy(x).float()
                if torch.cuda.is_available():
                    x = x.cuda()
                feature = self.base_model(x)
                feature = feature.mean(-1).mean(-1)
                feature = feature.data.cpu().numpy()
                feature_sequence.append(feature)
        feature_sequence = np.concatenate(feature_sequence, 0)
        return feature_sequence
