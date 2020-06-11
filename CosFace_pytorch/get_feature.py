import cv2
import sys
import numpy as np
import time
import scipy.io as sio
import os
from tqdm import tqdm
from PIL import Image

from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import net
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def extractDeepFeature(img, model, is_gray):
    #print('img',img.size)
    if is_gray:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5,), std=(0.5,))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
    img, img_ = transform(img), transform(F.hflip(img))
    img, img_ = img.unsqueeze(0).to('cuda'), img_.unsqueeze(0).to('cuda')
    #print('img',img.shape)
    ft = torch.cat((model(img), model(img_)), 1)[0].to('cpu')
    ft=ft.detach().numpy()
    return ft




def get_featurs(model, test_list):
    features = None
    cnt = 0
    pbar = tqdm(len(test_list))
    for i, img_path in enumerate(test_list):
        pbar.update(1)
        img =  Image.open(img_path).convert('RGB')
        image=img.resize((96,112))
        if image is None:
            print('read {} error'.format(img_path))
        else:
            cnt += 1
            feature= extractDeepFeature(image, model, False)
            #print(feature.shape)
            #print(feature.type)
            '''
            if i==10:
                break
            '''
            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))
    return features, cnt

def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
         # key = each.split('/')[1]
        '''
        if i==10:
            break
        '''
        fe_dict[each] = features[i]
    return fe_dict

if __name__ == '__main__':

    data_dir = '/media/Data/yangty/arcface/data/test_data/'
    name_list = [name for name in os.listdir(data_dir)]
    img_paths = [data_dir+name for name in os.listdir(data_dir)]
    print('Images number:', len(img_paths))

    model_path='ACC99.28.pth'
    model=net.sphere().to('cuda')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    s = time.time()
    features, cnt = get_featurs(model, img_paths)
    np.save('feature.npy',features)
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))
    fe_dict = get_feature_dict(name_list, features)
    print('Output number:', len(fe_dict))
    sio.savemat('cosface.mat', fe_dict)

