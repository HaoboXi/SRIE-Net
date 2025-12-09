import argparse
import scipy.io
import torch
import numpy as np
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from PIL import Image, ImageOps

from datasets.nkup import NKUP 
from datasets.ltcc import Ltcc
from datasets.prcc import PRCC
from datasets.vcclothes import VCClothes
__factory = {

    'ltcc': Ltcc,
    'prcc': PRCC,
    'nkup': NKUP,
    'vcclothes': VCClothes,
}
query_path=[]
gallery_path=[]
dataset = __factory['vcclothes'](root="/disk/wr/data/")    
for i in dataset.query:
    query_path.append(i[0])
for i in dataset.gallery:
    gallery_path.append(i[0])

parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=1, type=int, help='test_image_index')
parser.add_argument('--test_dir',default="/disk/wr/data/VC-Clothes/",type=str, help='./test_data')
opts = parser.parse_args()

def imshow(path, is_same,title=None):
    """Imshow for Tensor."""

    im = Image.open(path)

    im = im.resize((128, 256))
    if title: im = ImageOps.expand(im, border=6,fill='black')
    if not title:
        if is_same:
            im = ImageOps.expand(im, border=6, fill='green')
        else:
            im = ImageOps.expand(im, border=6, fill='red')

    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  
result = scipy.io.loadmat('/disk/wr/TransReID_transfg_irt/logs/SIEN-vcclothes-pytorch_result.mat')
query_feature = torch.FloatTensor(result['qf'])
query_cam = result['q_camids'][0]
query_label_all = result['q_pids'][0]
gallery_feature = torch.FloatTensor(result['gf'])
gallery_cam = result['g_camids'][0]
gallery_label = result['g_pids'][0]

multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1,1)

    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    index = np.argsort(score) 
    index = index[::-1]

    query_index = np.argwhere(gl==ql)

    camera_index = np.argwhere(gc==qc)

  
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index

# i = opts.query_index
prcc_list = [3001,3004,3090,3097,3103,3255,3499,3512,3536]
celeblight_list = [3,31, 37, 114, 146, 187, 223, 307, 311, 496, 546, 637, 801, 874]
vcclothes_list = [0,1, 7, 18, 25, 26, 29, 36, 41, 122, 123, 192, 197, 200, 203, 247, 260, 270, 274, 282, 311, 324,  329, 331, 332, 333, 340, 344, 347, 358, 361, 362,  458, 462, 465, 467, 499 ]
ltcc_list = [7, 9, 16, 31, 54, 77, 89, 208, 213, 231, 237, 247, 381]
nkup_list = [1]

for i in vcclothes_list:
    id = i
    index = sort_img(query_feature[i],query_label_all[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
    img_query_path = query_path[i]
    query_label = query_label_all[i]
    print(img_query_path, query_label)
    print('Top 10 images are as follow:')
    try: 
        fig = plt.figure(figsize=(16,4))
        ax = plt.subplot(1,11,1)
        ax.axis('off')
        imshow(img_query_path,None,'query')
        for i in range(10):
            ax = plt.subplot(1,11,i+2)
            ax.axis('off')

            img_gallery_path = gallery_path[index[i]]
            label = gallery_label[index[i]]
            is_same = label == query_label
            imshow(img_gallery_path,is_same)
            if label == query_label:
                ax.set_title('%d'%(i+1), color='green')
            else:
                ax.set_title('%d'%(i+1), color='red')
            print(img_gallery_path)
    except RuntimeError:
        print('If you want to see the visualization of the ranking result, graphical user interface is needed.')
    fig.savefig("/disk/wr/TransReID_transfg_irt/logs/SIEN-baseline-vcclothes-rank{}.png".format(id),transparent=True) 


