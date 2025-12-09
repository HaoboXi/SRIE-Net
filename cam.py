import argparse
import os

import cv2

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from grad_cam.utils import GradCAM, show_cam_on_image, center_crop_img
from config import cfg
from model import make_model

class ReshapeTransform:
    def __init__(self, model):

        input_size = model.base.patch_embed.img_size
        patch_size = model.base.patch_embed.patch_size
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):

        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        result = result.permute(0, 3, 1, 2)
        return result

def main(cfg):
    model = make_model(cfg, num_class=150, camera_num=15, view_num=1)

    weights_path = "/disk/wr/TransReID_transfg_irt/logs/prcc/transformer_best.pth"                         # nkup
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    target_layers = [model.base.blocks[-1].norm1]

    data_transform = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


    img_list=['cropped_rgb019.jpg', 'cropped_rgb046.jpg', 'cropped_rgb052.jpg', 'cropped_rgb076.jpg']

    for i in img_list:

        img_path = "/disk/wr/TransReID_transfg_irt/Visualize/heatmap/HSFE/{}".format(i)
 

        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)


        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)

        img = cv2.resize(img, (128, 384))

        img_tensor = data_transform(img)

        input_tensor = torch.unsqueeze(img_tensor, dim=0)

        cam = GradCAM(model=model,
                      target_layers=target_layers,
                      use_cuda=False,
                      reshape_transform=ReshapeTransform(model))
 
        target_category = None
   

        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)

        plt.axis('off')
        plt.imshow(visualization)

        plt.savefig("/disk/wr/TransReID_transfg_irt/Visualize/heatmap/HSFE/igcl-{}.jpg".format(i),transparent=True)








if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="123")
    parser.add_argument("--config_file", default="/disk/wr/TransReID_transfg_irt/configs/Prcc/vit_transreid_stride_384.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR  
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID


    main(cfg)
