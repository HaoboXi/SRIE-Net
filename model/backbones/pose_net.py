import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from torch import nn
from .hrnet import HRNet
from torch.nn import init



class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SimpleHRNet:
    """
    SimpleHRNet class.

    The class provides a simple and customizable method to load the HRNet network, load the official pre-trained
    weights, and predict the human pose on single images.
    Multi-person support with the YOLOv3 detector is also included (and enabled by default).
    """

    def __init__(self,
                 c,
                 nof_joints,
                 checkpoint_path,
                 model_name='HRNet',
                 resolution=(256, 192),
                 interpolation=cv2.INTER_CUBIC,
                 multiperson=False,
                 return_heatmaps=True,
                 return_bounding_boxes=False,
                 max_batch_size=64,
                 yolo_model_def="./models/detectors/yolo/config/yolov3.cfg",
                 yolo_class_path="./models/detectors/yolo/data/coco.names",
                 yolo_weights_path="./models/detectors/yolo/weights/yolov3.weights",
                 device=torch.device("cuda")):
        """
        Initializes a new SimpleHRNet object.
        HRNet (and YOLOv3) are initialized on the torch.device("device") and
        its (their) pre-trained weights will be loaded from disk.

        Args:
            c (int): number of channels (when using HRNet model) or resnet size (when using PoseResNet model).
            nof_joints (int): number of joints.
            checkpoint_path (str): path to an official hrnet checkpoint or a checkpoint obtained with `train_coco.py`.
            model_name (str): model name (HRNet or PoseResNet).
                Valid names for HRNet are: `HRNet`, `hrnet`
                Valid names for PoseResNet are: `PoseResNet`, `poseresnet`, `ResNet`, `resnet`
                Default: "HRNet"
            resolution (tuple): hrnet input resolution - format: (height, width).
                Default: (384, 288)
            interpolation (int): opencv interpolation algorithm.
                Default: cv2.INTER_CUBIC
            multiperson (bool): if True, multiperson detection will be enabled.
                This requires the use of a people detector (like YOLOv3).
                Default: True
            return_heatmaps (bool): if True, heatmaps will be returned along with poses by self.predict.
                Default: False
            return_bounding_boxes (bool): if True, bounding boxes will be returned along with poses by self.predict.
                Default: False
            max_batch_size (int): maximum batch size used in hrnet inference.
                Useless without multiperson=True.
                Default: 16
            yolo_model_def (str): path to yolo model definition file.
                Default: "./models/detectors/yolo/config/yolov3.cfg"
            yolo_class_path (str): path to yolo class definition file.
                Default: "./models/detectors/yolo/data/coco.names"
            yolo_weights_path (str): path to yolo pretrained weights file.
                Default: "./models/detectors/yolo/weights/yolov3.weights.cfg"
            device (:class:`torch.device`): the hrnet (and yolo) inference will be run on this device.
                Default: torch.device("cpu")
        """

        self.c = c
        self.nof_joints = nof_joints
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.resolution = resolution 
        self.interpolation = interpolation
        self.multiperson = multiperson
        self.return_heatmaps = return_heatmaps
        self.return_bounding_boxes = return_bounding_boxes
        self.max_batch_size = max_batch_size
        self.yolo_model_def = yolo_model_def
        self.yolo_class_path = yolo_class_path
        self.yolo_weights_path = yolo_weights_path
        self.device = device
        self.atten = SEAttention(channel=17,reduction=8)

        if self.multiperson:
            from models.detectors.YOLOv3 import YOLOv3

        if model_name in ('HRNet', 'hrnet'):
            self.model = HRNet(c=c, nof_joints=nof_joints)
        else:
            raise ValueError('Wrong model name.')

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)

        if 'cuda' in str(self.device):
            print("device: 'cuda' - ", end="")

            if 'cuda' == str(self.device):
                print("%d GPU(s) will be used" % torch.cuda.device_count())
                device_ids = None
            else:
                print("GPU(s) '%s' will be used" % str(self.device))
                device_ids = [int(x) for x in str(self.device)[5:].split(',')]

            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        elif 'cpu' == str(self.device):
            print("device: 'cpu'")
        else:
            raise ValueError('Wrong device name.')

        self.model = self.model.to(device)
        self.atten = self.atten.to(device)
        self.model.eval()

        if not self.multiperson:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        else:
            self.detector = YOLOv3(model_def=yolo_model_def,
                                   class_path=yolo_class_path,
                                   weights_path=yolo_weights_path,
                                   classes=('person',),
                                   max_batch_size=self.max_batch_size,
                                   device=device)
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resolution[0], self.resolution[1])), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def predict(self, image):
        """
        Predicts the human pose on a single image or a stack of n images.

        Args:
            image (:class:`np.ndarray`):
                the image(s) on which the human pose will be estimated.

                image is expected to be in the opencv format.
                image can be:
                    - a single image with shape=(height, width, BGR color channel)
                    - a stack of n images with shape=(n, height, width, BGR color channel)

        Returns:
            :class:`np.ndarray` or list:
                a numpy array containing human joints for each (detected) person.

                Format:
                    if image is a single image:
                        shape=(# of people, # of joints (nof_joints), 3);  dtype=(np.float32).
                    if image is a stack of n images:
                        list of n np.ndarrays with
                        shape=(# of people, # of joints (nof_joints), 3);  dtype=(np.float32).

                Each joint has 3 values: (y position, x position, joint confidence).

                If self.return_heatmaps, the class returns a list with (heatmaps, human joints)
                If self.return_bounding_boxes, the class returns a list with (bounding boxes, human joints)
                If self.return_heatmaps and self.return_bounding_boxes, the class returns a list with
                    (heatmaps, bounding boxes, human joints)
        """
        if len(image.shape) == 3:
            return self._predict_single(image)
        elif len(image.shape) == 4:
            return self._predict_batch(image)
        else:
            raise ValueError('Wrong image format.')

    def _predict_single(self, image):
        print("执行这个？_predict_single")
        if not self.multiperson:
            old_res = image.shape
            if self.resolution is not None:
                image = cv2.resize(
                    image,
                    (self.resolution[1], self.resolution[0]),
                    interpolation=self.interpolation
                )

            images = self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(dim=0) # (h,w,c)
            boxes = np.asarray([[0, 0, old_res[1], old_res[0]]], dtype=np.float32)  # [x1, y1, x2, y2]
            heatmaps = np.zeros((1, self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                                dtype=np.float32)

        else:
            detections = self.detector.predict_single(image)

            nof_people = len(detections) if detections is not None else 0
            boxes = np.empty((nof_people, 4), dtype=np.int32)
            images = torch.empty((nof_people, 3, self.resolution[0], self.resolution[1]))  # (height, width)
            heatmaps = np.zeros((nof_people, self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                                dtype=np.float32)

            if detections is not None:
                for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
                    x1 = int(round(x1.item()))
                    x2 = int(round(x2.item()))
                    y1 = int(round(y1.item()))
                    y2 = int(round(y2.item()))
                    correction_factor = self.resolution[0] / self.resolution[1] * (x2 - x1) / (y2 - y1)
                    if correction_factor > 1:
                        center = y1 + (y2 - y1) // 2
                        length = int(round((y2 - y1) * correction_factor))
                        y1 = max(0, center - length // 2)
                        y2 = min(image.shape[0], center + length // 2)
                    elif correction_factor < 1:
                        center = x1 + (x2 - x1) // 2
                        length = int(round((x2 - x1) * 1 / correction_factor))
                        x1 = max(0, center - length // 2)
                        x2 = min(image.shape[1], center + length // 2)

                    boxes[i] = [x1, y1, x2, y2]
                    images[i] = self.transform(image[y1:y2, x1:x2, ::-1])

        if images.shape[0] > 0:
            images = images.to(self.device)

            with torch.no_grad():
                if len(images) <= self.max_batch_size:
                    out = self.model(images)

                else:
                    out = torch.empty(
                        (images.shape[0], self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                        device=self.device
                    )
                    for i in range(0, len(images), self.max_batch_size):
                        out[i:i + self.max_batch_size] = self.model(images[i:i + self.max_batch_size])

            out = out.detach().cpu().numpy()
            pts = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)
            for i, human in enumerate(out):
                heatmaps[i] = human
                for j, joint in enumerate(human):
                    pt = np.unravel_index(np.argmax(joint), (self.resolution[0] // 4, self.resolution[1] // 4))
                    pts[i, j, 0] = pt[0] * 1. / (self.resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                    pts[i, j, 1] = pt[1] * 1. / (self.resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                    pts[i, j, 2] = joint[pt]

        else:
            pts = np.empty((0, 0, 3), dtype=np.float32)

        res = list()
        if self.return_heatmaps:
            res.append(heatmaps)
        if self.return_bounding_boxes:
            res.append(boxes)
        res.append(pts)

        if len(res) > 1:
            return res
        else:
            return res[0]

    def _predict_batch(self, images):
        if not self.multiperson:
            old_res = images[0].shape
            images = images
            boxes = np.repeat(
                np.asarray([[0, 0, old_res[2], old_res[1]]], dtype=np.float32), len(images), axis=0
            ) 
            heatmaps = np.zeros((len(images), self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                                dtype=np.float32)

        else:
            image_detections = self.detector.predict(images)

            base_index = 0
            nof_people = int(np.sum([len(d) for d in image_detections if d is not None]))
            boxes = np.empty((nof_people, 4), dtype=np.int32)
            images_tensor = torch.empty((nof_people, 3, self.resolution[0], self.resolution[1]))  # (height, width)
            heatmaps = np.zeros((nof_people, self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                                dtype=np.float32)

            for d, detections in enumerate(image_detections):
                image = images[d]
                if detections is not None and len(detections) > 0:
                    for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
                        x1 = int(round(x1.item()))
                        x2 = int(round(x2.item()))
                        y1 = int(round(y1.item()))
                        y2 = int(round(y2.item()))
                        correction_factor = self.resolution[0] / self.resolution[1] * (x2 - x1) / (y2 - y1)
                        if correction_factor > 1:
                            center = y1 + (y2 - y1) // 2
                            length = int(round((y2 - y1) * correction_factor))
                            y1 = max(0, center - length // 2)
                            y2 = min(image.shape[0], center + length // 2)
                        elif correction_factor < 1:
                            center = x1 + (x2 - x1) // 2
                            length = int(round((x2 - x1) * 1 / correction_factor))
                            x1 = max(0, center - length // 2)
                            x2 = min(image.shape[1], center + length // 2)

                        boxes[base_index + i] = [x1, y1, x2, y2]
                        images_tensor[base_index + i] = self.transform(image[y1:y2, x1:x2, ::-1])

                    base_index += len(detections)

            images = images_tensor

        images = images.to(self.device)

        if images.shape[0] > 0:
            with torch.no_grad():
                if len(images) <= self.max_batch_size:
                    out = self.model(images)
                else:
                    out = torch.empty(
                        (images.shape[0], self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                        device=self.device
                    )
                    for i in range(0, len(images), self.max_batch_size):
                        out[i:i + self.max_batch_size] = self.model(images[i:i + self.max_batch_size])
            out = out.detach().cpu().numpy()
            pts = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)
            for i, human in enumerate(out):
                heatmaps[i] = human
                for j, joint in enumerate(human):
                    pt = np.unravel_index(np.argmax(joint), (self.resolution[0] // 4, self.resolution[1] // 4)) 
                    pts[i, j, 0] = pt[0] * 1. / (self.resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                    pts[i, j, 1] = pt[1] * 1. / (self.resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                    pts[i, j, 2] = joint[pt]

            if self.multiperson:
                if self.return_heatmaps:
                    heatmaps_batch = []
                if self.return_bounding_boxes:
                    boxes_batch = []
                pts_batch = []
                index = 0
                for detections in image_detections:
                    if detections is not None:
                        pts_batch.append(pts[index:index + len(detections)])
                        if self.return_heatmaps:
                            heatmaps_batch.append(heatmaps[index:index + len(detections)])
                        if self.return_bounding_boxes:
                            boxes_batch.append(boxes[index:index + len(detections)])
                        index += len(detections)
                    else:
                        pts_batch.append(np.zeros((0, self.nof_joints, 3), dtype=np.float32))
                        if self.return_heatmaps:
                            heatmaps_batch.append(np.zeros((0, self.nof_joints, self.resolution[0] // 4,
                                                            self.resolution[1] // 4), dtype=np.float32))
                        if self.return_bounding_boxes:
                            boxes_batch.append(np.zeros((0, 4), dtype=np.float32))
                if self.return_heatmaps:
                    heatmaps = heatmaps_batch
                if self.return_bounding_boxes:
                    boxes = boxes_batch
                pts = pts_batch

            else:
                pts = np.expand_dims(pts, axis=1)

        else:
            print("len(img) <= 0")
            boxes = np.asarray([], dtype=np.int32)
            if self.multiperson:
                pts = []
                for _ in range(len(image_detections)):
                    pts.append(np.zeros((0, self.nof_joints, 3), dtype=np.float32))
            else:
                raise ValueError 

        res = list()
        if self.return_heatmaps:
            res.append(heatmaps)
        if self.return_bounding_boxes:
            res.append(boxes)
        res.append(pts)

        if len(res) > 1:
            return res
        else:
            return res[0]
