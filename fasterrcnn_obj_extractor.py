import torch
import numpy as np
import cv2
import h5py
import os
import torchvision.transforms as T
from PIL import Image
import torchvision.ops.roi_align as roi_align
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import urllib.request

# Initialize the global features variable
features = None
default_zero_tensor = None
default_zero_numpy = np.zeros((2048,))
default_zero_padding = [0] * 2048

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load Faster R-CNN model from PyTorch
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
model.to(device)

# Feature extractor
def save_features(mod, inp, outp):
    global features, default_zero_tensor
    if outp is not None:
        features = outp.to(device)
    else:
        if default_zero_tensor is None:
            default_zero_tensor = torch.zeros(1, 2048, 1, 1).to(device)
        features = default_zero_tensor

# Hook the feature extractor layer
layer_to_hook = 'model.backbone.body.layer4'
for name, layer in model.named_modules():
    if name == layer_to_hook:
        layer.register_forward_hook(save_features)

msvd_frame_loader='/mnt/4/haochen/YOLOv5_object_feature_extraction/frames'
msvd_obj_loader='/mnt/4/haochen/YOLOv5_object_feature_extraction/obj_feats'
frame_folder_list=[os.path.join(msvd_frame_loader,frame_folder_name) for frame_folder_name in os.listdir(msvd_frame_loader)]

with torch.no_grad():
    for frame_folder_path in frame_folder_list:
        vid = os.path.basename(frame_folder_path)
        frame_list = [os.path.join(frame_folder_path, frame_name) for frame_name in os.listdir(frame_folder_path)]
        frame_list = [Image.open(frame) for frame in frame_list]

        roi_align_out_per_video = []
        roi_align_boxs_per_video = []

        for img in frame_list:
            img_tensor = T.ToTensor()(img).to(device)
            results = model(img_tensor.unsqueeze(0))[0]

            roi_align_out_per_frame = []
            for j, box in enumerate(results['boxes']):
                if len(roi_align_out_per_frame) == 5:
                    break
                if features is not None:
                    roi_align_out = roi_align(features.to(device), [box.unsqueeze(0).to(device)], output_size=1, spatial_scale=1.0, aligned=True)
                else:
                    roi_align_out = None

                if roi_align_out is not None:
                    roi_align_out_per_frame.append(torch.squeeze(roi_align_out).cpu().numpy())
                else:
                    roi_align_out_per_frame.append(default_zero_numpy)

            if len(roi_align_out_per_frame) < 5:
                for y in range(len(roi_align_out_per_frame), 5):
                    if default_zero_tensor is not None:
                        zero_padding = [0] * default_zero_tensor.shape[1]
                    else:
                        zero_padding = default_zero_padding
                    roi_align_out_per_frame.append(zero_padding)

            roi_align_out_per_frame = np.stack(roi_align_out_per_frame)
            roi_align_out_per_video.append(roi_align_out_per_frame)

            roi_align_boxs_per_frame = []
            img_ylen, img_xlen = img_tensor.shape[1], img_tensor.shape[2]
            for i, box in enumerate(results['boxes']):
                if len(roi_align_boxs_per_frame) == 5:
                    break
                xywh = box.cpu().numpy()
                xywh[0] = round(xywh[0] / img_xlen, 6)
                xywh[1] = round(xywh[1] / img_ylen, 6)
                xywh[2] = round(xywh[2] / img_xlen, 6)
                xywh[3] = round(xywh[3] / img_ylen, 6)
                roi_align_boxs_per_frame.append(xywh)
            if len(roi_align_boxs_per_frame) < 5:
                for y in range(len(roi_align_boxs_per_frame), 5):
                    zero_padding = [0] * 4
                    roi_align_boxs_per_frame.append(zero_padding)

            roi_align_boxs_per_frame = np.stack(roi_align_boxs_per_frame)
            roi_align_boxs_per_video.append(roi_align_boxs_per_frame)

        roi_align_out_per_video = np.stack(roi_align_out_per_video)
        roi_align_boxs_per_video = np.stack(roi_align_boxs_per_video)

        msvd_obj_path = os.path.join(msvd_obj_loader, 'msvd_obj_fasterrcnn.h5')
        with h5py.File(msvd_obj_path, 'a') as h5file:
            group = h5file.create_group(vid)
            group.create_dataset('obj_features', data=roi_align_out_per_video)
            group.create_dataset('obj_boxes', data=roi_align_boxs_per_video)
