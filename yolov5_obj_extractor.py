import torch
import numpy as np
import cv2
import h5py
import os
import torchvision.transforms as T
from PIL import Image
import torchvision.ops.roi_align as roi_align

# Load YOLOv5 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5l6', pretrained=True)
model.eval()
print()

# Feature extractor
def save_features(mod, inp, outp):
    global features
    features=outp
#钩子
layer_to_hook = 'model.11.cv2.act'
for name, layer in model.model.model.named_modules():
    if name == layer_to_hook:
        layer.register_forward_hook(save_features)

msvd_frame_loader='/mnt/4/haochen/YOLOv5_object_feature_extraction/frames'
msvd_obj_loader='/mnt/4/haochen/YOLOv5_object_feature_extraction/obj_feats'
# 存放所有video帧文件夹路径
frame_folder_list=[os.path.join(msvd_frame_loader,frame_folder_name) for frame_folder_name in os.listdir(msvd_frame_loader)]
# print(f'frame_folder_list: {frame_folder_list}') #['MSVD_VTT/MSVD/MSVD_save_frames\\_0nX-El-ySo_83_93',...]

with torch.no_grad():
    for frame_folder_path in frame_folder_list: ##取出每个video帧的文件夹
        # print(f'frame_folder_path: {frame_folder_path}') #MSVD_VTT/MSVD/MSVD_save_frames\_0nX-El-ySo_83_93
        vid = os.path.basename(frame_folder_path)
        # print(f'vid:{vid}')
        frame_list = [os.path.join(frame_folder_path, frame_name) for frame_name in os.listdir(frame_folder_path)]
        # print(f'frame_list:{frame_list}')  #存放该video每一帧的路径 ['MSVD_VTT/MSVD/MSVD_save_frames\\_0nX-El-ySo_83_93\\frame_0.jpg',...
        frame_list = [Image.open(frame) for frame in frame_list]  # 取到每一帧的图片

        roi_align_out_per_video=[] #存放该video所有帧的obj特征
        roi_align_boxs_per_video = []  # 存放该video所有帧的obj位置信息
        # 处理该video的每一张图片
        for img in frame_list:
            img_tensor=T.ToTensor()(img)
            results = model(img)
            # print(f'features.shape:{features.shape}') #features.shape:torch.Size([1, 1024, 6, 10])
            # print(f'tensor.shape:{tensor.shape}') #img.shape:torch.Size([3, 224, 398])
            spat_scale = min(features.shape[2] / img_tensor.shape[1], features.shape[3] / img_tensor.shape[2])

            roi_align_out_per_frame = []  # 存放该帧所有obj特征
            for j, box in enumerate(results.xyxy[0].cpu().numpy()):  # for each box
                if len(roi_align_out_per_frame) == 5:  # max object per frame is 5
                    break
                roi_align_out = roi_align(features, [results.xyxy[0][:, :4][j:j + 1]], output_size=1,
                                          spatial_scale=spat_scale, aligned=True)
                # print(f'roi_align_out:{roi_align_out.shape}') #torch.Size([1, 1024, 1, 1])
                roi_align_out_per_frame.append(torch.squeeze(roi_align_out).cpu().numpy())
            if len(roi_align_out_per_frame) < 5:  # add zero padding if less than 5 object
                for y in range(len(roi_align_out_per_frame), 5):
                    zero_padding = [0] * 1024  # length of the roi_align_out is also 1024, hardcoded for now
                    roi_align_out_per_frame.append(zero_padding)

            roi_align_out_per_frame = np.stack(roi_align_out_per_frame)  # #[5,1024]
            roi_align_out_per_video.append(roi_align_out_per_frame)

            #存储xywh，注意归一化
            roi_align_boxs_per_frame = [] #存放该帧所有box位置信息
            img_ylen = img_tensor.shape[1]  #原图像y长
            img_xlen = img_tensor.shape[2]  #原图像x长
            for i, box in enumerate(results.xywh[0].cpu().numpy()):
                if len(roi_align_boxs_per_frame) == 5:
                    break
                xywh = box[:4]
                xywh[0] = round(xywh[0] / img_xlen, 6)
                xywh[1] = round(xywh[1] / img_ylen, 6)
                xywh[2] = round(xywh[2] / img_xlen, 6)
                xywh[3] = round(xywh[3] / img_ylen, 6)
                roi_align_boxs_per_frame.append(xywh)
            if len(roi_align_boxs_per_frame) < 5:  # add zero padding if less than 5 object
                for y in range(len(roi_align_boxs_per_frame), 5):
                    zero_padding = [0] * 4  # length of the roi_align_out is also 1024, hardcoded for now
                    roi_align_boxs_per_frame.append(zero_padding)

            roi_align_boxs_per_frame=np.stack(roi_align_boxs_per_frame) #[5,4]
            roi_align_boxs_per_video.append(roi_align_boxs_per_frame)

        roi_align_out_per_video=np.stack(roi_align_out_per_video)  #(6, 5, 1024)
        roi_align_boxs_per_video=np.stack(roi_align_boxs_per_video)  #(6, 5, 4)

        # 存储特征到HDF5文件
        msvd_obj_path=os.path.join(msvd_obj_loader, 'msvd_obj_yolov5_5.h5')
        with h5py.File(msvd_obj_path, 'a') as h5file:
            group = h5file.create_group(vid)
            # 为每个vid创建两个数据集：features和boxes
            group.create_dataset('obj_features', data=roi_align_out_per_video)
            group.create_dataset('obj_boxes', data=roi_align_boxs_per_video)














