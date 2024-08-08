import os


import numpy as np
import torch
import cv2
from PIL import Image
import imutils as imutils
# from visdom import Visdom

from infer_MCL import cam_maxnorm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def get_Pseudo_label(cam, label,T=0.4):
    cam =cam_maxnorm(cam)
    # cam=cam_maxnorm(cam)
    cam *= label.unsqueeze(2).unsqueeze(2)
    cam = cam.cpu().detach().squeeze().numpy().astype(np.half)
    
    cam[:,0,:,:] = T 
    predict = np.argmax(cam, axis=1).astype(np.uint8)
    #返回tensor
    return torch.from_numpy(predict)
    
    
def get_crf_label(img_batch, cam, label, T=0.4):
    cam = cam_maxnorm(cam)
    cam *= label.unsqueeze(2).unsqueeze(2)
    cam = cam.cpu().detach().squeeze().numpy().astype(np.half)
    
    cam[:,0,:,:] = T 

    cam_list = []
    for idx, orig_img in enumerate(img_batch):
        # print(orig_img.shape)
        # print(cam[idx].shape)
        
        orig_img_np = orig_img.transpose((1, 2, 0))
        orig_img_np = np.ascontiguousarray(orig_img_np)
        cam_i = imutils.crf_inference(orig_img_np, cam[idx], t=6)

       
        cam_list.append(cam_i)

    cam_batch = np.stack(cam_list)
    predict = np.argmax(cam_batch, axis=1).astype(np.uint8)
    #返回tensor
    return torch.from_numpy(predict)




def get_cam_region(predict):
    # cam = cam.cpu().data.numpy()
    #cam = cam_nornalize(cam)
    # cam = label_Filtering(cam,label)
    # cam = thresholding_Filtering(cam,thres)
    # # cam[cam ==0] = 0
    # cam[cam > 0] = 1
    #np 转 tensor
    # cam = torch.from_numpy(cam).cuda()  
    # cam =cam_maxnorm(cam)
    # # cam=cam_maxnorm(cam)
    # cam *= label.unsqueeze(2).unsqueeze(2)
    # cam = cam.cpu().detach().squeeze().numpy().astype(np.half)
    
    # cam[:,0,:,:] = thres 
    # predict = np.argmax(cam, axis=1).astype(np.uint8)

    predict[predict == 0] = 0
    predict[predict > 0] = 1
    
    return predict

def get_maskImage(ori_images,Pseudo_label_mask):
    
    
    mask = (Pseudo_label_mask != 0).unsqueeze(1).repeat(1, 3, 1, 1)  # Expand mask to have the same number of channels as the image
    device = ori_images.device  # Get the device of the images
    masked_images = torch.where(mask, ori_images, torch.tensor([0.]).to(device))  # Replace image pixels where mask is True with 0, and move the replacement tensor to the same device as the images
    return masked_images

def get_mix_image(img1, img2, sam, label,thres=0.5,decay=0.9):
    # viz = Visdom()
    # viz.images(img1, win='img1')
    # viz.images(img2, win='img2')
    sam = get_cam_region(sam)
    sam = sam.to(device)
    # sam = torch.from_numpy(sam).cuda()
    # 对imag1做一些数据增强
   

    img1 = img1 * sam[:, None, :, :]
    negsam=1-sam
    img2 = img2 * negsam[:, None, :, :]
    # viz.images(img1, win='img3')
    # viz.images(img2, win='img4')
    # mix_img = img1 + img2
    # viz.images(mix_img, win='mix_img')
    return img1 + img2

def Switching_imglist_order(img):
    #重新排序
    newimg = img.clone()
    for i in range(img.shape[0]):
        newimg[i,:,:,:] = img[img.shape[0]-i-1,:,:,:]
    return newimg

def Switching_label_order(label):
    #重新排序
    newlabel = label.clone()
    for i in range(label.shape[0]):
        newlabel[i,:] = label[label.shape[0]-i-1,:]
    return newlabel

def cam_nornalize(cam):
    # cam = cam.
    # norm_SGC = np.sum(cam, axis=0)
    # norm_SGC = cam.cpu().data.numpy()
    norm_SGC = cam
    norm_SGC[norm_SGC < 0] = 0
    SGC_max = np.max(norm_SGC, (2,3), keepdims=True)
    SGC_min = np.min(norm_SGC, (2,3), keepdims=True)
    norm_SGC[norm_SGC < SGC_min+1e-6] = 0
    norm_SGC = (norm_SGC-SGC_min-1e-6) / (SGC_max - SGC_min + 1e-6)
    return norm_SGC
     
def label_Filtering(cam,label):
    
    # label = label.cpu().data.numpy
    
    for i in range(cam.shape[0]):
       
        label_idx = np.where(label[i] == 0)
        cam[i,label_idx,:,:] = 0 
    # label_inx = torch.where(label == 0)#.cpu().data.numpy()
    # cam[label_idx] = 0
  
           
    return cam

    

def thresholding_Filtering(predict_dict,threshold =0.3):
    
   
    predict_dict[:,0,:,:] = threshold 
    Pseudo_label = np.argmax(predict_dict, axis=1).astype(np.uint8)
    return Pseudo_label
   
def get_mix_label(label_ori,label_new):
    label_mix = label_ori.clone()
    label_mix[label_mix==0]=label_new[label_mix==0]
    return label_mix
    
   
def Random_mix_process(SAM_MAP,img):
    
    b,h,w = SAM_MAP.shape
    new_SAM_MAP = torch.zeros_like(SAM_MAP)
    new_img = torch.zeros_like(img)

    # 将SAM_MAP随机resize 0.5-2 之间,并将其随机位置地放入新的SAM_MAP中
    # 调换img成cv2格式
    img = img.cpu().data.numpy()
    img = np.transpose(img, (0, 2, 3, 1))

    for i in range(b):
        # 随机resize
        scale = np.random.uniform(0.5,2)
        SAM_MAP_resize = cv2.resize(SAM_MAP[i].cpu().data.numpy(),(int(w*scale),int(h*scale)),interpolation=cv2.INTER_LINEAR)
        
        img_resize = cv2.resize(img[i],(int(w*scale),int(h*scale)),interpolation=cv2.INTER_LINEAR)
        #如果尺寸一样大，直接赋值
        if SAM_MAP_resize.shape == SAM_MAP[i].shape:
            new_SAM_MAP[i] = torch.from_numpy(SAM_MAP_resize)
            new_img[i] = torch.from_numpy(img_resize.transpose(2,0,1))
        else:
            #如果resize后的图像比原图像大，就将其裁剪成原图像大小,再将其放入新的SAM_MAP的随机位置中
            if SAM_MAP_resize.shape[0] >= h and SAM_MAP_resize.shape[1] >= w:
                x = np.random.randint(0,SAM_MAP_resize.shape[0]-h)
                y = np.random.randint(0,SAM_MAP_resize.shape[1]-w)
                new_SAM_MAP[i] = torch.from_numpy(SAM_MAP_resize[x:x+h,y:y+w])
                new_img[i] = torch.from_numpy(img_resize[x:x+h,y:y+w,:].transpose(2,0,1))
                
            #如果resize后的图像比原图像小，就将其放入新的SAM_MAP的随机位置中
            else:
                x = np.random.randint(0,h-SAM_MAP_resize.shape[0])
                y = np.random.randint(0,w-SAM_MAP_resize.shape[1])
                new_SAM_MAP[i,x:x+SAM_MAP_resize.shape[0],y:y+SAM_MAP_resize.shape[1]] = torch.from_numpy(SAM_MAP_resize)
                new_img[i,:,x:x+SAM_MAP_resize.shape[0],y:y+SAM_MAP_resize.shape[1]] = torch.from_numpy(img_resize.transpose(2,0,1))

    # new_img = torch.from_numpy(new_img).cuda()
    return new_SAM_MAP, new_img


def Random_mix_process_centre(SAM_MAP,img):
    
    b,h,w = SAM_MAP.shape
    new_SAM_MAP = torch.zeros_like(SAM_MAP)
    new_img = torch.zeros_like(img)

    img = img.cpu().data.numpy()
    img = np.transpose(img, (0, 2, 3, 1))

    for i in range(b):
        scale = np.random.uniform(0.5,2)
        SAM_MAP_resize = cv2.resize(SAM_MAP[i].cpu().data.numpy(),(int(w*scale),int(h*scale)),interpolation=cv2.INTER_LINEAR)
        
        img_resize = cv2.resize(img[i],(int(w*scale),int(h*scale)),interpolation=cv2.INTER_LINEAR)
        
        if SAM_MAP_resize.shape == SAM_MAP[i].shape:
            new_SAM_MAP[i] = torch.from_numpy(SAM_MAP_resize)
            new_img[i] = torch.from_numpy(img_resize.transpose(2,0,1))
        else:
            if SAM_MAP_resize.shape[0] >= h and SAM_MAP_resize.shape[1] >= w:
                x = (SAM_MAP_resize.shape[0] - h) // 2
                y = (SAM_MAP_resize.shape[1] - w) // 2
                new_SAM_MAP[i] = torch.from_numpy(SAM_MAP_resize[x:x+h,y:y+w])
                new_img[i] = torch.from_numpy(img_resize[x:x+h,y:y+w,:].transpose(2,0,1))
            else:
                x = (h - SAM_MAP_resize.shape[0]) // 2
                y = (w - SAM_MAP_resize.shape[1]) // 2
                new_SAM_MAP[i,x:x+SAM_MAP_resize.shape[0],y:y+SAM_MAP_resize.shape[1]] = torch.from_numpy(SAM_MAP_resize)
                new_img[i,:,x:x+SAM_MAP_resize.shape[0],y:y+SAM_MAP_resize.shape[1]] = torch.from_numpy(img_resize.transpose(2,0,1))

    return new_SAM_MAP, new_img


import torch
from torchvision import transforms

# def aug_process_rotation(Pseudo_label_batch,img_batch):
#     tensor_images = []
#     tensor_labels = []
#     for i in range(Pseudo_label_batch.shape[0]): # loop over the batch
#         #随机degrees值
#         degrees = np.random.randint(0,360) 
#         # 将图像和标签转换为PIL Image
#         pil_image = transforms.ToPILImage()(img_batch[i])
#         pil_label = transforms.ToPILImage()(Pseudo_label_batch[i].byte()) # convert to byte tensor before converting to PIL image

#         # 旋转图像和标签
#         rotated_image = transforms.functional.rotate(pil_image, degrees)
#         rotated_label = transforms.functional.rotate(pil_label, degrees)

#         # 将旋转后的图像和标签转换回张量形式
#         # tensor_image = transforms.ToTensor()(rotated_image)
#         # For the label, convert it back to a numpy array first, and then to a tensor

#         tensor_label = torch.from_numpy(np.array(rotated_label))
#         tensor_image = torch.from_numpy(np.array(rotated_image))

#         tensor_images.append(tensor_image)
#         tensor_labels.append(tensor_label)

#     tensor_images = torch.stack(tensor_images) # concatenate all tensor_images into a batch
#     tensor_labels = torch.stack(tensor_labels) # concatenate all tensor_labels into a batch
#     tensor_labels = torch.squeeze(tensor_labels, 1) # remove the channel dimension

#     return tensor_images, tensor_labels


import cv2
import numpy as np

def aug_process_rotation(Pseudo_label_batch,img_batch):
    tensor_images = []
    tensor_labels = []
    for i in range(Pseudo_label_batch.shape[0]): # loop over the batch
        #随机degrees值
        degrees = np.random.randint(-90,90) 

        # convert tensor to numpy
        img_np = img_batch[i].cpu().permute(1, 2, 0).numpy()
        Pseudo_label_np = Pseudo_label_batch[i].cpu().numpy()

        # get image height and width
        h, w = img_np.shape[:2]

        # calculate the center of the image
        center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, degrees, 1)
        rotated_image_np = cv2.warpAffine(img_np, M, (w, h))
        rotated_label_np = cv2.warpAffine(Pseudo_label_np, M, (w, h), flags=cv2.INTER_NEAREST)

        # convert back to tensor
        tensor_image = torch.from_numpy(rotated_image_np).permute(2, 0, 1) # HWC to CHW
        tensor_label = torch.from_numpy(rotated_label_np)
     
        tensor_images.append(tensor_image)
        tensor_labels.append(tensor_label)

    tensor_images = torch.stack(tensor_images).to(Pseudo_label_batch.device) # concatenate all tensor_images into a batch
    tensor_labels = torch.stack(tensor_labels).to(Pseudo_label_batch.device) # concatenate all tensor_labels into a batch

    return tensor_images, tensor_labels



import random
import math
import torch

def random_erase_batch(img_batch, label_batch, p=0.5, scale=(0.02,0.33), ratio=(0.3,3.3), value=0):
    for idx in range(img_batch.size()[0]):  # Loop over the batch
        if random.uniform(0, 1) > p:
            continue

        img = img_batch[idx]
        label = label_batch[idx]

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                
                img[0, x1:x1+h, y1:y1+w] = value
                img[1, x1:x1+h, y1:y1+w] = value
                img[2, x1:x1+h, y1:y1+w] = value
                label[x1:x1+h, y1:y1+w] = value

    return img_batch, label_batch


def merge_saliency_maps(primary, auxiliary):
    # 创建掩膜，对应于主Saliency Map为0的位置
    mask = primary == 0

    # 使用辅助Saliency Map的值来替换
    merged = primary.clone()
    merged[mask] = auxiliary[mask]
    
    return merged



import cv2
import torch
import numpy as np

def aug_process_affine(Pseudo_label_batch, img_batch):
    tensor_images = []
    tensor_labels = []
    for i in range(Pseudo_label_batch.shape[0]):  # loop over the batch
        # 随机degrees值
        degrees = np.random.randint(-90, 90)
        # 随机scale因子
        scale = np.random.uniform(0.8, 1.2)
        # 随机平移量
        # print(Pseudo_label_batch.shape)
        tx = np.random.uniform(-0.2, 0.2) * Pseudo_label_batch.shape[2]
        ty = np.random.uniform(-0.2, 0.2) * Pseudo_label_batch.shape[1]
        # 随机倾斜因子
        shear = np.random.uniform(-10, 10)

        # convert tensor to numpy
        img_np = img_batch[i].cpu().permute(1, 2, 0).numpy()
        Pseudo_label_np = Pseudo_label_batch[i].cpu().numpy()

        # get image height and width
        h, w = img_np.shape[:2]

        # calculate the transformation matrix
        M = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, scale)

        # add translation to the transformation matrix
        M[:, 2] += (tx, ty)

        # add shear to the transformation matrix
        M[0, 1] = shear / h
        M[1, 0] = shear / w

        rotated_image_np = cv2.warpAffine(img_np, M, (w, h))
        rotated_label_np = cv2.warpAffine(Pseudo_label_np, M, (w, h), flags=cv2.INTER_NEAREST)

        # convert back to tensor
        tensor_image = torch.from_numpy(rotated_image_np).permute(2, 0, 1)  # HWC to CHW
        tensor_label = torch.from_numpy(rotated_label_np)
     
        tensor_images.append(tensor_image)
        tensor_labels.append(tensor_label)

    tensor_images = torch.stack(tensor_images).to(Pseudo_label_batch.device)  # concatenate all tensor_images into a batch
    tensor_labels = torch.stack(tensor_labels).to(Pseudo_label_batch.device)  # concatenate all tensor_labels into a batch

    return tensor_images, tensor_labels



    



    