import os

import torch
import numpy as np

import random
import cv2
import time

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from data import *
import argparse

# from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from loss_multilabel import *
from network.EffSeg_V2 import *
import imutils,  torchutils
from PIL import Image
from tools import *
import pandas as pd
from visdom import Visdom

from tqdm import tqdm
from imutils import color_map,color_denorm

import logging





label_dict = {0:'Background', 1:'Aeroplane', 2:'Bicycle', 3:'Bird', 4:'Boat', 5:'Bottle', 6:'Bus',
              7:'Car', 8:'Cat', 9:'Chair', 10:'Cow', 11:'Diningtable',
              12:'Dog', 13:'Horse', 14:'Motorbike', 15:'Person', 16:'Pottedplant',
              17:'Sheep', 18:'Sofa', 19:'Train', 20:'TVmonitor'}

def cam_maxnorm(cams):
    cams = torch.relu(cams)
    n,c,h,w = cams.shape
    cam_min = torch.min(cams.view(n,c,-1), dim=-1)[0].view(n,c,1,1)
    cam_max = torch.max(cams.view(n,c,-1), dim=-1)[0].view(n,c,1,1)
    norm_cam = (cams - cam_min - 1e-6)/ (cam_max - cam_min + 1e-6)
    norm_cam = torch.relu(norm_cam)
    return norm_cam

def cam_softmaxnorm(cams):
    n,c,h,w = cams.shape
    foreground = torch.softmax(cams[:,1:,:,:], dim=1)
    background = (1-torch.max(foreground, dim=1)[0]).unsqueeze(1)
    norm_cam = torch.cat([background, foreground], dim=1)
    return norm_cam


from PIL import Image

def mask_to_colorful(mask):
    arr = mask.astype(np.uint8)
    im = Image.fromarray(arr)

    palette = []
    for i in range(256):
        palette.extend((i, i, i))
    palette[:3*21] = np.array([[0, 0, 0],
                               [128, 0, 0],
                               [0, 128, 0],
                               [128, 128, 0],
                               [0, 0, 128],
                               [128, 0, 128],
                               [0, 128, 128],
                               [128, 128, 128],
                               [64, 0, 0],
                               [192, 0, 0],
                               [64, 128, 0],
                               [192, 128, 0],
                               [64, 0, 128],
                               [192, 0, 128],
                               [64, 128, 128],
                               [192, 128, 128],
                               [0, 64, 0],
                               [128, 64, 0],
                               [0, 192, 0],
                               [128, 192, 0],
                               [0, 64, 128]], dtype='uint8').flatten()

    im.putpalette(palette)
    return im.convert("RGB") 

import cv2
import numpy as np

def show_mask_on_image(img, mask, save_path):
   
    colored_mask = np.array(mask_to_colorful(mask))

    img_float = np.float32(img) / 255.0

    colored_mask_float = np.float32(colored_mask) / 255.0

    overlay = cv2.addWeighted(img_float, 0.5, colored_mask_float, 0.5, 0)

    cv2.imwrite(save_path, np.uint8(overlay * 255))






def get_sample_weight(dataset):
    t1 = time.time()
    sample_weight = [0]*len(dataset)
    class_count = [590, 504, 705, 468, 714, 393, 1150, 1005, 1228, 267,
               613, 1188, 445, 492, 4155, 522, 300, 649, 503, 567]
    sum_instance = len(dataset)  #adjustable
    for i, (name, img, label) in enumerate(dataset):
        multihots = torch.where(label==1)[0]
        instance_count = 0
        for hot in multihots:
            instance_count += class_count[hot.item()]
        sample_weight[i] = sum_instance/instance_count
    print(f'calculate sample weight takes:{time.time()-t1}seconds')
    return sample_weight



   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation_name", default="Your project name", type=str)

    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--max_epoches", default=30, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=1e-6, type=float) 
    parser.add_argument("--train_list", default="data/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt", type=str)
    parser.add_argument("--num_classes", default=21, type=int)

    base_path = "Your project path"
    parser.add_argument("--session_name", default=os.path.join(base_path, '{ablation_name}/EffiB3_PLMix'), type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--weights", default=None, type=str)
    parser.add_argument("--voc12_root", default=os.path.join(base_path, 'data/VOCdevkit/VOC2012'), type=str)
    parser.add_argument("--tblog_dir", default=os.path.join(base_path, '{ablation_name}/tblog_new'), type=str)
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--Random_mix", default=True, type=bool)
    parser.add_argument("--Random_order", default=True, type=bool)
    parser.add_argument("--mix_cls_loss", default=False, type=bool)
    parser.add_argument("--crf", default=True, type=bool)
    parser.add_argument("--mix_loss", default='Focal_loss', type=str, choices=['Focal_loss', 'CE_loss', 'Focal_CE_loss'])
    parser.add_argument("--test_path", default=os.path.join(base_path, '{ablation_name}_npy'), type=str)
    parser.add_argument("--test_path_png", default=os.path.join(base_path, '{ablation_name}_png'), type=str)
    parser.add_argument("--log_name",default=os.path.join(base_path, '{ablation_name}_log'), type=str)
    parser.add_argument("--camT",default=0.30, type=float)
    parser.add_argument("--MaxmIou",default=50, type=float)
   

    args = parser.parse_args()

    # Replace placeholders with ablation_name
    args.session_name = args.session_name.format(ablation_name=args.ablation_name)
    args.tblog_dir = args.tblog_dir.format(ablation_name=args.ablation_name)
    args.test_path = args.test_path.format(ablation_name=args.ablation_name)
    args.test_path_png = args.test_path_png.format(ablation_name=args.ablation_name)

   

    save_path = args.session_name.rsplit('/', 1)[0]

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    


    print(vars(args))
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       

    model = EfficientSeg(num_classes=args.num_classes, pretrained='efficientnet-b3', layers=3, MemoryEfficient=True, last_pooling=False)
  
    model = model.to(device)
    for module in model.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.affine = False
            
    os.makedirs(args.tblog_dir, exist_ok=True)
    logger = logging.getLogger(args.log_name)

  
    logger.setLevel(logging.DEBUG)

   
    tblog_dir_path=args.tblog_dir+'/training_log.log'  
    fh = logging.FileHandler(tblog_dir_path)

    ch = logging.StreamHandler()


    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

 
    logger.addHandler(fh)
    logger.addHandler(ch)




    train_dataset = VOC12ClsPix(args.train_list, voc12_root=args.voc12_root,
                                               transform=transforms.Compose([
                        imutils.RandomResizeLong(448, 768),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                        np.asarray,
                        imutils.color_norm,
                        imutils.RandomCrop(args.crop_size),
                        imutils.HWC_to_CHW,
                        torch.from_numpy,
                    ]), view_size=(224,224))

    
    eval_dataset = VOC12ClsDatasetMSF(args.val_list, voc12_root=args.voc12_root,
                                                  scales=[1],
                                                  inter_transform=torchvision.transforms.Compose(
                                                       [np.asarray,
                                                        imutils.color_norm,
                                                        imutils.HWC_to_CHW]))

    eval_data_loader = DataLoader(eval_dataset, shuffle=False, num_workers=args.num_workers)


    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                   worker_init_fn=worker_init_fn, shuffle=True)
    max_step = len(train_dataset) // args.batch_size * args.max_epoches






    optimizer = optim.Adam(params=model.parameters(), lr=args.lr ,weight_decay=args.wt_dec)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=0, cooldown=0, factor=0.5, min_lr=1e-5) 
    epsave=0
    if args.weights:
        
        if args.weights[-5:] == '.ckpt':
            model_dict = torch.load(args.weights)
            model.load_state_dict(model_dict['model_state_dict'])
            optimizer.load_state_dict(model_dict['optimizer_state_dict'])
            epsave = model_dict['iteration']+1
            
            
        else:
            weights_dict = torch.load(args.weights)
            model.load_state_dict(weights_dict, strict=False)
    print('Resuming training from epoch %d' % epsave)

    model = model.cuda()
    torch.autograd.set_detect_anomaly(True)
    criterion1 = FocalLoss()
    criterion2 = Log_Sum_Exp_Pairwise_Loss
    criterion3 = nn.MultiLabelSoftMarginLoss()
    criterion4 = EMD() 
    criterion5 = image_level_contrast
   
    if args.mix_loss == 'CE_loss':
        criterion6 = nn.CrossEntropyLoss()
    elif args.mix_loss == 'Focal_loss':
        criterion6 = FocalLoss_DEEPLAB()
    elif args.mix_loss == 'Focal_CE_loss':
        criterion6 = FocalCELoss()

    criterion7 = SoftIoULoss()
    criterion8 = SoftDiceLoss()
    criterion9 = SpatialContinuityLoss()
        


    
    
  

    valid_cam = 0
    mIoU_result = [ ]

    for ep in range(epsave, args.max_epoches):
        
        
        for iter, pack in enumerate(train_data_loader):
            optimizer.zero_grad()
           
            model.train()
            name, img, label, view1, view2, coord1, coord2, ori_coord = pack
            
    

            _, C, H, W = img.shape

            if torch.cuda.is_available():
                
                label = label.to(device).float()
                img = img.to(device).float()
                view1 = view1.to(device).float()
                view2 = view2.to(device).float()

            with torch.no_grad():
                label_with_bg = label.clone()
                bg_score = torch.ones((label.shape[0], 1), device=label_with_bg.device)
                label_with_bg = torch.cat((bg_score, label_with_bg), dim=1)

 
            raw_cams, raw_sgcs, emb, logits = model(img, cam='cam')

            cams = cam_softmaxnorm(raw_cams).detach()
            sgcs = cam_softmaxnorm(raw_sgcs)
            SAM_MAP=sgcs

            valid_channel = int(label.sum().cpu().data)
            
            loss_focal = criterion1(torch.sigmoid(logits[:, 1:]), label) 
            loss_softmargin = criterion3(logits[:, 1:], label) 
            loss_pair = criterion2(torch.sigmoid(logits[:, 1:]), label).mean() 
            loss_cls = loss_pair + loss_softmargin + loss_focal

            cams = cams*label_with_bg.unsqueeze(2).unsqueeze(3)
            sgcs = sgcs*label_with_bg.unsqueeze(2).unsqueeze(3)
            n,c,h,w = cams.shape
            loss_er = torch.topk(torch.flatten(torch.abs(cams.detach()-sgcs), start_dim=1), k=int(0.2*valid_channel*h*w), dim=-1)[0].mean()
            loss = loss_cls + loss_er

            loss_imc = 0.0
            loss_SC1 = 0.0
            loss_SC2 = 0.0
            
            if ep >= 4: 
                loss_imc = criterion5(emb, label)
                if torch.is_tensor(loss_imc):
                    loss = loss_cls + loss_imc + loss_er
                
            
            
            loss_mix = 0.0
            loss_mixcls = 0.0
            pos_loss_mixcls = 0.0
            neg_loss_mixcls = 0.0
            mix_loss_cls = 0.0
            keshihua = 0
            loSS_IoU = 0
            loSS_Dice = 0
            loss_SC = 0
            mix_loss_softmargin = 0

           
            
            if ep >=6: 

                model.eval()
                with torch.no_grad():
                    label_gt = label_with_bg.clone() #[N, 21]
                    if args.Random_order == False:
                    
                        new_img_list = Switching_imglist_order(img) #
                        new_labels = Switching_label_order(label_gt) #
                    else:
                        #随机打乱
                        Random_order = torch.randperm(img.shape[0])
                        new_img_list = img[Random_order]
                        new_labels = label_gt[Random_order]
                        Random_order_list = Random_order.tolist()
                        new_name = [name[i] for i in Random_order_list]

                    mixlabel = torch.max(label_gt[:, 1:], new_labels[:, 1:])
                   

                    _, SAM_ori, _, _ = model(new_img_list, cam='cam')

                    new_img_list_flipped = torch.flip(new_img_list, [3])  
                    _, SAM_flipped, _, _ = model(new_img_list_flipped, cam='cam')

                   
                    SAM_flipped = torch.flip(SAM_flipped, [3])
                    SAM_new = (SAM_ori + SAM_flipped) / 2.0
                                      
                    if args.crf == True:
                        img_denorm = color_denorm(img)
                        new_img_list_denorm = color_denorm(new_img_list)
                        
                        Pseudo_label_ori = get_crf_label(img_denorm,SAM_MAP,label_gt,T=camT)
                        Pseudo_label_new = get_crf_label(new_img_list_denorm, SAM_new,new_labels,T=camT)
                    
                                        
                    else:
                        
                        Pseudo_label_ori = get_Pseudo_label(SAM_MAP,label_gt,T=camT)
                        Pseudo_label_new = get_Pseudo_label(SAM_new,new_labels,T=camT)
                            
                    if ep >= 8:
                       
                        img_aug, Pseudo_label_ori=aug_process_rotation(Pseudo_label_ori,img)
                       
                        Pseudo_label_ori,img=Random_mix_process(Pseudo_label_ori,img_aug)  
                        
                    Pseudo_label_ori=Pseudo_label_ori.cuda()
                    img=img.cuda()
                    Pseudo_label_new=Pseudo_label_new.cuda()
                    new_img_list=new_img_list.cuda()

                     
                    mix_label = get_mix_label(Pseudo_label_ori,Pseudo_label_new)
                    mix_img = get_mix_image(img,new_img_list,Pseudo_label_ori,label_gt,thres=camT)

                
                mixSAM, _, _,_ = model(mix_img, cam='cam')

                

                mixSAM=cam_softmaxnorm(mixSAM)
                mix_label = mix_label.to(device).type(torch.long)

                
                loss_mixcls = criterion6(mixSAM, mix_label)
                loSS_IoU = criterion7(mixSAM, mix_label)
                loSS_Dice = criterion8(mixSAM, mix_label)  
                loss_SC = criterion9 (mixSAM,mixlabel)
                
                
               
                
                loss = loss + loss_mixcls+loSS_Dice+loSS_IoU + loss_SC
                del mixSAM
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            loss_pixpro = 0.0
            if ep >= 9:
                cams_max, cams_mean = cams.max(), cams.mean()
                sgcs_max, sgcs_mean = sgcs.max().detach(), sgcs.mean().detach()
                
                model.eval()
                
                _, sgcs_vw1 = model(view1, cam='pix')
                
                with torch.no_grad():
                    cams_vw2, _, = model(view2, cam='pix')

                loss_pixpro = PixPro(cam_maxnorm(sgcs_vw1)*label_with_bg.unsqueeze(2).unsqueeze(3), cam_maxnorm(cams_vw2)*label_with_bg.unsqueeze(2).unsqueeze(3), coord1, coord2)

            loss = loss_pixpro


            loss_emd = 0.0
            if ep >= 12: 
                vw1 = cam_softmaxnorm(sgcs_vw1)
                vw2 = cam_softmaxnorm(cams_vw2)

                vw1 = F.normalize(vw1, dim=1)
                vw2 = F.normalize(vw2, dim=1)
                crops_vw1, crops_vw2, batch_indices = torchutils.get_dynamic_crops(vw1, coord1, vw2.detach(), coord2)
                loss_emd = criterion4(crops_vw1, crops_vw2, mode='dynamic') 
                del crops_vw1
                del crops_vw2
                loss += loss_emd

            if torch.is_tensor(loss):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            del cams
            del sgcs


            if iter % 500 == 0:
               

                cams = cam_maxnorm(raw_cams)[0] #C,H,W
                sgcs = cam_maxnorm(raw_sgcs.detach())[0]
                norm_sgc = sgcs.cpu().data.numpy().transpose(1,2,0) # H,W,C
                norm_cam = cams.cpu().data.numpy().transpose(1,2,0) # H,W,

                logger.info('Iter:%5d/%5d loss_focal:%.4f loss_softmargin:%.4f loss_pair:%.4f '
                            'loss_er:%.4f loss_imc:%.4f loss_pixc:%.4f loss_emd:%.4f '
                            'loss_mix:%.4f loss_mixcls:%.4f pos_loss_mixcls %.4f '
                            'loSS_IoU:%.4f loSS_Dice:%.4f mix_loss_softmargin:%.4f loss_SC:%.4f'
                            'neg_loss_mixcls %.4f lr: %.7f'% 
                            (iter + max_step//args.max_epoches*ep, max_step, loss_focal, 
                                loss_softmargin, loss_pair, loss_er, loss_imc, loss_pixpro, 
                                loss_emd, loss_mix, loss_mixcls, pos_loss_mixcls, 
                                loSS_IoU, loSS_Dice,mix_loss_softmargin, loss_SC,
                                neg_loss_mixcls, optimizer.param_groups[0]['lr']))



                img_8 = img[0].cpu().numpy().transpose((1,2,0))
                img_8 = np.ascontiguousarray(img_8)
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)
                img_8[:,:,0] = (img_8[:,:,0]*std[0] + mean[0])*255
                img_8[:,:,1] = (img_8[:,:,1]*std[1] + mean[1])*255
                img_8[:,:,2] = (img_8[:,:,2]*std[2] + mean[2])*255
                img_8[img_8 > 255] = 255
                img_8[img_8 < 0] = 0
                img_8 = img_8.astype(np.uint8)


              
                for i in range(1, norm_cam.shape[2]):
                    if label[0,i-1]>0: #if its interested classes
                        cam = norm_cam[:,:,i]
                        sgc = norm_sgc[:,:,i]
                        vis_cam = show_cam_on_image(img_8, cam)
                        vis_sgc = show_cam_on_image(img_8, sgc)


                        valid_cam += 1
        
       
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': ep
        }, os.path.join(args.session_name + 'train' + '.ckpt'))# 'model%d.ckpt' % epoch))
         


       ##eval
        model.eval()
        stamp = time.time()
        print('start test')

        
        
        os.makedirs(args.test_path, exist_ok=True)
        

        for iter, (img_name, img_list, label) in tqdm(enumerate(eval_data_loader)):
            img_name = img_name[0]; label = label[0].unsqueeze(0)

           
            img_path = get_img_path(img_name, args.voc12_root)
            orig_img = np.asarray(Image.open(img_path))
            H, W, _ = orig_img.shape
            raw_cam_list, SGC_list, score_list = [], [], []


            label_with_bg = label.clone()
            bg_score = torch.ones((label.shape[0], 1), device=label_with_bg.device)
            label_with_bg = torch.cat((bg_score, label_with_bg), dim=1)

            for i, img in enumerate(img_list):
                with torch.no_grad():
                    _, SGC, _, _ = model(img.cuda().float(), cam='cam')
                    SGC = SGC.squeeze(0).cpu().data.numpy() # C,H,W
                    SGC = SGC.transpose(1,2,0) #H,W,C
                    SGC = cv2.resize(SGC.astype(np.float32), (W, H))

                if i % 2 == 1:
                    SGC = np.flip(SGC, axis=1)

                SGC = SGC.transpose(2,0,1) #C,H,W
    
                SGC_list.append(SGC[1:])


            norm_SGC = np.sum(SGC_list, axis=0)
            norm_SGC[norm_SGC < 0] = 0
            SGC_max = np.max(norm_SGC, (1,2), keepdims=True)
            SGC_min = np.min(norm_SGC, (1,2), keepdims=True)
            norm_SGC[norm_SGC < SGC_min+1e-6] = 0
            norm_SGC = (norm_SGC-SGC_min-1e-6) / (SGC_max - SGC_min + 1e-6)

           
            sgc_dict = {}
        
            for i in range(20):
               
                if label[:,i] > 1e-5:
                    sgc_dict[i] = norm_SGC[i]

           
            np.save(os.path.join(args.test_path, img_name + '.npy'), sgc_dict)


        df = pd.read_csv(args.val_list, names=['filename'])
        name_list = df['filename'].values
        from evaluation import do_python_eval
        mious = []
        for t in range(20,52, 2):
            t /= 100.0
            loglist = do_python_eval(args.test_path, 'data/VOCdevkit/VOC2012/SegmentationClass', name_list, 21, 'npy', t, printlog=False)
            mious.append(loglist['mIoU'])
        max_miou = max(mious)
        max_t = mious.index(max_miou)*0.02 + 0.2
        print(f'\n Epoch:{ep} max miou:{max_miou} max t:{max_t}',
               f'Time elapse:{time.time()-stamp}s')
        scheduler.step(max_miou)

        camT=max_t
        mIoU_result.append(max_miou)
        
        logger.info('mIoU Result: %s', mIoU_result)

        

        test_path_png = args.test_path_png+str(ep)
        os.makedirs(test_path_png, exist_ok=True)

        cmap = color_map()[:, np.newaxis, :]
        for npy_file in os.listdir(args.test_path):
            if npy_file.endswith('.npy'):
                # load npy file
                cam_npy = np.load(os.path.join(args.test_path, npy_file),allow_pickle=True).item()
                h, w = list(cam_npy.values())[0].shape
                tensor = np.zeros((21,h,w),np.float32)
                
                for key in cam_npy.keys():
                    tensor[key+1] = cam_npy[key]
                tensor[0,:,:] = camT 
                predict = np.argmax(tensor, axis=0).astype(np.uint8)
               


                # get the corresponding original image
                img_name = npy_file.split('.')[0]
                img_path = get_img_path(img_name, args.voc12_root)
                orig_img = cv2.imread(img_path)
              

                # use the mask to generate a CAM image and save it
                cam_img_path = os.path.join(test_path_png, f'{img_name}_cam.png')
                show_mask_on_image(orig_img, predict, cam_img_path)

        
        if max_miou > MaxmIou:
            MaxmIou = max_miou
           
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': ep
            }, os.path.join(args.session_name  + '_best.ckpt'))# 'model%d.ckpt' % epoch))
         
            
            print('best model saved')

       


 
