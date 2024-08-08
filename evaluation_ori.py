import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse
import cv2
import imutils
import torch.nn.functional as F
import torch

categories = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
              'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

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

def show_mask_on_image(img, mask, save_path):
    # Convert the mask to a colorful image
    colored_mask = np.array(mask_to_colorful(mask))

    # Convert the original image to float
    img_float = np.float32(img) / 255.0

    # Convert the colored mask to float
    colored_mask_float = np.float32(colored_mask) / 255.0

    # Blend the original image with the colored mask
    overlay = cv2.addWeighted(img_float, 0.5, colored_mask_float, 0.5, 0)

    # Save the result
    cv2.imwrite(save_path, np.uint8(overlay * 255))


def do_python_eval(predict_folder, gt_folder, img_dir, save_folder, name_list, num_cls=21, input_type='png', threshold=1.0, printlog=False,pngsave = False):
    TP = []
    P = []
    T = []

    npy_save_folder = os.path.join(save_folder + "-npy")
    os.makedirs(npy_save_folder, exist_ok=True)

    # 创建这个新的文件夹，如果它已经存在则不会抛出错误


    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
        

    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))

    def compare(start,step,TP,P,T,input_type,threshold):
        

        for idx in range(start,len(name_list),step):
            name = name_list[idx]
            orig_image = cv2.imread(os.path.join(img_dir, name + '.jpg'))
            # orig_image = np.array(Image.open(os.path.join(img_dir, name + '.jpg')).convert("RGB"))
            if input_type == 'png':
                predict_file = os.path.join(predict_folder,'%s.png'%name)
                predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)
                if num_cls == 81:
                    predict = predict - 91
            elif input_type == 'npy':
                predict_file = os.path.join(predict_folder,'%s.npy'%name)
                predict_dict = np.load(predict_file, allow_pickle=True).item()

                h, w = list(predict_dict.values())[0].shape
                tensor = np.zeros((num_cls,h,w),np.float32)


         


                for key in predict_dict.keys():
                    tensor[key+1] = predict_dict[key]
                tensor[0,:,:] = threshold 
                cam = F.softmax(torch.tensor(tensor).float(), dim=0).numpy()
                   
                predict = np.argmax(cam, axis=0).astype(np.uint8)
                # print('predict shape,',predict.shape)

                
                if pngsave == True:
                    # pred = imutils.crf_inference_label(orig_image, predict, n_labels=21)
                    # predict= np.argmax(pred, axis=0).astype(np.uint8)


                    npy_file_path = os.path.join(npy_save_folder, f"{name}.npy")

                    # 保存 pred 数组到 npy 文件
                    # np.save(npy_file_path, pred)
                
                    cam_img_path = os.path.join(save_folder, '%s.png' % name)
                    # show_mask_on_image(orig_image, predict, cam_img_path)
                    # 保存预测结果为 PNG 图像
                    cv2.imwrite(cam_img_path, predict)



            gt_file = os.path.join(gt_folder,'%s.png'%name)
            
            gt = np.array(Image.open(gt_file))

             # 将 predict 的大小调整为与 gt 相同
            predict = cv2.resize(predict, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
       
            # print('gt shape:',gt.shape)
            cal = gt<255
            mask = (predict==gt) * cal
      
            for i in range(num_cls):
                P[i].acquire()
                P[i].value += np.sum((predict==i)*cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt==i)*cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt==i)*mask)
                TP[i].release()
    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i,8,TP,P,T,input_type,threshold))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = [] 
    for i in range(num_cls):
        IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        T_TP.append(T[i].value/(TP[i].value+1e-10))
        P_TP.append(P[i].value/(TP[i].value+1e-10))
        FP_ALL.append((P[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
    loglist = {}
    for i in range(num_cls):
        loglist[categories[i]] = IoU[i] * 100
               
    miou = np.mean(np.array(IoU))
    loglist['mIoU'] = miou * 100
    fp = np.mean(np.array(FP_ALL))
    loglist['FP'] = fp * 100
    fn = np.mean(np.array(FN_ALL))
    loglist['FN'] = fn * 100
    if printlog:
        for i in range(num_cls):
            if i%2 != 1:
                print('%11s:%7.3f%%'%(categories[i],IoU[i]*100),end='\t')
            else:
                print('%11s:%7.3f%%'%(categories[i],IoU[i]*100))
        print('\n======================================================')
        print('%11s:%7.3f%%'%('mIoU',miou*100))
        print('\n')
        print(f'FP = {fp*100}, FN = {fn*100}')
    return loglist

def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  '%(key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)

def writelog(filepath, metric, comment):
    filepath = filepath
    logfile = open(filepath,'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n'%comment)
    writedict(logfile, metric)
    logfile.write('=====================================\n')
    logfile.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default='/mnt/petrelfs/jinglinglin/Yifan/MCTformer/voc12/train_id.txt', type=str)
    parser.add_argument("--predict_dir", default='/mnt/petrelfs/jinglinglin/Yifan/SEAM/SEAM_ori_npy', type=str)
    parser.add_argument("--gt_dir", default='/mnt/petrelfs/jinglinglin/Yifan/CAMMIX/data/VOCdevkit/VOC2012/SegmentationClass', type=str)#SegmentationClass
    parser.add_argument("--save_folder", default='/mnt/petrelfs/jinglinglin/Yifan/SEAM/SEAM_ori_npy-segpng', type=str)
    parser.add_argument("--img_dir", default='/mnt/petrelfs/jinglinglin/Yifan/CAMMIX/data/VOCdevkit/VOC2012/JPEGImages', type=str)
   
    parser.add_argument('--logfile', default='evallog.txt',type=str)
    parser.add_argument('--comment', default="train1464", type=str) #required=True,
    parser.add_argument('--type', default='npy', choices=['npy', 'png'], type=str)
    parser.add_argument('--t', default= 0.210, type=float)
    parser.add_argument('--curve', default=False, type=bool)
    parser.add_argument('--num_classes', default=21, type=int)
    parser.add_argument('--start', default=20, type=int)
    parser.add_argument('--end', default=70, type=int)
    args = parser.parse_args()

    if args.type == 'npy':
        assert args.t is not None or args.curve
    df = pd.read_csv(args.list, names=['filename'])
    name_list = df['filename'].values
    if not args.curve:
        loglist = do_python_eval(args.predict_dir, args.gt_dir, args.img_dir, args.save_folder, name_list, args.num_classes, args.type, args.t, printlog=True, pngsave =True) #True
        writelog(args.logfile, loglist, args.comment)
    else:
        l = []
        max_mIoU = 0.0
        best_thr = 0.0
        for i in range(args.start, args.end):
            t = i/100.0
            loglist = do_python_eval(args.predict_dir, args.gt_dir,args.img_dir, args.save_folder, name_list, args.num_classes, args.type, t, pngsave =False)
            l.append(loglist['mIoU'])
            print('%d/60 background score: %.3f\tmIoU: %.3f%%'%(i, t, loglist['mIoU']))
            if loglist['mIoU'] > max_mIoU:
                max_mIoU = loglist['mIoU']
                best_thr = t
            else:
                break
        print('Best background score: %.3f\tmIoU: %.3f%%' % (best_thr, max_mIoU))
        writelog(args.logfile, {'mIoU':l, 'Best mIoU': max_mIoU, 'Best threshold': best_thr}, args.comment)

