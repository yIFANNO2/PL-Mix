import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse

categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, printlog=False):
    sample_miou = multiprocessing.Manager().dict()

    def compare(start, step, sample_miou):
        for idx in range(start, len(name_list), step):
            name = name_list[idx]
            predict_file = os.path.join(predict_folder, '%s.png' % name)
            predict = np.array(Image.open(predict_file))

            gt_file = os.path.join(gt_folder, '%s.png' % name)
            gt = np.array(Image.open(gt_file))
            cal = gt < 255
            mask = (predict == gt) * cal

            per_sample_tp = np.zeros(num_cls)
            per_sample_p = np.zeros(num_cls)
            per_sample_t = np.zeros(num_cls)

            for i in range(num_cls):
                per_sample_tp[i] = np.sum((gt == i) * mask)
                per_sample_p[i] = np.sum((predict == i) * cal)
                per_sample_t[i] = np.sum((gt == i) * cal)
            
            per_sample_iou = per_sample_tp / (per_sample_t + per_sample_p - per_sample_tp + 1e-10)
            sample_miou[name] = np.mean(per_sample_iou)

    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i, 8, sample_miou))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    
    return dict(sample_miou)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default='/mnt/petrelfs/jinglinglin/Yifan/MCTformer/voc12/train_id.txt', type=str)
    parser.add_argument("--predict_dir_mct", default='/mnt/petrelfs/jinglinglin/Yifan/SEAM/SEAM_ori_npy-segpng', type=str)
    parser.add_argument("--predict_dir_our", default='/mnt/petrelfs/jinglinglin/Yifan/MCTformer/SEAM-plmix-segpng', type=str)
    parser.add_argument("--gt_dir", default='/mnt/petrelfs/jinglinglin/Yifan/CAMMIX/data/VOCdevkit/VOC2012/SegmentationClass', type=str)
    parser.add_argument('--logfile', default='/mnt/petrelfs/jinglinglin/Yifan/SEAM/evallog.txt', type=str)
    parser.add_argument('--comment', default='MCTori')
    args = parser.parse_args()

    df = pd.read_csv(args.list, names=['filename'])
    name_list = df['filename'].values

    sample_miou_mct = do_python_eval(args.predict_dir_mct, args.gt_dir, name_list)
    sample_miou_our = do_python_eval(args.predict_dir_our, args.gt_dir, name_list)

    # 确保样本顺序一致
    mIoU_MCT = np.array([sample_miou_mct[name] for name in name_list])
    mIoU_our = np.array([sample_miou_our[name] for name in name_list])

    # 配对t检验
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(mIoU_our, mIoU_MCT)

    print(f't-statistic: {t_stat}, p-value: {p_value}')
