import os
import shutil

# python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 16 --ckpt /home/jin/mnt/github/OpenPCDet/output/kitti_models/pv_rcnn/default/ckpt/latest_model.pth

tester = 'python /home/jin/mnt/github/OpenPCDet/tools/test.py'

widths    = [2048, 1024, 512, 256]
widths    = [1024, 512, 256]
intr_ints = [1.25, 1.5, 1.75, 2]

output_root = '/home/jin/mnt/Data/KITTI/results/preT_orig4500_cvinters/'

models = [# 'parta2',
          # 'pointpillar',
          'pointrcnn',
          # 'pv_rcnn',
          # 'second',
        ]

model_ckpts = [
        # '/home/jin/mnt/Data/KITTI/pretrained_models/original/PartA2_7940.pth',
        # '/home/jin/mnt/Data/KITTI/pretrained_models/original/pointpillar_7728.pth',
        '/home/jin/mnt/Data/KITTI/pretrained_models/original/pointrcnn_7870.pth',
        # '/home/jin/mnt/Data/KITTI/pretrained_models/original/pv_rcnn_8369.pth',
        # '/home/jin/mnt/Data/KITTI/pretrained_models/original/second_7862.pth',
        ]

cfg_files = [
        # '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/PartA2.yaml',
        # '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml',
        '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/pointrcnn.yaml',
        # '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml',
        # '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/second.yaml',
        ]

data_root = '/home/jin/mnt/Data/KITTI/cv_inter/'
cv_algs = ['area', 'cubic', 'linear', 'lz4', 'nearest']

for cv_alg in cv_algs:
    cv_data_root = data_root + cv_alg
    for width in widths:
        for intr_int in intr_ints:
            intr_width = int(width * intr_int)

            training_data_path = cv_data_root + '/' + str(intr_width) + '/training'
            training_link_path = '/home/jin/mnt/github/OpenPCDet/data/kitti/training/velodyne'

            os.system("rm /home/jin/mnt/github/OpenPCDet/data/kitti/training/velodyne")

            command = f"ln -s {training_data_path} {training_link_path}"
            os.system(command)

            for model, model_ckpt, cfg_file in zip(models, model_ckpts, cfg_files):
                output_dir = output_root + cv_alg + '/' + str(intr_width)+ '/' + model
                command = f"{tester} --cfg_file {cfg_file} --batch_size 4 --ckpt {model_ckpt} --save_to_file --output_dir {output_dir}"
                print(command)
                os.system(command)
            #    python train.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 4 --epochs 5 --save_to_file --output_dir ""
