import os
import shutil

# python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 16 --ckpt /home/jin/mnt/github/OpenPCDet/output/kitti_models/pv_rcnn/default/ckpt/latest_model.pth

tester = 'python /home/jin/mnt/github/OpenPCDet/tools/test.py'

widths    = [4500, 2048, 1024, 512, 256]

output_root = '/home/jin/mnt/Data/KITTI/results/preT_orig4500_orig/'

models = ['parta2',
        # 'parta2_free',
        # 'pointpillar',
        # 'pointrcnn',
        # 'pointrcnn_iou',
        # 'pv_rcnn',
        # 'second',
        # 'second_iou',
        # 'voxel_rcnn_car'
        ]

model_ckpts = [
        '/home/jin/mnt/Data/KITTI/pretrained_models/original/PartA2_7940.pth',
        ]

cfg_files = [
        '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/PartA2.yaml',
        # '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/PartA2_free.yaml',
        # '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml',
        # '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/pointrcnn.yaml',
        # '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/pointrcnn_iou.yaml',
        # '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml',
        # '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/second.yaml',
        # '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/second_iou.yaml',
        # '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/voxel_rcnn_car.yaml'
        ]

data_root = '/home/jin/mnt/Data/KITTI/original/'

for width in widths:
    training_data_path = data_root + str(width) + '/training'
    training_link_path = '/home/jin/mnt/github/OpenPCDet/data/kitti/training/velodyne'
    print(training_data_path)

    os.system("rm /home/jin/mnt/github/OpenPCDet/data/kitti/training/velodyne")

    command = f"ln -s {training_data_path} {training_link_path}"
    os.system(command)

    for model, model_ckpt, cfg_file in zip(models, model_ckpts, cfg_files):
        output_dir = output_root + str(width)+ '/' + model
        command = f"{tester} --cfg_file {cfg_file} --batch_size 24 --ckpt {model_ckpt} --save_to_file --output_dir {output_dir}"
        print(command)
        os.system(command)
    #    python train.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 4 --epochs 5 --save_to_file --output_dir ""
