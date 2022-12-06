import os
import shutil

# python train.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 4 --epochs 8370 --ckpt ~/mnt/Data/KITTI/openpcdet_pretrained/pv_rcnn_8369.pth --save_to_file --output_dir ""

# python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 16 --ckpt /home/jin/mnt/github/OpenPCDet/output/kitti_models/pv_rcnn/default/ckpt/latest_model.pth

trainer = 'python /home/jin/mnt/github/OpenPCDet/tools/my_train.py'

widths    = [2048, 1024, 512, 256]
intr_ints = [1.25, 1.5, 1.75]

output_root = '/home/jin/mnt/Data/KITTI/trained_models/cv_int/'
# output_dirs = [output_root+'4500', output_root+'1024', output_root+'512', output_root+'256']
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
cfgs = ['parta2',
        # 'parta2_free',
        # 'pointpillar',
        # 'pointrcnn',
        # 'pointrcnn_iou',
        # 'pv_rcnn',
        # 'second',
        # 'second_iou',
        # 'voxel_rcnn_car'
        ]

data_root = '/home/jin/mnt/Data/KITTI/cv_inter/'
cv_algs = ['area', 'cubic', 'linear', 'lz4', 'nearest']

for cv_alg in cv_algs:
    cv_data_root = data_root + cv_alg
    for width in widths:
        for intr_int in intr_ints:
            intr_width = int(width * intr_int)
            training_data_path = cv_data_root + '/' + str(intr_width) + '/training'
            testing_data_path  = cv_data_root + '/' + str(intr_width) + '/testing'
            training_link_path = '/home/jin/mnt/github/OpenPCDet/data/kitti/training/velodyne'
            testing_link_path = '/home/jin/mnt/github/OpenPCDet/data/kitti/testing/velodyne'

            os.system("rm /home/jin/mnt/github/OpenPCDet/data/kitti/training/velodyne")
            os.system("rm /home/jin/mnt/github/OpenPCDet/data/kitti/testing/velodyne")

            command = f"ln -s {training_data_path} {training_link_path}"
            os.system(command)

            command = f"ln -s {testing_data_path} {testing_link_path}"
            os.system(command)

            for cfg, cfg_file in zip(cfgs, cfg_files):
                output_dir = output_root + cv_alg + '/' + str(intr_width)+ '/' + cfg
                command = f"{trainer} --cfg_file {cfg_file} --batch_size 4 --epochs 5 --save_to_file --output_dir {output_dir}"
                # print(command)
                os.system(command)
            #    python train.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 4 --epochs 5 --save_to_file --output_dir ""
