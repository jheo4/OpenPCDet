import os
import shutil

# python train.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 4 --epochs 8370 --ckpt ~/mnt/Data/KITTI/openpcdet_pretrained/pv_rcnn_8369.pth --save_to_file --output_dir ""

# python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 16 --ckpt /home/jin/mnt/github/OpenPCDet/output/kitti_models/pv_rcnn/default/ckpt/latest_model.pth

trainer = 'python /home/jin/mnt/github/OpenPCDet/tools/my_train.py'

data_root = '/home/jin/mnt/Data/KITTI/original/'
# widths  = [2048, 1024, 512, 256]
widths  = [4500, 1024, 512, 256]

output_root = '/home/jin/mnt/Data/KITTI/trained_models/orig/'
# output_dirs = [output_root+'2048', output_root+'1024', output_root+'512', output_root+'256']
output_dirs = [output_root+'4500', output_root+'1024', output_root+'512', output_root+'256']
cfg_files = [
        '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/PartA2.yaml',
        '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/PartA2_free.yaml',
        '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml',
        '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/pointrcnn.yaml',
        '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/pointrcnn_iou.yaml',
        '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml',
        '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/second.yaml',
        '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/second_iou.yaml',
        '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/voxel_rcnn_car.yaml'
        ]
cfgs = ['parta2', 'parta2_free', 'pointpillar', 'pointrcnn', 'pointrcnn_iou', 'pv_rcnn', 'second', 'second_iou', 'voxel_rcnn_car']

for width, output_dir in zip(widths, output_dirs):
    training_data_path = data_root + str(width) + '/training'
    testing_data_path = data_root + str(width) + '/testing'
    training_link_path = '/home/jin/mnt/github/OpenPCDet/data/kitti/training/velodyne'
    testing_link_path = '/home/jin/mnt/github/OpenPCDet/data/kitti/testing/velodyne'

    os.system("rm /home/jin/mnt/github/OpenPCDet/data/kitti/training/velodyne")
    os.system("rm /home/jin/mnt/github/OpenPCDet/data/kitti/testing/velodyne")

    command = f"ln -s {training_data_path} {training_link_path}"
    os.system(command)

    command = f"ln -s {testing_data_path} {testing_link_path}"
    os.system(command)

    # not need...
    # os.system("python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos ../cfgs/dataset_configs/kitti_dataset.yaml")

    for cfg, cfg_file in zip(cfgs, cfg_files):
        command = f"{trainer} --cfg_file {cfg_file} --batch_size 4 --epochs 5 --save_to_file --output_dir {output_dir+'/'+cfg}"
        print(command)
        os.system(command)
    #    python train.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 4 --epochs 5 --save_to_file --output_dir ""
