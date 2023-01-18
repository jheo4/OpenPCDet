import os
import shutil

# python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 16 --ckpt /home/jin/mnt/github/OpenPCDet/output/kitti_models/pv_rcnn/default/ckpt/latest_model.pth

trainer = 'python /home/jin/mnt/github/OpenPCDet/tools/my_train.py'

widths    = [2048, 1024, 512, 256]
intr_ints = [1.25, 1.5, 1.75, 2]

output_root = '/home/jin/mnt/Data/KITTI/retrained_models/'

models = ['parta2',
          'pointpillar',
          'pv_rcnn',
          'second',
        ]

add_epoch = 5
model_epochs = [
        80+add_epoch,
        80+add_epoch,
        77+add_epoch,
        80+add_epoch,
        ]

model_ckpts = [
         '/home/jin/mnt/Data/KITTI/pretrained_models/original/PartA2_7940.pth',        # 80
         '/home/jin/mnt/Data/KITTI/pretrained_models/original/pointpillar_7728.pth',   # 80
         '/home/jin/mnt/Data/KITTI/pretrained_models/original/pv_rcnn_8369.pth',       # 77
         '/home/jin/mnt/Data/KITTI/pretrained_models/original/second_7862.pth',        # 80
        # '/home/jin/mnt/Data/KITTI/pretrained_models/original/PartA2_free_7872.pth',   # 72
        # '/home/jin/mnt/Data/KITTI/pretrained_models/original/pointrcnn_iou_7875.pth', # 78
        # '/home/jin/mnt/Data/KITTI/pretrained_models/original/second_iou7909.pth',     # 7909
        # '/home/jin/mnt/Data/KITTI/pretrained_models/original/voxel_rcnn_car_84.54.pth', # 54
        ]

cfg_files = [
         '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/PartA2.yaml',
         '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml',
         '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml',
         '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/second.yaml',
        ]

data_root = '/home/jin/mnt/Data/KITTI/interpolation0.5/'

for width in widths:
    for intr_int in intr_ints:
        intr_width = int(width * intr_int)

        training_data_path = data_root + str(intr_width) + '/training'
        training_link_path = '/home/jin/mnt/github/OpenPCDet/data/kitti/training/velodyne'

        command = f"rm {training_link_path}"
        os.system(command)
        print(command)

        command = f"ln -s {training_data_path} {training_link_path}"
        os.system(command)
        print(command)

        for model, model_epoch, model_ckpt, cfg_file in zip(models, model_epochs, model_ckpts, cfg_files):
            output_dir = output_root + str(intr_width)+ '/' + model
            command = f"{trainer} --cfg_file {cfg_file} --batch_size 4 --epochs {model_epoch} --ckpt {model_ckpt} --save_to_file --output_dir {output_dir}"
            os.system(command)
            print(command)

