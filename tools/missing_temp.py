import os
import shutil

# python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 16 --ckpt /home/jin/mnt/github/OpenPCDet/output/kitti_models/pv_rcnn/default/ckpt/latest_model.pth

trainer = 'python /home/jin/mnt/github/OpenPCDet/tools/my_train.py'

widths      = [2048, 1024, 512,  256]
intr_ints   = [2,    1.75, 1.5, 1.25]
batch_sizes = [4,       4,   4,    4]

output_root = '/home/jin/mnt/Data/KITTI/retrained_models/cv_intr/'

models = [
        'pv_rcnn',
        ]

add_epoch = 5
model_epochs = [
        77+add_epoch,
        ]

model_ckpts = [
         '/home/jin/mnt/Data/KITTI/pretrained_models/original/pv_rcnn_8369.pth',       # 77
        # '/home/jin/mnt/Data/KITTI/pretrained_models/original/PartA2_free_7872.pth',   # 72
        # '/home/jin/mnt/Data/KITTI/pretrained_models/original/pointrcnn_iou_7875.pth', # 78
        # '/home/jin/mnt/Data/KITTI/pretrained_models/original/second_iou7909.pth',     # 7909
        # '/home/jin/mnt/Data/KITTI/pretrained_models/original/voxel_rcnn_car_84.54.pth', # 54
        ]

cfg_files = [
         '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml',
        ]

data_root = '/home/jin/mnt/Data/KITTI/cv_inter/'
cv_algs = ['linear', 'lz4', 'nearest']

for cv_alg in cv_algs:
    cv_data_root = data_root + cv_alg
    for width, batch_size in zip(widths, batch_sizes):
        for intr_int in intr_ints:
            intr_width = int(width * intr_int)

            training_data_path = cv_data_root + '/' + str(intr_width) + '/training'
            training_link_path = '/home/jin/mnt/github/OpenPCDet/data/kitti/training/velodyne'
            print(training_data_path)

            os.system("rm f{training_link_path}")

            command = f"ln -s {training_data_path} {training_link_path}"
            os.system(command)

            for model, model_epoch, model_ckpt, cfg_file in zip(models, model_epochs, model_ckpts, cfg_files):
                output_dir = output_root + cv_alg + "/" + str(intr_width)+ '/' + model
                command = f"{trainer} --cfg_file {cfg_file} --batch_size {batch_size} --epochs {model_epoch} --ckpt {model_ckpt} --save_to_file --output_dir {output_dir}"
                print(command)
                os.system(command)
            #    python train.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 4 --epochs 5 --save_to_file --output_dir ""

