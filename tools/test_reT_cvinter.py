import os
import shutil

# python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 16 --ckpt /home/jin/mnt/github/OpenPCDet/output/kitti_models/pv_rcnn/default/ckpt/latest_model.pth

trainer = 'python /home/jin/mnt/github/OpenPCDet/tools/my_train.py'
tester = 'python /home/jin/mnt/github/OpenPCDet/tools/test.py'

widths    = [2048, 1024, 512, 256]
intr_ints = [1.25, 1.5, 1.75, 2]

output_root = '/home/jin/mnt/Data/KITTI/retrained_models/epochs/cv_inter/'

models = ['parta2',
          'pointpillar',
          'pv_rcnn',
          'second',
        ]

add_epoch = 5
model_epochs = [
        80,
        80,
        77,
        80,
        ]

model_ckpt_root = "/home/jin/mnt/Data/KITTI/retrained_models/cv_intr/"
# model_ckpts = [
#          '/home/jin/mnt/Data/KITTI/pretrained_models/original/PartA2_7940.pth',        # 80
#          '/home/jin/mnt/Data/KITTI/pretrained_models/original/pointpillar_7728.pth',   # 80
#          '/home/jin/mnt/Data/KITTI/pretrained_models/original/pv_rcnn_8369.pth',       # 77
#          '/home/jin/mnt/Data/KITTI/pretrained_models/original/second_7862.pth',        # 80
#         ]

cfg_files = [
         '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/PartA2.yaml',
         '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml',
         '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml',
         '/home/jin/mnt/github/OpenPCDet/tools/cfgs/kitti_models/second.yaml',
        ]

data_root = '/home/jin/mnt/Data/KITTI/cv_inter/'
cv_algs = ['area', 'cubic', 'linear', 'lz4', 'nearest']

for cv_alg in cv_algs:
    cv_data_root = data_root + cv_alg + '/'
    cv_model_ckpt_root = model_ckpt_root + cv_alg + '/'
    cv_output_root = output_root + cv_alg + '/'

    for width in widths:
        for intr_int in intr_ints:
            intr_width = int(width * intr_int)

            training_data_path = cv_data_root + str(intr_width) + '/training'
            training_link_path = '/home/jin/mnt/github/OpenPCDet/data/kitti/training/velodyne'

            command = f'rm {training_link_path}'
            os.system(command)
            print(command)

            command = f"ln -s {training_data_path} {training_link_path}"
            os.system(command)
            print(command)

            for model, model_epoch, cfg_file in zip(models, model_epochs, cfg_files):
                for epoch in range(1, add_epoch+1):
                    model_ckpt = cv_model_ckpt_root + str(intr_width) + '/' + model + '/ckpt/checkpoint_epoch_' + str(model_epoch + epoch) + '.pth'
                    output_dir = cv_output_root + str(intr_width) + '/' + model + '/' + str(epoch)
                    command = f"{tester} --cfg_file {cfg_file} --batch_size 4 --ckpt {model_ckpt} --save_to_file --output_dir {output_dir}"
                    os.system(command)
                    print(command)

