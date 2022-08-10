import os

if __name__ == '__main__':

    model_cmds = ["python precision.py --cfg_file cfgs/kitti_models/PartA2.yaml --ckpt /home/jin/mnt/PCDetModels/PartA2_7940.pth",
            "python precision.py --cfg_file cfgs/kitti_models/PartA2_free.yaml --ckpt /home/jin/mnt/PCDetModels/PartA2_free_7872.pth",
            "python precision.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt /home/jin/mnt/PCDetModels/pointpillar_7728.pth",
            "python precision.py --cfg_file cfgs/kitti_models/pointrcnn.yaml --ckpt /home/jin/mnt/PCDetModels/pointrcnn_7870.pth",
            "python precision.py --cfg_file cfgs/kitti_models/pointrcnn_iou.yaml --ckpt /home/jin/mnt/PCDetModels/pointrcnn_iou_7875.pth",
            "python precision.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt /home/jin/mnt/PCDetModels/pv_rcnn_8369.pth",
            "python precision.py --cfg_file cfgs/kitti_models/second.yaml --ckpt /home/jin/mnt/PCDetModels/second_7862.pth",
            "python precision.py --cfg_file cfgs/kitti_models/second_iou.yaml --ckpt /home/jin/mnt/PCDetModels/second_iou7909.pth",
            "python precision.py --cfg_file cfgs/kitti_models/voxel_rcnn_car.yaml --ckpt /home/jin/mnt/PCDetModels/voxel_rcnn_car_84.54.pth"]

    resolutions = [2048, 1024, 512, 256]


    for res in resolutions:
        remove_cmd = "rm -r /home/jin/github/OpenPCDet/data/kitti/training/velodyne"
        os.system(remove_cmd)
        print(remove_cmd)
        data_move_cmd = "cp ~/mnt/github/3D_PCC/bin/interpolation_test/" + str(res) + "/myinter /home/jin/github/OpenPCDet/data/kitti/training/velodyne -r"
        os.system(data_move_cmd)
        print(data_move_cmd)
        for model_cmd in model_cmds:
            os.system(model_cmd)
            print(model_cmd)

