import os
import argparse
import torch

from networks.net_factory import net_factory
from networks.uent_3D_adv_semi import unet_3D_dv_semi
from utils.test_patch import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='/home/jwsu/semi/', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='URPC', help='exp_name')
parser.add_argument('--model', type=str, default='URPC', help='model_name')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--detail', type=int, default=0, help='print metrics for every samples?')
parser.add_argument('--labelnum', type=int, default=16, help='labeled data')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')

FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
# Pancreas_CT_MCNet_B_dice_5_spatial_12_labeled
# /mnt/imtStu/jwsu/WeightSample/LA_MCNet_B_dice_4_labeled
# /mnt/imtStu/jwsu/WeightSample/LA_MCNet_4_labeled_5_spatial_only_class/mine3d_v1
# LA_MCNet_8_labeled_5_spatial_info_t_0.1_onlyclass_proj_mat
# /mnt/imtStu/jwsu/WeightSample/LA_MCNet_4_labeled_9_spatial_only_class_1226
snapshot_path = "/mnt/imtStu/jwsu/WeightSample/{}_{}_{}_labeled_9_spatial_only_class_1226/{}".format(
    FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum, FLAGS.model)
test_save_path = "/mnt/imtStu/jwsu/WeightSample/{}_{}_{}_labeled_9_spatial_only_class_1226/{}_predictions/".format(
    FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum, FLAGS.model)

# /mnt/imtStu/jwsu/NewWeightSample/LA_MCNet_4_labeled_baseline

# snapshot_path = "/mnt/imtStu/jwsu/NewWeightSample/{}_{}_{}_labeled_baseline_ori/{}".format(
#     FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum, FLAGS.model)
# test_save_path = "/mnt/imtStu/jwsu/NewWeightSample/{}_{}_{}_labeled_baseline_ori/{}_predictions/".format(
#     FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum, FLAGS.model)

num_classes = 2
if FLAGS.dataset_name == "LA":
    patch_size = (112, 112, 80)
    FLAGS.root_path = FLAGS.root_path + 'data/LA'
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [
        FLAGS.root_path + "/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list
    ]

elif FLAGS.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    FLAGS.root_path = FLAGS.root_path + 'data/Pancreas/'
    # FLAGS.root_path = '/home/jwsu/semi' + 'data/Pancreas'
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)


def test_calculate_metric():

    # net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=num_classes, mode="test")
    net = unet_3D_dv_semi(n_classes=num_classes, in_channels=1).cuda()
    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path), strict=False)
    print("init weight from {}".format(save_mode_path))
    net.eval()

    if FLAGS.dataset_name == "LA":
        avg_metric = test_all_case(FLAGS.model,
                                   2,
                                   net,
                                   image_list,
                                   num_classes=num_classes,
                                   patch_size=(112, 112, 80),
                                   stride_xy=18,
                                   stride_z=4,
                                   save_result=True,
                                   test_save_path=test_save_path,
                                   metric_detail=FLAGS.detail,
                                   nms=FLAGS.nms)
    elif FLAGS.dataset_name == "Pancreas_CT":
        avg_metric = test_all_case(FLAGS.model,
                                   2,
                                   net,
                                   image_list,
                                   num_classes=num_classes,
                                   patch_size=(96, 96, 96),
                                   stride_xy=16,
                                   stride_z=16,
                                   save_result=True,
                                   test_save_path=test_save_path,
                                   metric_detail=FLAGS.detail,
                                   nms=FLAGS.nms)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)
