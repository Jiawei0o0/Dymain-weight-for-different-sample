from networks.unet import UNet, MCNet2d_v1, MCNet2d_v2, MCNet2d_v3, Mine2d_v1
from networks.VNet import VNet, MCNet3d_v1, MCNet3d_v2, Mine3d_v1, Mine3d_v2

def net_factory(net_type="unet", in_chns=1, class_num=4, mode = "train"):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mine2d_v1":
        net = Mine2d_v1(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "vnet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "mine3d_v1" and mode == "train":
        net = Mine3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "mine3d_v2" and mode == "train":
        net = Mine3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "mine3d_v1" and mode == "test":
        net = Mine3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "mine3d_v2" and mode == "test":
        net = Mine3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    return net
