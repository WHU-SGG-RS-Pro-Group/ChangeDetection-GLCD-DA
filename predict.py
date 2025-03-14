from argparse import ArgumentParser

import utils
import torch
from models.basic_model import CDEvaluator
from thop import profile
import os
import time
from models.networks import *
"""
Predict
sample files in the ./SAMPLES_CALIFORNIA
save prediction files in the ./RESULTS/RESULT_CALIFORNIA and ./RESULTS/COLOR_RESULT_CALIFORNI

"""


def get_args():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--project_name', default='California', type=str)
    parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')#CPU
    parser.add_argument('--checkpoint_root', default='CHECKPOINTS_CALIFORNIA', type=str)#checkpoint_root
    parser.add_argument('--output_folder', default='./RESULTS/CALIFORNIA', type=str)
    parser.add_argument('--output_folder2', default='./RESULTS/COLOR_CALIFORNIA', type=str)
    parser.add_argument('--feature_map_dir',default='MiddleFeature',type=str)
    # data
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='predict', type=str)

    parser.add_argument('--batch_size', default=1, type=int) #默认
    parser.add_argument('--split', default="demo", type=str) #默认
    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)

    parser.add_argument('--net_G', default='base_transformer_pos_s4', type=str,
                        help='base_resnet18 | base_transformer_pos_s4 |')
    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)
    args = parser.parse_args()
    return args
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':

    args = get_args()
    utils.get_device(args)

    device = torch.device("cuda:%s" % args.gpu_ids[0]
                          if torch.cuda.is_available() and len(args.gpu_ids)>0
                        else "cpu")
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.output_folder, exist_ok=True)

    log_path = os.path.join(args.output_folder, 'log_vis.txt')

    data_loader: object = utils.get_loader(args.data_name, img_size=args.img_size,
                                   batch_size=args.batch_size,
                                   split=args.split, is_train=False)

    model = CDEvaluator(args)


    model.load_checkpoint(args.checkpoint_name)
    model.eval()
    total_time=0
    for i, batch in enumerate(data_loader):
        name = batch['name']
        print('process: %s' % name)
        score_map = model._forward_pass(batch)
        # model._save_predictions()
        # model._save_predictions2()
        model.save_intermediate_feature_maps()




