from argparse import ArgumentParser
import torch
from models.trainer import *
#from models.trainer_unet import *
from thop import profile
print(torch.cuda.is_available())
import numpy as np
"""
the main function for training the CD networks
"""


def train(args):
    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()


def test(args):
    from models.evaluator import CDEvaluator
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test')
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models()


if __name__ == '__main__':

    print(torch.cuda.is_available())
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='California', type=str)
    parser.add_argument('--checkpoint_root', default='CHECKPOINTS_CALIFORNIA', type=str)

    # data--
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='ChangeDetection', type=str)

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='base_transformer_pos_s4', type=str,
                        help='base_resnet18 | base_transformer_pos_s4|')
    parser.add_argument('--loss', default='multi',  type=str,help='ce|ce+dice|multi')

    # optimizer
    parser.add_argument('--optimizer', default='sgd', type=str,help='sgd or adm or ranger')
    parser.add_argument('--lr', default=0.01,type=float)
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--lr_policy', default='step', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=10, type=int)
    parser.add_argument('--gamma', default=0.95, type=float)

    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train(args)
    test(args)
    Filename=os.path.join(args.checkpoint_root, args.project_name)
    File= np.load(Filename+'/scores_dict.npy', encoding="latin1",
                   allow_pickle=True)  # 加载文件
    File1 = np.load(Filename+'/train_acc.npy', encoding="latin1", allow_pickle=True)
    File2 = np.load(Filename+'/val_acc.npy', encoding="latin1", allow_pickle=True)
    doc = open(Filename+'/scores_dict.txt', 'a')
    doc1 = open(Filename+'/train_acc.txt', 'a')
    doc2 = open(Filename+'/val_acc.txt', 'a')
    print(File, file=doc)  # train_acc  val_acc
    print(File1, file=doc1)
    print(File2, file=doc2)
