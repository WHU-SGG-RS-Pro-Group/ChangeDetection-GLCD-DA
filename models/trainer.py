import numpy as np
import matplotlib.pyplot as plt
import os

import utils
from models.networks import *

import torch
import torch.optim as optim
from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy
import models.losses as losses
import models.focalloss as focalloss
from misc.logger_tool import Logger, Timer
from models.drop import dropblock_step
from utils import de_norm


class CDTrainer():

    def __init__(self, args, dataloaders):

        self.dataloaders = dataloaders

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)

        # Learning rate and Beta1 for Adam optimizers
        self.lr = args.lr

        # define optimizers
        if args.optimizer =='adm':
            self.optimizer_G = optim.Adam(self.net_G.parameters(),lr=self.lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
            print('adm')
            #self.optimizer_G = adamod.AdaMod(self.net_G.parameters(), lr=self.lr, beta3=0.9, eps=1e-8, weight_decay=0)
        elif args.optimizer =='sgd':
            self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr,
                                         momentum=0.9,
                                         weight_decay=5e-4)
        else:
            return NotImplementedError('optimizer [%s] is not implemented', args.optimizer)

        # define lr schedulers
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)

        self.running_metric = ConfuseMatrixMeter(n_class=2)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs

        self.global_step = 0
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.G_loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        # self.vis_dir = args.vis_dir
        self.no_optim=0
        self.warmup_epochs=10

        # define the loss functions
        if args.loss == 'ce':
            self._pxl_loss = cross_entropy
        elif args.loss == 'bce':
            self._pxl_loss = losses.binary_ce
        elif args.loss == 'ce+dice':
            self._pxl_loss = losses.new_loss
        elif args.loss == 'focalloss':
            self._pxl_loss = focalloss.FocalLoss
        elif args.loss == 'ce+dice+edge':
            self._pxl_loss = losses.entropy_Dice_edge_loss
        elif args.loss == 'multi':
            self._pxl_loss1 = losses.edge_loss
            self._pxl_loss2 = losses.new_loss

        else:
            raise NotImplemented(args.loss)

        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))

        self.VAL_loss = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_loss.npy')):
            self.VAL_loss = np.load(os.path.join(self.checkpoint_dir, 'val_loss.npy'))

        self.TRAIN_loss = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_loss.npy')):
            self.TRAIN_loss= np.load(os.path.join(self.checkpoint_dir, 'train_loss.npy'))
        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            print('training from scratch...')
    def _load_best_checkpoint(self, ckpt_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir_1, ckpt_name)):
            self.logger.write('loading best checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir_1, ckpt_name),
                                    map_location=self.device)

            def transfer_weights(checkpoint_state_dict, model):
                model_state_dict = model.state_dict()
                for name, param in checkpoint_state_dict.items():
                    if name in model_state_dict and param.size() == model_state_dict[name].size():
                        model_state_dict[name].copy_(param)

            # update net_G states
            transfer_weights(checkpoint['model_G_state_dict'], self.net_G)
            self.net_G.to(self.device)
            self.logger.write('\n')

        else:
            print('training from scratch...')


    def _timer_update(self):
        self.global_step = (self.epoch_id-self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est

    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()

    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()

        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])###加入显示test精度后修改
            #m = len(self.dataloaders['test'])

        imps, est = self._timer_update()
        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, running_mf1: %.5f\n' %\
                      (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                     imps*self.batch_size, est,
                     self.G_loss.item(), running_acc)
            self.logger.write(message)

    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']#修改此处改变评价所看指标
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
              (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_acc))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message+'\n')
        self.logger.write('\n')

    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n'
              % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

        # update the best model (based on eval acc)
        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')
            self.no_optim=0
        else:
            self.no_optim += 1

    def _update_training_acc_curve(self):
        # update train acc curve
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        # update val acc curve
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

    def _clear_cache(self):
        self.running_metric.clear()


    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.G_pred_64,self.G_pred_128,self.G_pred= self.net_G(img_in1, img_in2)
        # self.G_pred = self.net_G(img_in1, img_in2)
        # IMG = torch.cat((img_in2, img_in2), dim=1)  # UNET++
        # self.G_pred = self.net_G(IMG)

    def _backward_G(self):
        gt = self.batch['L'].to(self.device)#(8,1,256,256)
        gt2 = F.interpolate(gt, size=128, mode='nearest')
        gt3 = F.interpolate(gt, size=64, mode='nearest')
        # gt2 = F.interpolate(gt, size=64, mode='nearest')
        # gt3 = F.interpolate(gt, size=32, mode='nearest')
        gt = gt.long()
        gt2 = gt2.long()
        gt3 = gt3.long()
        self.G_loss = self._pxl_loss2(self.G_pred, gt) +  self._pxl_loss2(self.G_pred_64,gt3) +  self._pxl_loss1(
            self.G_pred_128, gt2)
        self.G_loss.backward()


    def train_models(self):
        self._load_checkpoint()
        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):

            ################## train #################
            ##########################################
            self._clear_cache()
            self.is_training = True
            self.net_G.train()  # Set model to training mode
            # if self.warmup_epochs and (self.epoch_id + 1) < self.warmup_epochs:
            #     warmup_percent_done = (self.epoch_id + 1) / self.warmup_epochs
            #     warmup_learning_rate = self.lr * warmup_percent_done  # gradual warmup_lr
            #     learning_rate = warmup_learning_rate
            #     self.optimizer_G.param_groups[0]['lr'] = learning_rate * 2
            #     # self.optimizer_G.param_groups[1]['lr'] = learning_rate
            # elif self.warmup_epochs and (self.epoch_id + 1) == self.warmup_epochs:
            #     self.optimizer_G.param_groups[0]['lr'] = self.lr
            #     # self.optimizer_G.param_groups[1]['lr'] = self.lr

            # Iterate over data.
            self.logger.write('lr: %0.7f\n' % self.optimizer_G.param_groups[0]['lr'])
            for self.batch_id, batch in enumerate(self.dataloaders['train'], 0):
                self._forward_pass(batch)
                # update G
                self.optimizer_G.zero_grad()
                self._backward_G()
                self.optimizer_G.step()
                self._collect_running_batch_states()
                self._timer_update()

            self._collect_epoch_states()
            self._update_training_acc_curve()
            self._update_lr_schedulers()
            # dropblock_step(self.net_G)


            ################## Eval ##################
            ##########################################
            self.logger.write('Begin evaluation...\n')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()

            # Iterate over data.遍历数据
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    self._forward_pass(batch)
                self._collect_running_batch_states()
            self._collect_epoch_states()

            ########### Update_Checkpoints ###########
            ##########################################
            self._update_val_acc_curve()
            self._update_checkpoints()

            # #################test#################
            #
            # self.logger.write('Begin test...\n')
            # self._clear_cache()
            # self.is_training = False
            # self.net_G.eval()
            #
            # # Iterate over data.遍历数据
            # for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
            #     with torch.no_grad():
            #         self._forward_pass(batch)
            #     self._collect_running_batch_states()
            # self._collect_epoch_states()

