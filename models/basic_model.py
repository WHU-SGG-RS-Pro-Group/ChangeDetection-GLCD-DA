import os
import numpy as np
import torch

from misc.imutils import save_image
from models.networks import *
import matplotlib.pyplot as plt

class CDEvaluator():

    def __init__(self, args):

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        self.device = torch.device("cuda:%s" % args.gpu_ids[0]
                                   if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")

        print(self.device)

        self.checkpoint_dir = args.checkpoint_dir

        self.pred_dir = args.output_folder
        self.pred_dir2 = args.output_folder2
        self.feature_map_dir = args.feature_map_dir
        os.makedirs(self.pred_dir, exist_ok=True)
        os.makedirs(self.pred_dir2, exist_ok=True)
        os.makedirs(self.feature_map_dir, exist_ok=True)
        # self.feature_map = None
        # self.module_name = None

    def load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name),
                                    map_location=self.device)

            self.net_G.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_G_state_dict'].items()})
            self.net_G.to(self.device)
            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)
        return self.net_G

    def _save_feature_map(self, feature_map, input_image_name, output_dir, module_name):
        if feature_map is None:
            print("Feature map not available.")
            return

        num_feature_maps = feature_map.shape[1]
        size = int(np.ceil(np.sqrt(num_feature_maps)))
        os.makedirs(output_dir, exist_ok=True)
        file_name = os.path.join(output_dir, f"{input_image_name[:-4]}_{module_name}.png")
        if feature_map.shape[1] == 1:  # 只有 1 个通道，直接保存灰度图
            plt.imshow(feature_map[0, 0, :, :], cmap='gray')
            plt.axis('off')
            plt.savefig(file_name)
            plt.close()
        else:
            fig, axes = plt.subplots(size, size, figsize=(100, 100))
            for i, ax in enumerate(axes.flat):
                if i < num_feature_maps:
                    # ax.imshow(feature_map[0, i, :, :], cmap='viridis')
                    ax.imshow(feature_map[0, i, :, :], cmap='rainbow')
                ax.axis('off')
            plt.savefig(file_name)
            plt.close(fig)

    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255

        return pred_vis


    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        mask = self.batch['L'].to(self.device)
        self.shape_h = img_in1.shape[-2]
        self.shape_w = img_in1.shape[-1]
        self.G_pred_64,self.G_pred_128,self.G_pred= self.net_G(img_in1, img_in2)
        # self.G_pred = self.net_G(img_in1, img_in2)
        return self._visualize_pred()


    def eval(self):
        self.net_G.eval()


    def _save_predictions(self):
        """
        保存模型输出结果，二分类图像
        """

        preds = self._visualize_pred()
        name = self.batch['name']
        for i, pred in enumerate(preds):
            file_name = os.path.join(
                self.pred_dir, name[i].replace('.jpg', '.png'))
            pred = pred[0].cpu().numpy()
            save_image(pred, file_name)

    def _save_predictions2(self):
        preds = self._visualize_pred()/255
        #print(preds)
        print('preds.shape',preds.shape)

        name = self.batch['name']
        print('name',name)
        for i, pred in enumerate(preds):
            file_name = os.path.join(
                self.pred_dir2, name[i].replace('.jpg', '.png'))
            mask = self.batch['L'].to(self.device)
            # print('mask1', mask.shape)
            mask = mask.to(self.device).numpy()
            pred = pred.to(self.device).numpy()
            # print('pred1',pred.shape)
            pred = np.squeeze(pred)
            mask = mask[i, :, :,:]
            mask = np.squeeze(mask)
            Conf_map = np.zeros_like(pred)
            Conf_map = np.tile(Conf_map[..., np.newaxis], (1, 1, 3))
            index = np.logical_and(mask, pred)
            Conf_map[index] = [1, 1, 1]
            Conf_map[np.logical_and(mask, np.logical_not(pred)), :] = [1, 0, 0]
            Conf_map[np.logical_and(np.logical_not(mask), pred), :] = [0, 1, 0]
            Conf_map=Conf_map*255.0
            save_image(Conf_map, file_name)

    def save_intermediate_feature_maps(self):
        CD=self.net_G.CD
        if CD is not None:
            input_image_name = self.batch['name'][0].replace('.jpg', '')
            self._save_feature_map(CD.sigmoid().cpu().detach().numpy(), input_image_name,
                                   self.feature_map_dir + '/CD_Map', 'CD')#Activation map
        else:
            print("Intermediate feature maps not available.")

