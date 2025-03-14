import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler

import functools
from einops import rearrange

import models
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d,RCSA
from Unet.unetPlusPlus import unetPlusPlus
from models.aspp import RCSA_ASPP
from models.ELGCA import ELGCA
from models.MFDS import MDPM
from models.laplace import make_laplace_pyramid
from torchvision.transforms.functional import rgb_to_grayscale
from models.FUSE import DFIM
###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            #lr_l = 1.0 -epoch*0
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size =args.lr_decay_iters
        gamma_try=args.gamma
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma_try)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'base_resnet18':
        net = ResNet(input_nc=3, output_nc=2, output_sigmoid=False)

    elif args.net_G == 'base_transformer_pos_s4':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', dim_head=64, decoder_dim_head=64)
        #California:enc_depth=1, dec_depth=1,dim_head=64, decoder_dim_head=64
    elif args.net_G == 'unet++':
        net = unetPlusPlus(6,2)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)


###############################################################################
# main Functions
###############################################################################


class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18( nc= input_nc,pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])  # 用扩张代替步幅

        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        self.classifier2 = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        self.classifier3 = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)
        self.conv_pred2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)#128

        x_1 = self.resnet.layer1(x) # 1/4 256,in=64, out=64
        x_2 = self.resnet.layer2(x_1) # 1/8 128, in=64, out=128

        if self.resnet_stages_num > 3:
            x_4 = self.resnet.layer3(x_2) # 1/8,64 in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_4) # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_4)
            x_1 = self.upsamplex2(x_1)
        else:
            x = x_4
        # output layers
        x = self.conv_pred(x)
        x_1=self.conv_pred2(x_1)
        return x_1,x
def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )

class BASE_Transformer(ResNet):
    """

    """
    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder = True,with_fuion=True,
                 use_drop_California=False,use_drop_Gloucester2=False):
        super(BASE_Transformer, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.tokenizer = tokenizer
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        self.with_fuion=with_fuion
        dim = 32
        mlp_dim = 2*dim
        num_patches=4

        self.with_pos = with_pos
        if with_pos is 'learned':
            # self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2 , 32))
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head

        self.transformer = Transformer(dim=dim, num_patches=num_patches, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0.)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0.,
                                                      softmax=decoder_softmax)
        #opt and sar
        self.resnet1 = ResNet(input_nc=input_nc, output_nc=output_nc, backbone=backbone,
                              resnet_stages_num=resnet_stages_num,
                              if_upsample_2x=if_upsample_2x)
        self.resnet2 = ResNet(input_nc=input_nc, output_nc=output_nc, backbone=backbone,
                              resnet_stages_num=resnet_stages_num,
                              if_upsample_2x=if_upsample_2x)
        self.ELGCA=ELGCA(dim=dim*2,heads=4)
        self.ASPP=RCSA_ASPP(dim,dim,1,0.1)
        self.RCSA_rcsa=RCSA(nFeat=32)
        self.mdpm_1 = MDPM(32, 32)
        self.fusion_conv=nn.Conv2d(64, 32, kernel_size=1, padding=0, bias=False)
        self.fusex_x128=DFIM(32,32,32)
        self.use_drop_California = use_drop_California
        self.use_drop_Gloucester2=use_drop_Gloucester2
        if self.use_drop_California:
            self.drop_out = nn.Dropout(0.2)
    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape #b:8 c: 32 w:64 h:64
        spatial_attention = self.conv_a(x)  #(8,4,64,64)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous() #(8,4,4096)
        spatial_attention = torch.softmax(spatial_attention, dim=-1)#(8,4,4096)
        x = x.view([b, c, -1]).contiguous()#(8,32,4096)
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x) #(8,4,32)

        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode is 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode is 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):#x1和token1
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h) #rearrange重新排列
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x
    def _forward_fusion(self,x1,x2):#局部信息，全局信息
        x = torch.cat([x1, x2], dim=1)
        x = self.ELGCA(x)
        out = self.fusion_conv(x)
        return out





    def forward(self, x1, x2):
        #edge
        grayscale_img_x1 = rgb_to_grayscale(x1)
        edge_feature_x1 = make_laplace_pyramid(grayscale_img_x1, 5, 1)
        edge_feature_x1 = edge_feature_x1[1]
        grayscale_img_x2 = rgb_to_grayscale(x2)
        edge_feature_x2 = make_laplace_pyramid(grayscale_img_x2, 5, 1)
        edge_feature_x2 = edge_feature_x2[1]
        # forward backbone resnet
        x1_128,x1 = self.forward_single(x1)#torch.Size([8, 32, 64, 64])
        x2_128,x2 = self.forward_single(x2)

        x1=self.ASPP(x1)
        x2 = self.ASPP(x2)
        x1_1=x1
        x2_1 = x2
        # low feature
        x1_128=self.ASPP(x1_128)
        x2_128 = self.ASPP(x2_128)

        x1_128, am_x1 = self.mdpm_1(x1_128, edge_feature_x1)
        x2_128, am_x2 = self.mdpm_1(x2_128, edge_feature_x2)

        x1_128 = self.RCSA_rcsa( x1_128)
        x2_128 = self.RCSA_rcsa(x2_128)

        x_128 = x1_128 - x2_128
        out1=x_128
        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1) #torch.Size([8, 4, 32])
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1) #torch.Size([8, 8, 32])
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)
        #California
        if self.use_drop_California:
            x1 = self.drop_out(x1)
            x2 = self.drop_out(x2)
        if self.with_fuion:
            x1 = self._forward_fusion(x1_1, x1)
            x2 = self._forward_fusion(x2_1, x2)
        x1 = self.RCSA_rcsa(x1)
        x2 = self.RCSA_rcsa(x2)
        x = torch.abs(x1 - x2)
        out2=x
        x = F.interpolate(x, size=[x1_128.shape[2], x1_128.shape[3]], mode='bilinear', align_corners=True)  # up sample
        #FFM
        x = self.fusex_x128(x, x_128)
        #CAT
        # x = torch.cat([x_128, x], dim=1)
        # x=self.conv_x12(x)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex2(x)
        # forward small cnn
        x = self.classifier(x)
        if self.use_drop_Gloucester2:
            x= self.drop_out(x)
        out3=x
        self.CD = out3
        out1 = self.classifier2(out1)
        out2= self.classifier3(out2)#64
        if self.output_sigmoid:
            x = self.sigmoid(x)
        return out2,out1,out3#64,128,256