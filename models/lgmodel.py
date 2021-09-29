import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from .convlstm import ConvLSTM
from .modules import GAT, AdjGenerator

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
    elif norm_type == 'batch3d':
         norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
    elif norm_type == 'instance3d':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
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
        elif classname.find(
                'BatchNorm2d') != -1 or classname.find('BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            # print("BatchNorm Layer's weight is not a matrix; only normal distribution applies.")
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
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

def define_G(netG, input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator
    """
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == "resnet_6blocks_attention":
        net = ResnetAttentionGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == "C3D_attention":
        net = C3DAttentionGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    return init_net(net, init_type, init_gain, gpu_ids)

class ResnetAttentionGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', step=4):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetAttentionGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        self.recurrent_model = ConvLSTM(input_dim=ngf * mult,
                                   hidden_dim=[ngf * mult, ngf * mult, ngf * mult],
                                   kernel_size=(3, 3),
                                   num_layers=3,
                                   batch_first=True,
                                   bias=True,
                                   return_all_layers=False)

        de_model = [nn.Conv2d(step * ngf * mult, ngf * mult, kernel_size=3, padding=1, bias=use_bias),
                    norm_layer(ngf * mult),
                    nn.ReLU(True)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            de_model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1,
                                            bias=use_bias),
                         norm_layer(int(ngf * mult / 2)),
                         nn.ReLU(True)]
        de_model += [nn.ReflectionPad2d(3)]
        de_model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        de_model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.de_model = nn.Sequential(*de_model)

        # generate attention
        r = 2
        self.attention_meta_learner = nn.Sequential(nn.Linear(4096, int(4096/r)), nn.ReLU(),
                                                    nn.Linear(int(4096/r), 4096)) # 这里代表整个网络的frame输入尺寸为256*256

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.adjGenerator = AdjGenerator(feat_dim=256)
        self.GAT = GAT(nfeat=256, nhid=64, nclass=256, dropout=0.7, alpha=0.1, nheads=1)

        # flow feature extractor
        self.flow_feature_extractor = FlowFeatureExtractor(input_nc=2, output_nc=2)
    
    def get_attention(self, f, b):
        # calc <f,b>
        attention = []
        for i in range(b.shape[0]):
            bi = b[i]
            # cosine_matrix = torch.cosine_similarity(f, bi, dim=0)
            f = f/torch.norm(f, dim=0, p=2)
            bi = bi/torch.norm(bi, dim=0, p=2)
            cosine_matrix = f.t().mm(bi)
            # print("cosine_matrix.shape: ", cosine_matrix.shape)
            p = torch.nn.functional.avg_pool1d(cosine_matrix.unsqueeze(0), kernel_size=cosine_matrix.shape[1], stride=None)
            # print("p.shape： ", p.shape)            
            w = self.attention_meta_learner(p.view(4096))
            # print("w.shape: ", w.shape)            
            attention.append(w.view(64, 64))       
                
        attention_map = torch.stack(attention, 0)
        attention_map = torch.mean(attention_map, 0)
        attention_map = torch.nn.functional.softmax(attention_map / 0.025, dim=-1) + 1
        # print("attention_map.shape: ", attention_map.shape)
        return attention_map

    def cal_attention(self, frame_feat, object_feat):
        # frame_feat: [c, d1=w*h]  object_feat:[N, c]
        # frame_feat = frame_feat / torch.norm(frame_feat, dim=0, p=2)
        # object_feat = object_feat / torch.norm(object_feat, dim=1, p=2)
        frame_feat = nn.functional.normalize(frame_feat, dim=0, p=2)
        object_feat = nn.functional.normalize(object_feat, dim=1, p=2)
        cosine_matrix = torch.matmul(object_feat, frame_feat).t() # [d1, N]
        p = torch.nn.functional.avg_pool1d(cosine_matrix.unsqueeze(0), kernel_size=cosine_matrix.shape[1],
                                           stride=None)
        attention_map = self.attention_meta_learner(p.view(4096))
        attention_map = torch.nn.functional.softmax(attention_map / 0.025, dim=-1) + 1
        attention_map = attention_map.reshape(64, 64)
        return attention_map


    def forward(self, frames, objects, rois, flows):    # frames/objects: N x T x c x h x w，roi为目标的坐标
        # 这里的N是batchsize， 都是1
        T = frames.shape[1]
        frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
        objects = objects.reshape(-1, objects.shape[-3], objects.shape[-2], objects.shape[-1]) #[N, c, w, h]
        flows = flows.reshape(-1, flows.shape[-3], flows.shape[-2], flows.shape[-1])
        # print("objects.shape: ",objects.shape)

        frame_feat = self.model(frames)    # f: (N * T) x (ngf * 2 ** n_downsampling) x h' x w' 
        # print(frame_feat.shape) # [4, 256, 64, 64]
        frame_med_feat = frame_feat[int(T/2)].view(frame_feat.shape[1], -1)   # [c=256, d1=4096]
        # print("frame_med_feat.shape: ", frame_med_feat.shape) 
        # print("f.shape: {}, ft.shape: {}".format(f.shape, ft.shape))

        object_feat = self.model(objects) # [N, c, h'', w'']
        # print("object_feat.shape: ", object_feat.shape)
        object_feat = self.GAP(object_feat).squeeze(-1).squeeze(-1)  # [N, c]
        # print("object_feat.shape: ", object_feat.shape)

        flow_feat, flow_out = self.flow_feature_extractor(flows)
        flow_feat = self.GAP(flow_feat).squeeze(-1).squeeze(-1)
        # print("flow_feat.shape: ", flow_feat.shape)

        # 生成邻接矩阵然后采用GAT聚合特征，这里暂且写个草稿
        position_embedding = self.adjGenerator.cal_position_embedding(rois, rois)
        adj = self.adjGenerator(flow_feat, flow_feat, position_embedding) # [N, gap.nclass, N]
        # print("adj.shape: ", adj.shape)

        box_feature_aggr = self.GAT(object_feat, adj)   # [N, F=gap.nclass]
        # print("box_feature_aggr: ", box_feature_aggr.shape)

        #聚合完了后续就类似地聚合全局特征
        attention_map = self.cal_attention(frame_med_feat, box_feature_aggr)
        z = attention_map * frame_feat
        # print(frame_feat.shape)
        
        # # 对比实验
        # z = frame_feat

        # print("z.shape: ", z.shape)
        
        z = z.reshape(-1, T, z.shape[-3], z.shape[-2], z.shape[-1])
        out_recurrent,_ = self.recurrent_model(z)
        out_recurrent = out_recurrent[0].reshape(out_recurrent[0].shape[0], -1, out_recurrent[0].shape[-2], out_recurrent[0].shape[-1])
        # print("out_recurrent.shape: ", out_recurrent.shape) #[bs, 1024, 64, 64]
        out = self.de_model(out_recurrent)
        # print(out.shape)

        return out, flow_out

class C3DAttentionGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6,
                 padding_type='zero', step=4):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(C3DAttentionGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        model = [nn.Conv3d(input_nc, ngf, kernel_size=(3,3,3), padding=(1,1,1), bias=use_bias),
                norm_layer(ngf),
                nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=(1,2,2), padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [Resnet3dBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias),
                    nn.ReLU(True)]

        model += [nn.Conv3d(ngf*mult, ngf*mult, kernel_size=(4,1,1), stride=1,bias=use_bias),
                    nn.ReLU(True)]

        norm_layer = nn.BatchNorm2d
        de_model = [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, padding=1, bias=use_bias),
                    norm_layer(ngf * mult),
                    nn.ReLU(True)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            de_model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1,
                                            bias=use_bias),
                         norm_layer(int(ngf * mult / 2)),
                         nn.ReLU(True)]
        de_model += [nn.ReflectionPad2d(3)]
        de_model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        de_model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.de_model = nn.Sequential(*de_model)

        # generate attention
        r = 2
        self.attention_meta_learner = nn.Sequential(nn.Linear(4096, int(4096/r)), nn.ReLU(),
                                                    nn.Linear(int(4096/r), 4096)) # 这里代表整个网络的frame输入尺寸为256*256

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.adjGenerator = AdjGenerator(feat_dim=256)
        self.GAT = GAT(nfeat=256, nhid=64, nclass=256, dropout=0.7, alpha=0.1, nheads=1)

        # flow feature extractor
        self.flow_feature_extractor = FlowFeatureExtractor(input_nc=2, output_nc=2)
    
    def get_attention(self, f, b):
        # calc <f,b>
        attention = []
        for i in range(b.shape[0]):
            bi = b[i]
            # cosine_matrix = torch.cosine_similarity(f, bi, dim=0)
            f = f/torch.norm(f, dim=0, p=2)
            bi = bi/torch.norm(bi, dim=0, p=2)
            cosine_matrix = f.t().mm(bi)
            # print("cosine_matrix.shape: ", cosine_matrix.shape)
            p = torch.nn.functional.avg_pool1d(cosine_matrix.unsqueeze(0), kernel_size=cosine_matrix.shape[1], stride=None)
            # print("p.shape： ", p.shape)            
            w = self.attention_meta_learner(p.view(4096))
            # print("w.shape: ", w.shape)            
            attention.append(w.view(64, 64))       
                
        attention_map = torch.stack(attention, 0)
        attention_map = torch.mean(attention_map, 0)
        attention_map = torch.nn.functional.softmax(attention_map / 0.025, dim=-1) + 1
        # print("attention_map.shape: ", attention_map.shape)
        return attention_map

    def cal_attention(self, frame_feat, object_feat):
        # frame_feat: [c, d1=w*h]  object_feat:[N, c]
        # frame_feat = frame_feat / torch.norm(frame_feat, dim=0, p=2)
        # object_feat = object_feat / torch.norm(object_feat, dim=1, p=2)
        frame_feat = nn.functional.normalize(frame_feat, dim=0, p=2)
        object_feat = nn.functional.normalize(object_feat, dim=1, p=2)
        cosine_matrix = torch.matmul(object_feat, frame_feat).t() # [d1, N]
        p = torch.nn.functional.avg_pool1d(cosine_matrix.unsqueeze(0), kernel_size=cosine_matrix.shape[1],
                                           stride=None)
        attention_map = self.attention_meta_learner(p.view(4096))
        attention_map = torch.nn.functional.softmax(attention_map / 0.025, dim=-1) + 1
        attention_map = attention_map.reshape(64, 64)
        return attention_map


    def forward(self, frames, objects, rois, flows):    # frames/objects: N x T x c x h x w，roi为目标的坐标
        # input: frames[batchsize,T,c,h,w], objects[batchsize, num, T, c, h, w]
        # frames 和 object都reshape成[N,C,T,H,W] N=batch_size*object_num
        T = frames.shape[1]
        frames = frames.permute(0,2,1,3,4)
        # print("frames.shape: ", frames.shape)
        objects = objects.view(-1, objects.shape[-4], objects.shape[-3], objects.shape[-2], objects.shape[-1]) #[N, T, c, w, h]
        objects = objects.permute(0,2,1,3,4)
        # print("objects.shape: ", objects.shape)
        flows = flows.reshape(-1, flows.shape[-3], flows.shape[-2], flows.shape[-1])
        # print("objects.shape: ",objects.shape)

        frame_feat = self.model(frames)    
        # print("frame_feat.shape", frame_feat.shape) # [bs=1, c=256, 1, h=64, w=64]
        frame_feat = frame_feat.squeeze(2) #[bs=1,c=256, h=64, w=64]
        # frame_feat = torch.mean(frame_feat, dim=2)# [1, 256, 64, 64] # mixed 
        # print(frame_feat.shape) #[256, 4096]

        object_feat = self.model(objects) # [N, 256, 1, 16, 16]
        object_feat = object_feat.squeeze(2)
        # print("object_feat.shape: ", object_feat.shape) #[N, 256, 4, 16, 16]
        # object_feat = torch.mean(object_feat, dim=2) # [N, 256, 16, 16]
        object_feat = self.GAP(object_feat).view(object_feat.shape[0], object_feat.shape[1]) # [N, 256]
        # print("object_feat.shape: ", object_feat.shape)

        flow_feat, flow_out = self.flow_feature_extractor(flows)
        flow_feat = self.GAP(flow_feat).squeeze(-1).squeeze(-1)
        # print("flow_feat.shape: ", flow_feat.shape)

        # 生成邻接矩阵然后采用GAT聚合特征，这里暂且写个草稿
        position_embedding = self.adjGenerator.cal_position_embedding(rois, rois)
        adj = self.adjGenerator(flow_feat, flow_feat, position_embedding) # [N, gap.nclass, N]
        # print("adj.shape: ", adj.shape)

        box_feature_aggr = self.GAT(object_feat, adj)   # [N, 256]
        # print("box_feature_aggr: ", box_feature_aggr.shape) 

        #聚合完了后续就类似地聚合全局特征
        # for batch_size = 1
        frame_feature_reshape = frame_feat.view(frame_feat.shape[1], -1)
        # print(frame_feature_reshape.shape)        
        attention_map = self.cal_attention(frame_feature_reshape, box_feature_aggr) 
        # print("attention_map.shape:", attention_map.shape)
        # print("frame_feat.shape", frame_feat.shape)
        z = attention_map * frame_feat
        # print("z.shape: ",z.shape) #[1, 256, 64, 64]

        # # 对比实验
        # z = frame_feat

        out = self.de_model(z)
        # print(out.shape)

        return out, flow_out


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class Resnet3dBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):

        super(Resnet3dBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        
        padding_type = 'zero'
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
        
class FlowFeatureExtractor(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=3,
                 padding_type='reflect', step=4):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(FlowFeatureExtractor, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        self.model = nn.Sequential(*model)

        de_model = [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, padding=1, bias=use_bias),
                    norm_layer(ngf * mult),
                    nn.ReLU(True)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            de_model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1,
                                            bias=use_bias),
                         norm_layer(int(ngf * mult / 2)),
                         nn.ReLU(True)]
        de_model += [nn.ReflectionPad2d(3)]
        de_model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        de_model += [nn.Tanh()]


        self.demodel = nn.Sequential(*de_model)

    def forward(self, input):    # input: N x T x c x h x w 
        feat = self.model(input)    # z: (N * T) x (ngf * 2 ** n_downsampling) x h' x w'
        out =self.demodel(feat) 
        return feat, out

