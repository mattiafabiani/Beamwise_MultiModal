import math
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import models
from swinv2 import SwinTransformerBlock

class SwinBasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """
    
    def __init__(self, dim, input_resolution, depth, num_heads, seq_len,window_size, config, 
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.config = config
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.seq_len = seq_len
        self.ape = nn.Parameter(torch.zeros(1, (self.config.n_views) * seq_len * self.config.vert_anchors * self.config.horz_anchors+ self.seq_len*self.config.n_gps, self.dim))

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])


    def forward(self, image, gps):
        
            bz = image.shape[0] // (self.config.n_views * self.seq_len)
            h, w = image.shape[2:4]
            image = image.view(bz, self.config.n_views * self.seq_len, -1, h, w)
            x = image.permute(0,1,3,4,2).contiguous()
            x = x.view(bz, -1, self.dim)
            x = torch.cat([x,gps], dim=1)
            x = x + self.ape
            
            for blk in self.blocks:
                x = blk(x)
            
            gps = x[:, (self.config.n_views) * self.seq_len * self.config.vert_anchors * self.config.horz_anchors:, :]
            x =  x[:,:(self.config.n_views) * self.seq_len * self.config.vert_anchors * self.config.horz_anchors,:]

            x  = x.view(bz, (self.config.n_views) * self.seq_len, self.config.vert_anchors, self.config.horz_anchors, self.dim)
            x  = x.permute(0,1,4,2,3).contiguous()
            x  = x[:, :self.config.n_views*self.seq_len, :, :, :].contiguous().view(bz * self.config.n_views * self.seq_len, -1, h, w)

            return x, gps

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class Encoder(nn.Module):
    """
    Fusion Transformer to fuse image and GPS features
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.avgpool = nn.AdaptiveAvgPool2d((self.config.vert_anchors, self.config.horz_anchors))
        self.image_encoder_stem = models.efficientnet_b3(pretrained =True).features[0]

        self.image_encoder_layer1 = models.efficientnet_b3(pretrained =True).features[1]
        self.image_encoder_layer2 = models.efficientnet_b3(pretrained =True).features[2]

        self.image_encoder_layer3 = models.efficientnet_b3(pretrained =True).features[3]

        self.image_encoder_layer4 = models.efficientnet_b3(pretrained =True).features[4]

        self.image_encoder_layer5 = models.efficientnet_b3(pretrained =True).features[5]
        self.image_encoder_layer6 = models.efficientnet_b3(pretrained =True).features[6]
        
        self.image_encoder_avgpool = models.efficientnet_b3(pretrained =True).avgpool
        
        self.vel_emb1 = nn.Linear(2, 32)
        self.vel_emb2 = nn.Linear(32, 48)
        self.vel_emb3 = nn.Linear(48, 96)
        self.vel_emb4 = nn.Linear(96, 232)

        self.transformer1 = SwinBasicLayer(dim = 32, 
                                           input_resolution = (self.config.input_resolution),
                                           depth = config.n_layer, num_heads = config.n_head,
                                           window_size = 5, seq_len = self.config.seq_len, config = self.config)

        self.transformer2 = SwinBasicLayer(dim = 48, 
                                           input_resolution =  (self.config.input_resolution),
                                           depth = config.n_layer, num_heads = config.n_head,
                                           window_size = 5, seq_len = self.config.seq_len, config = self.config)
                                           
        self.transformer3 = SwinBasicLayer(dim = 96, 
                                           input_resolution =  (self.config.input_resolution),
                                           depth = config.n_layer, num_heads = config.n_head,
                                           window_size = 5, seq_len = self.config.seq_len, config = self.config)
                                           
        self.transformer4 = SwinBasicLayer(dim = 232, 
                                           input_resolution =  (self.config.input_resolution),
                                           depth = config.n_layer, num_heads = config.n_head,
                                           window_size = 5, seq_len = self.config.seq_len, config = self.config)
        
    def forward(self, image_list, gps_list):
        '''
        Using transformers to fuse image and GPS features
        Args:
            image_list (list): list of input images
            lidar_list (list): list of gps measurements
        '''
        
        bz, _, h, w = image_list[0].shape
        img_channel = image_list[0].shape[1]
        
        self.config.n_views = len(image_list) // self.config.seq_len

        image_tensor = torch.stack(image_list, dim=1).view(bz * self.config.n_views * self.config.seq_len, img_channel, h, w)
        gps = torch.stack(gps_list, dim=1).view(bz,-1,2)
        
        image_features = self.image_encoder_stem(image_tensor)
        image_features = self.image_encoder_layer1(image_features)
        image_features = self.image_encoder_layer2(image_features)
        
        # fusion at (batch_size, 32, 56, 56)
        image_embd_layer1 = self.avgpool(image_features)
        gps_embd_layer1 = self.vel_emb1(gps)
        image_features_layer1, gps_features_layer1 = self.transformer1(image_embd_layer1, gps_embd_layer1)
        image_features_layer1 = F.interpolate(image_features_layer1, scale_factor=8, mode='bilinear')
        image_features = image_features + image_features_layer1
        image_features = self.image_encoder_layer3(image_features)
        
        # fusion at (batch_size, 48, 28, 28)
        image_embd_layer2 = self.avgpool(image_features)
        gps_embd_layer2 = self.vel_emb2(gps_features_layer1)
        image_features_layer2, gps_features_layer2 = self.transformer2(image_embd_layer2, gps_embd_layer2)
        image_features_layer2 = F.interpolate(image_features_layer2, scale_factor=4, mode='bilinear')
        image_features = image_features + image_features_layer2
        image_features = self.image_encoder_layer4(image_features)

        # fusion at (batch_size, 96, 14, 14)
        image_embd_layer3 = self.avgpool(image_features)
        gps_embd_layer3 = self.vel_emb3(gps_features_layer2)
        image_features_layer3, gps_features_layer3 = self.transformer3(image_embd_layer3, gps_embd_layer3)
        image_features_layer3 = F.interpolate(image_features_layer3, scale_factor=2, mode='bilinear')
        image_features = image_features + image_features_layer3
        image_features = self.image_encoder_layer5(image_features)
        image_features = self.image_encoder_layer6(image_features)
        
        # fusion at (batch_size, 232, 14, 14)
        image_embd_layer4 = self.avgpool(image_features)
        gps_embd_layer4 = self.vel_emb4(gps_features_layer3)
        image_features_layer4, gps_features = self.transformer4(image_embd_layer4, gps_embd_layer4)
        image_features = image_features + image_features_layer4
        
        image_features = self.image_encoder_avgpool(image_features)
        image_features = torch.flatten(image_features, 1)
        image_features = image_features.view(bz, self.config.n_views * self.config.seq_len, -1)

        fused_features = torch.cat([image_features, gps_features], dim=1)
        fused_features = torch.sum(fused_features, dim=1)

        return fused_features

class SwinFuser(nn.Module):
    '''
    Transformer-based feature fusion followed by GRU-based waypoint prediction network and PID controller
    '''

    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config = config
        self.encoder = Encoder(config).to(self.device)

        self.join = nn.Sequential(
                            nn.Linear(232, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 256),
                        ).to(self.device)
        
    def forward(self, image_list, gps_list):
        '''
        Predicts the future beam after fusing the images + gps features
        Args:
            image_list (list): list of input images
            gps_list (list): list of input gps
        '''
        fused_features = self.encoder(image_list, gps_list)
        z = self.join(fused_features)
        
        return z