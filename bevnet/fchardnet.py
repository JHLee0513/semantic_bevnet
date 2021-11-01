#https://github.com/PingoLH/FCHarDNet/blob/master/ptsemseg/models/hardnet.py
import collections
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, bn=False):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                          stride=stride, padding=kernel//2, bias = False))

        if bn:
            self.add_module('norm', nn.BatchNorm2d(out_channels))
        # self.add_module('norm', nn.GroupNorm(num_groups=4, num_channels=out_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

        #print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)

    def forward(self, x):
        return super().forward(x)
        

class BRLayer(nn.Sequential):
    def __init__(self, in_channels, bn=False):
        super().__init__()
        if bn:
            self.add_module('norm', nn.BatchNorm2d(in_channels))
        # self.add_module('norm', nn.GroupNorm(num_groups=4, num_channels=in_channels))
        self.add_module('relu', nn.ReLU(True))
    def forward(self, x):
        return super().forward(x)


class HarDBlock_v2(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
          return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
          dv = 2 ** i
          if layer % dv == 0:
            k = layer - dv
            link.insert(0, k)
            if i > 0:
                out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
          ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
          in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, dwconv=False):
        super().__init__()
        self.links = []
        conv_layers_ = []
        bnrelu_layers_ = []
        self.layer_bias = []
        self.out_channels = 0
        self.out_partition = collections.defaultdict(list)

        for i in range(n_layers):
          outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
          self.links.append(link)
          for j in link:
            self.out_partition[j].append(outch)

        cur_ch = in_channels
        for i in range(n_layers):
          accum_out_ch = sum( self.out_partition[i] )
          real_out_ch = self.out_partition[i][0]
          #print( self.links[i],  self.out_partition[i], accum_out_ch)
          conv_layers_.append( nn.Conv2d(cur_ch, accum_out_ch,
              kernel_size=3, stride=1, padding=1, bias=True) )
          bnrelu_layers_.append( BRLayer(real_out_ch) )
          cur_ch = real_out_ch
          if (i % 2 == 0) or (i == n_layers - 1):
            self.out_channels += real_out_ch
        #print("Blk out =",self.out_channels)

        self.conv_layers = nn.ModuleList(conv_layers_)
        self.bnrelu_layers = nn.ModuleList(bnrelu_layers_)
    
    def transform(self, blk, trt=False):
        # Transform weight matrix from a pretrained HarDBlock v1
        in_ch = blk.layers[0][0].weight.shape[1]
        for i in range(len(self.conv_layers)):
            link = self.links[i].copy()
            link_ch = [blk.layers[k-1][0].weight.shape[0] if k > 0 else 
                       blk.layers[0  ][0].weight.shape[1] for k in link]
            part = self.out_partition[i]
            w_src = blk.layers[i][0].weight
            b_src = blk.layers[i][0].bias

            self.conv_layers[i].weight[0:part[0], :, :,:] = w_src[:, 0:in_ch, :,:]
            self.layer_bias.append(b_src)
            
            if b_src is not None:
                if trt:
                    self.conv_layers[i].bias[1:part[0]] = b_src[1:]
                    self.conv_layers[i].bias[0] = b_src[0]
                    self.conv_layers[i].bias[part[0]:] = 0
                    self.layer_bias[i] = None
                else:
                    #for pytorch, add bias with standalone tensor is more efficient than within conv.bias
                    #this is because the amount of non-zero bias is small, 
                    #but if we use conv.bias, the number of bias will be much larger
                    self.conv_layers[i].bias = None
            else:
                self.conv_layers[i].bias = None 

            in_ch = part[0]
            link_ch.reverse()
            link.reverse()
            if len(link) > 1:
                for j in range(1, len(link) ):
                    ly  = link[j]
                    part_id  = self.out_partition[ly].index(part[0])
                    chos = sum( self.out_partition[ly][0:part_id] )
                    choe = chos + part[0]
                    chis = sum( link_ch[0:j] )
                    chie = chis + link_ch[j]
                    self.conv_layers[ly].weight[chos:choe, :,:,:] = w_src[:, chis:chie,:,:]
            
            #update BatchNorm or remove it if there is no BatchNorm in the v1 block
            self.bnrelu_layers[i] = None
            if isinstance(blk.layers[i][1], nn.GroupNorm):
                self.bnrelu_layers[i] = nn.Sequential(
                         blk.layers[i][1],
                         blk.layers[i][2])
            else:
                self.bnrelu_layers[i] = blk.layers[i][1]

    def forward(self, x):
        layers_ = []
        outs_ = []
        xin = x
        for i in range(len(self.conv_layers)):
            link = self.links[i]
            part = self.out_partition[i]

            xout = self.conv_layers[i](xin)
            layers_.append(xout)

            xin = xout[:,0:part[0],:,:] if len(part) > 1 else xout
            if self.layer_bias[i] is not None:
                xin += self.layer_bias[i].view(1,-1,1,1)

            if len(link) > 1:
                for j in range( len(link) - 1 ):
                    ly  = link[j]
                    part_id  = self.out_partition[ly].index(part[0])
                    chs = sum( self.out_partition[ly][0:part_id] )
                    che = chs + part[0]                    
                    
                    xin += layers_[ly][:,chs:che,:,:]
                    
            xin = self.bnrelu_layers[i](xin)

            if i%2 == 0 or i == len(self.conv_layers)-1:
              outs_.append(xin)

        out = torch.cat(outs_, 1)
        return out


class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
          return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
          dv = 2 ** i
          if layer % dv == 0:
            k = layer - dv
            link.append(k)
            if i > 0:
                out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
          ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
          in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels
 
    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, bn=False):
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.grmul = grmul
        self.n_layers = n_layers
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0 # if upsample else in_channels
        for i in range(n_layers):
          outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
          self.links.append(link)
          use_relu = residual_out
          layers_.append(ConvLayer(inch, outch, bn=bn))
          if (i % 2 == 0) or (i == n_layers - 1):
            self.out_channels += outch
        #print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)


    def forward(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
          if (i == 0 and self.keepBase) or \
             (i == t-1) or (i%2 == 1):
              out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #print("upsample",in_channels, out_channels)

    def forward(self, x, skip, concat=True):
        out = F.interpolate(
                x,
                size=(skip.size(2), skip.size(3)),
                mode="bilinear",
                align_corners=True,
                            )
        if concat:                            
          out = torch.cat([out, skip], 1)
          
        return out


class hardnet(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes=19,
                 final_conv=False, batch_norm=False, output_feature_map=False, resize_input=False):
        super(hardnet, self).__init__()

        self.output_feature_map = output_feature_map
        self.resize_input = resize_input

        first_ch  = [16,24,32,48]
        ch_list = [  64, 96, 160, 224, 320]
        grmul = 1.7
        gr       = [  10,16,18,24,32]
        n_layers = [   4, 4, 8, 8, 8]

        blks = len(n_layers) 
        self.shortcut_layers = []

        self.base = nn.ModuleList([])
        self.base.append (ConvLayer(in_channels=in_channels, out_channels=first_ch[0], kernel=3,
                                    stride=2, bn=batch_norm))
        self.base.append ( ConvLayer(first_ch[0], first_ch[1],  kernel=3, bn=batch_norm))
        self.base.append ( ConvLayer(first_ch[1], first_ch[2],  kernel=3, stride=2, bn=batch_norm) )
        self.base.append ( ConvLayer(first_ch[2], first_ch[3],  kernel=3, bn=batch_norm) )

        skip_connection_channel_counts = []
        ch = first_ch[3]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], bn=batch_norm)
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)
            self.base.append ( blk )
            if i < blks-1:
              self.shortcut_layers.append(len(self.base)-1)

            self.base.append ( ConvLayer(ch, ch_list[i], kernel=1, bn=batch_norm) )
            ch = ch_list[i]
            
            if i < blks-1:            
              self.base.append ( nn.AvgPool2d(kernel_size=2, stride=2) )

        cur_channels_count = ch
        prev_block_channels = ch
        n_blocks = blks-1
        self.n_blocks =  n_blocks

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up    = nn.ModuleList([])
        
        for i in range(n_blocks-1,-1,-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append(ConvLayer(cur_channels_count, cur_channels_count//2, kernel=1, bn=batch_norm))
            cur_channels_count = cur_channels_count//2

            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i], bn=batch_norm)
            
            self.denseBlocksUp.append(blk)
            prev_block_channels = blk.get_out_ch()
            cur_channels_count = prev_block_channels

        if final_conv:
            self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
                   out_channels=n_classes, kernel_size=1, stride=1,
                   padding=0, bias=True)

        # def weights_init(m):
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_normal_(m.weight)
        # self.apply(weights_init)
    
    def v2_transform(self, trt=False):        
        for i in range( len(self.base)):
            if isinstance(self.base[i], HarDBlock):
                blk = self.base[i]
                self.base[i] = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers)
                self.base[i].transform(blk, trt)

        for i in range(self.n_blocks):
            blk = self.denseBlocksUp[i]
            self.denseBlocksUp[i] = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers)
            self.denseBlocksUp[i].transform(blk, trt)

    def forward(self, x):
        skip_connections = []
        size_in = x.size()

        if self.resize_input:
            x = F.interpolate(x, size=(1024, 2048), mode='bilinear', align_corners=True)

        for i in range(len(self.base)):
            x = self.base[i](x)
            if i in self.shortcut_layers:
                skip_connections.append(x)
        out = x
        
        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip, True)
            out = self.conv1x1_up[i](out)
            out = self.denseBlocksUp[i](out)

        if not self.output_feature_map:
            out = self.finalConv(out)

        out = F.interpolate(
                            out,
                            size=(size_in[2], size_in[3]),
                            mode="bilinear",
                            align_corners=True)
        return out


class hardnet256(nn.Module):
    """
    Tuned for input size of 256 x 256 (or similar)
    """
    def __init__(self, input_ch, n_classes=19):
        super(hardnet256, self).__init__()

        ch_list = [64, 96, 160, 224, 320]
        grmul = 1.7
        gr = [10, 16, 18, 24, 32]
        n_layers = [4, 4, 8, 8, 8]

        blks = len(n_layers)
        self.shortcut_layers = []

        self.base = nn.ModuleList([])

        skip_connection_channel_counts = []
        ch = input_ch
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)
            self.base.append(blk)
            if i < blks - 1:
                self.shortcut_layers.append(len(self.base) - 1)

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]

            if i < blks - 1:
                self.base.append(nn.AvgPool2d(kernel_size=2, stride=2))

        cur_channels_count = ch
        prev_block_channels = ch
        n_blocks = blks - 1
        self.n_blocks = n_blocks

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up = nn.ModuleList([])

        for i in range(n_blocks - 1, -1, -1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append(ConvLayer(cur_channels_count, cur_channels_count // 2, kernel=1))
            cur_channels_count = cur_channels_count // 2

            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])

            self.denseBlocksUp.append(blk)
            prev_block_channels = blk.get_out_ch()
            cur_channels_count = prev_block_channels

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
                                   out_channels=n_classes, kernel_size=1, stride=1,
                                   padding=0, bias=True)

        # def weights_init(m):
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_normal_(m.weight)
        # self.apply(weights_init)

    def v2_transform(self, trt=False):
        for i in range(len(self.base)):
            if isinstance(self.base[i], HarDBlock):
                blk = self.base[i]
                self.base[i] = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers)
                self.base[i].transform(blk, trt)

        for i in range(self.n_blocks):
            blk = self.denseBlocksUp[i]
            self.denseBlocksUp[i] = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers)
            self.denseBlocksUp[i].transform(blk, trt)

    def forward(self, x):
        skip_connections = []
        size_in = x.size()

        for i in range(len(self.base)):
            x = self.base[i](x)
            if i in self.shortcut_layers:
                skip_connections.append(x)
        out = x

        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip, True)
            out = self.conv1x1_up[i](out)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        out = F.interpolate(
            out,
            size=(size_in[2], size_in[3]),
            mode="bilinear",
            align_corners=True)
        return out


class HardNet256Skip(hardnet256):
    def __init__(self, input_ch, n_classes=19):
        super(hardnet256, self).__init__()

        ch_list = [64, 96, 160, 224, 320]
        grmul = 1.7
        gr = [10, 16, 18, 24, 32]
        n_layers = [4, 4, 8, 8, 8]

        blks = len(n_layers)
        self.shortcut_layers = []

        self.base = nn.ModuleList([])

        skip_connection_channel_counts = []
        ch = input_ch
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)
            self.base.append(blk)
            if i < blks - 1:
                self.shortcut_layers.append(len(self.base) - 1)

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]

            if i < blks - 1:
                self.base.append(nn.AvgPool2d(kernel_size=2, stride=2))

        cur_channels_count = ch
        prev_block_channels = ch
        n_blocks = blks - 1
        self.n_blocks = n_blocks

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up = nn.ModuleList([])

        for i in range(n_blocks - 1, -1, -1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append(ConvLayer(cur_channels_count, cur_channels_count // 2, kernel=1))
            cur_channels_count = cur_channels_count // 2

            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])

            self.denseBlocksUp.append(blk)
            prev_block_channels = blk.get_out_ch()
            cur_channels_count = prev_block_channels

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count + input_ch,
                                   out_channels=n_classes, kernel_size=1, stride=1,
                                   padding=0, bias=True)

    def forward(self, x):
        skip_connections = []
        size_in = x.size()

        inputs = x

        for i in range(len(self.base)):
            x = self.base[i](x)
            if i in self.shortcut_layers:
                skip_connections.append(x)
        out = x

        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip, True)
            out = self.conv1x1_up[i](out)
            out = self.denseBlocksUp[i](out)

        out = torch.cat([inputs, out], dim=1)

        out = self.finalConv(out)
        out = F.interpolate(
            out,
            size=(size_in[2], size_in[3]),
            mode="bilinear",
            align_corners=True)
        return out

class HardNet256Skip_SR(hardnet256):
    def __init__(self, input_ch, n_classes=19):
        super(hardnet256, self).__init__()

        ch_list = [64, 96, 160, 224, 320]
        grmul = 1.7
        gr = [10, 16, 18, 24, 32]
        n_layers = [4, 4, 8, 8, 8]

        blks = len(n_layers)
        self.shortcut_layers = []

        self.base = nn.ModuleList([])

        skip_connection_channel_counts = []
        ch = input_ch
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)
            self.base.append(blk)
            if i < blks - 1:
                self.shortcut_layers.append(len(self.base) - 1)

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]

            if i < blks - 1:
                self.base.append(nn.AvgPool2d(kernel_size=2, stride=2))

        cur_channels_count = ch
        prev_block_channels = ch
        n_blocks = blks - 1
        self.n_blocks = n_blocks

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up = nn.ModuleList([])

        for i in range(n_blocks - 1, -1, -1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append(ConvLayer(cur_channels_count, cur_channels_count // 2, kernel=1))
            cur_channels_count = cur_channels_count // 2

            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])

            self.denseBlocksUp.append(blk)
            prev_block_channels = blk.get_out_ch()
            cur_channels_count = prev_block_channels


        num_latent_channels = cur_channels_count + input_ch

        self.sr = SqueezeReweighted(num_latent_channels, num_latent_channels, 4)

        self.finalConv = nn.Conv2d(in_channels=num_latent_channels,
                                   out_channels=n_classes, kernel_size=1, stride=1,
                                   padding=0, bias=True)

    def forward(self, x):
        skip_connections = []
        size_in = x.size()

        inputs = x

        for i in range(len(self.base)):
            x = self.base[i](x)
            if i in self.shortcut_layers:
                skip_connections.append(x)
        out = x

        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip, True)
            out = self.conv1x1_up[i](out)
            out = self.denseBlocksUp[i](out)

        out = torch.cat([inputs, out], dim=1)
        out = self.sr(out)
        out = self.finalConv(out)
        out = F.interpolate(
            out,
            size=(size_in[2], size_in[3]),
            mode="bilinear",
            align_corners=True)
        return out


class HardNet1024Skip(nn.Module):
    """
    Tuned for input size of 1024 x 1024 (or similar)
    """
    def __init__(self, input_ch, n_classes=19, guide=[], guide_num_channels=0):
        super(HardNet1024Skip, self).__init__()

        for x in guide:
            if x not in ['input', 'encoder', 'decoder']:
                raise Exception('Invalid guide {}'.format(x))


        self.guide_input = ('input' in guide) * guide_num_channels
        self.guide_encoder = ('encoder' in guide) * guide_num_channels
        self.guide_decoder = ('decoder' in guide) * guide_num_channels
      
        
        ch_list = [32, 64, 96, 128, 160]
        grmul = 1.7
        gr = [10, 16, 18, 24, 32]
        n_layers = [4, 4, 8, 8, 8]

        blks = len(n_layers)
        self.shortcut_layers = []

        self.base = nn.ModuleList([])
        self.predownsample_ch = 32
        self.base.append(ConvLayer(in_channels=input_ch+self.guide_input,
            out_channels=self.predownsample_ch, kernel=3, stride=2))

        self.guided_layers = []
        skip_connection_channel_counts = []
        ch = 32
        for i in range(blks):
            self.guided_layers.append(len(self.base) - 1)
            ch += self.guide_encoder
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)
            self.base.append(blk)
            if i < blks - 1:
                self.shortcut_layers.append(len(self.base) - 1)

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]

            if i < blks - 1:
                self.base.append(nn.AvgPool2d(kernel_size=2, stride=2))

        cur_channels_count = ch
        prev_block_channels = ch
        n_blocks = blks - 1
        self.n_blocks = n_blocks

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up = nn.ModuleList([])

        for i in range(n_blocks - 1, -1, -1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append(ConvLayer(cur_channels_count, cur_channels_count // 2, kernel=1))
            cur_channels_count = cur_channels_count // 2 + self.guide_decoder

            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])

            self.denseBlocksUp.append(blk)
            prev_block_channels = blk.get_out_ch()
            cur_channels_count = prev_block_channels

        # 2x upsample
        self.final_upsample = nn.ConvTranspose2d(cur_channels_count, cur_channels_count, 3,
                                                 stride=2, padding=1)
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count + input_ch,
                                   out_channels=n_classes, kernel_size=1, stride=1,
                                   padding=0, bias=True)

    def v2_transform(self, trt=False):
        for i in range(len(self.base)):
            if isinstance(self.base[i], HarDBlock):
                blk = self.base[i]
                self.base[i] = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers)
                self.base[i].transform(blk, trt)

        for i in range(self.n_blocks):
            blk = self.denseBlocksUp[i]
            self.denseBlocksUp[i] = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers)
            self.denseBlocksUp[i].transform(blk, trt)

    def forward(self, x, guide=None):
        skip_connections = []
        size_in = x.size()
        inputs = x


        mulres_guide = {x.shape[2:]: guide}
        def cat_guide(x):
            res = x.shape[2:]
            if res in mulres_guide:
                ng = mulres_guide[res]
            else:
                ng = F.interpolate(guide, size=res, align_corners=True, mode='bilinear')
                mulres_guide[res] = ng
            return torch.cat([x, ng], dim=1)

        if self.guide_input:
            x = cat_guide(x)

        for i in range(len(self.base)):
            x = self.base[i](x)
            if self.guide_encoder and i in self.guided_layers:
                x = cat_guide(x)
            if i in self.shortcut_layers:
                skip_connections.append(x)
        out = x

        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip, True)
            out = self.conv1x1_up[i](out)
            if self.guide_decoder:
                out = cat_guide(out)
            out = self.denseBlocksUp[i](out)

        out = F.relu(self.final_upsample(out, output_size=(out.size(0), out.size(1)) + size_in[2:4]))
        out = torch.cat([inputs, out], dim=1)

        out = self.finalConv(out)
        return out


class HardNet512to1024Skip(nn.Module):
    """
    Assume input size of 512 x 512 and output of size 1024 x 1024
    """
    def __init__(self, input_ch, n_classes=19):
        super(HardNet512to1024Skip, self).__init__()

        ch_list = [32, 64, 96, 128, 160]
        grmul = 1.7
        gr = [10, 16, 18, 24, 32]
        n_layers = [4, 4, 8, 8, 8]

        blks = len(n_layers)
        self.shortcut_layers = []

        self.base = nn.ModuleList([])

        skip_connection_channel_counts = []
        ch = input_ch
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)
            self.base.append(blk)
            if i < blks - 1:
                self.shortcut_layers.append(len(self.base) - 1)

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]

            if i < blks - 1:
                self.base.append(nn.AvgPool2d(kernel_size=2, stride=2))

        cur_channels_count = ch
        prev_block_channels = ch
        n_blocks = blks - 1
        self.n_blocks = n_blocks

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up = nn.ModuleList([])

        for i in range(n_blocks - 1, -1, -1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append(ConvLayer(cur_channels_count, cur_channels_count // 2, kernel=1))
            cur_channels_count = cur_channels_count // 2

            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])

            self.denseBlocksUp.append(blk)
            prev_block_channels = blk.get_out_ch()
            cur_channels_count = prev_block_channels

        # 2x upsample
        self.final_upsample = nn.ConvTranspose2d(cur_channels_count + input_ch, cur_channels_count + input_ch, 3,
                                                 stride=2, padding=1)
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count + input_ch,
                                   out_channels=n_classes, kernel_size=1, stride=1,
                                   padding=0, bias=True)

    def v2_transform(self, trt=False):
        for i in range(len(self.base)):
            if isinstance(self.base[i], HarDBlock):
                blk = self.base[i]
                self.base[i] = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers)
                self.base[i].transform(blk, trt)

        for i in range(self.n_blocks):
            blk = self.denseBlocksUp[i]
            self.denseBlocksUp[i] = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers)
            self.denseBlocksUp[i].transform(blk, trt)

    def forward(self, x):
        skip_connections = []
        size_in = x.size()
        inputs = x

        for i in range(len(self.base)):
            x = self.base[i](x)
            if i in self.shortcut_layers:
                skip_connections.append(x)
        out = x

        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip, True)
            out = self.conv1x1_up[i](out)
            out = self.denseBlocksUp[i](out)

        out = torch.cat([inputs, out], dim=1)
        out = F.relu(self.final_upsample(out, output_size=(out.size(0), out.size(1), 1024, 1024)))

        out = self.finalConv(out)
        return out


class SimpleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, 7, stride=2)
        self.conv2 = nn.Conv2d(256, 256, 3, stride=2)
        self.conv3 = nn.Conv2d(256, 256, 3, stride=1)
        self.conv4 = nn.Conv2d(256, out_channels, 3, stride=1)

    def forward(self, x):
        size = x.size()  # N x C x H x W
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.interpolate(out, size=size[2:], mode='bilinear', align_corners=True)
        return out


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


class FCHardNet_Pretrained(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCHardNet_Pretrained, self).__init__()
        # resize_input will degrade predictions if input image is much smaller than 2048 x 1024
        net = hardnet(3, 48, final_conv=True, batch_norm=True, output_feature_map=True,
                      resize_input=False)
        this_dir = os.path.dirname(__file__)
        state_dict = convert_state_dict(
            torch.load(os.path.join(this_dir, 'hardnet70_cityscapes_model_2.pkl'))['model_state'])
        net.load_state_dict(state_dict)
        self.hardnet = net

    def forward(self, x):
        self.eval()  # Freeze batch norm.
        return self.hardnet(x).detach()


class FCHardNet_PretrainedProb(nn.Module):
    """
    Output the class distribution.
    """
    def __init__(self, in_channels, out_channels):
        super(FCHardNet_PretrainedProb, self).__init__()
        # resize_input will degrade predictions if input image is much smaller than 2048 x 1024
        net = hardnet(3, 19, final_conv=True, batch_norm=True, output_feature_map=False,
                      resize_input=False)
        this_dir = os.path.dirname(__file__)
        state_dict = convert_state_dict(
            torch.load(os.path.join(this_dir, 'hardnet70_cityscapes_model_2.pkl'))['model_state'])
        net.load_state_dict(state_dict)
        self.hardnet = net

    def forward(self, x):
        self.eval()  # Freeze batch norm.
        return torch.softmax(self.hardnet(x), dim=1).detach()


if __name__ == '__main__':
    def test_hardnet256():
        net = hardnet256(input_ch=192)
        net.to(device='cuda')
        net(torch.randn((1, 192, 256, 256), device='cuda'))

    def test_hardnet256_skip():
        net = HardNet256Skip(input_ch=192)
        net.to(device='cuda')
        out = net(torch.randn((1, 192, 256, 256), device='cuda'))
        print(out.size())

    def test_simple_feature_extractor():
        net = SimpleFeatureExtractor(in_channels=3, out_channels=48)
        net.to(device='cuda')
        out = net(torch.randn((1, 3, 376, 1241), device='cuda'))
        print(out.size())

    def test_fchardnet_pretrained():
        net = FCHardNet_Pretrained(3, 48)
        net.to(device='cuda')
        out = net(torch.randn((1, 3, 376, 1241), device='cuda'))
        print(out.size())

    def test_fchardnet_pretrained_prob():
        net = FCHardNet_PretrainedProb(3, 19)
        net.to(device='cuda')
        out = net(torch.randn((1, 3, 376, 1241), device='cuda'))
        print(out.size())

    def test_hardnet1024_skip():
        net = HardNet1024Skip(input_ch=192)
        net.to(device='cuda')
        out = net(torch.randn((1, 192, 1024, 1024), device='cuda'))
        print(out.size())

    # test_simple_feature_extractor()
    # test_fchardnet_pretrained_prob()
    # test_hardnet256_skip()
    # test_hardnet256()
    test_hardnet1024_skip()
