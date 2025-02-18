# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch import nn
from torch.nn import functional as F

import pdb

# ===================
#     RGA Module
# ===================

class RGA_Module(nn.Module):
	def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=True, \
		cha_ratio=8, spa_ratio=8, down_ratio=8):
		super(RGA_Module, self).__init__()

		self.in_channel = in_channel
		self.in_spatial = in_spatial
		
		self.use_spatial = use_spatial
		self.use_channel = use_channel

		print ('Use_Spatial_Att: {};\tUse_Channel_Att: {}.'.format(self.use_spatial, self.use_channel))

		self.inter_channel = in_channel // cha_ratio
		self.inter_spatial = in_spatial // spa_ratio
		
		# Embedding functions for original features
		if self.use_spatial:
			self.gx_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)
		if self.use_channel:
			self.gx_channel = nn.Sequential(
				nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_spatial),
				nn.ReLU()
			)
		
		# Embedding functions for relation features
		if self.use_spatial:
			self.gg_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_spatial * 2, out_channels=self.inter_spatial,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_spatial),
				nn.ReLU()
			)
		if self.use_channel:
			self.gg_channel = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel*2, out_channels=self.inter_channel,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)
		
		# Networks for learning attention weights
		if self.use_spatial:
			num_channel_s = 1 + self.inter_spatial
			self.W_spatial = nn.Sequential(
				nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s//down_ratio,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(num_channel_s//down_ratio),
				nn.ReLU(),
				nn.Conv2d(in_channels=num_channel_s//down_ratio, out_channels=1,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(1)
			)
		if self.use_channel:	
			num_channel_c = 1 + self.inter_channel
			self.W_channel = nn.Sequential(
				nn.Conv2d(in_channels=num_channel_c, out_channels=num_channel_c//down_ratio,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(num_channel_c//down_ratio),
				nn.ReLU(),
				nn.Conv2d(in_channels=num_channel_c//down_ratio, out_channels=1,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(1)
			)

		# Embedding functions for modeling relations
		if self.use_spatial:
			self.theta_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
								kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)
			self.phi_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
							kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)
		if self.use_channel:
			self.theta_channel = nn.Sequential(
				nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
								kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_spatial),
				nn.ReLU()
			)
			self.phi_channel = nn.Sequential(
				nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
							kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_spatial),
				nn.ReLU()
			)
				
	def forward(self, x):
		b, c, h, w = x.size()  # b,256,64,32
		
		if self.use_spatial:  # 是否做空间相关的特征
			# spatial attention
			theta_xs = self.theta_spatial(x)  # (b,32,64,32)
			phi_xs = self.phi_spatial(x)  # (b,32,64,32)
			theta_xs = theta_xs.view(b, self.inter_channel, -1)  # (b,32,2048)
			theta_xs = theta_xs.permute(0, 2, 1)  # (b,2048,32)
			phi_xs = phi_xs.view(b, self.inter_channel, -1)  # (b,32,2048)
			Gs = torch.matmul(theta_xs, phi_xs)  # (b,2048,2048)
			Gs_in = Gs.permute(0, 2, 1).view(b, h*w, h, w)  # (b,2048,64,32)
			Gs_out = Gs.view(b, h*w, h, w)  # (b,2048,64,32)
			Gs_joint = torch.cat((Gs_in, Gs_out), 1)  # (b,4096,64,32)
			Gs_joint = self.gg_spatial(Gs_joint)  # (b,256,64,32)
		
			g_xs = self.gx_spatial(x)  # (b,32,64,32)
			g_xs = torch.mean(g_xs, dim=1, keepdim=True)  # (b,1,64,32)
			ys = torch.cat((g_xs, Gs_joint), 1)  # (b,257,64,32)

			W_ys = self.W_spatial(ys)  # (b,1,64,32)
			if not self.use_channel:
				out = F.sigmoid(W_ys.expand_as(x)) * x
				return out
			else:
				x = F.sigmoid(W_ys.expand_as(x)) * x  # (b,256,64,32)

		if self.use_channel:  # 是否做维度相关的特征
			# channel attention
			xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)  # (b,2048,256,1)
			theta_xc = self.theta_channel(xc).squeeze(-1).permute(0, 2, 1)  # (b,256,256)
			phi_xc = self.phi_channel(xc).squeeze(-1)  # (b,256,256)
			Gc = torch.matmul(theta_xc, phi_xc)  # (b,256,256)
			Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)  # (b,256,256,1)
			Gc_out = Gc.unsqueeze(-1)  # (b,256,256,1)
			Gc_joint = torch.cat((Gc_in, Gc_out), 1)  # (b,512,256,1)
			Gc_joint = self.gg_channel(Gc_joint)  # (b,32,256,1)

			g_xc = self.gx_channel(xc)  # (b,256,256,1)
			g_xc = torch.mean(g_xc, dim=1, keepdim=True)  # (b,1,256,1)
			yc = torch.cat((g_xc, Gc_joint), 1)  # (b,33,256,1)

			W_yc = self.W_channel(yc).transpose(1, 2)  # (b,256,1,1)
			out = F.sigmoid(W_yc) * x  # (b,256,64,32)

			return out