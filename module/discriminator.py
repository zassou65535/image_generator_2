#encoding:utf-8
from .importer import *

class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		self.minbatch_std = MiniBatchStd()
		# conv modules & toRGBs
		scale = 1
		inchs = np.array([256,128, 64,32,16, 8], dtype=np.uint32)*scale
		outchs  = np.array([512,256,128,64,32,16], dtype=np.uint32)*scale
		sizes = np.array([1,4,8,16,32,64], dtype=np.uint32)
		finals = np.array([True, False, False, False, False, False], dtype=np.bool)
		blocks, fromRGBs = [], []
		for s, inch, outch, final in zip(sizes, inchs, outchs, finals):
			fromRGBs.append(nn.Conv2d(3, inch, 1, padding=0))
			blocks.append(ConvModuleD(s, inch, outch, final=final))
		self.fromRGBs = nn.ModuleList(fromRGBs)
		self.blocks = nn.ModuleList(blocks)
	def forward(self, x, res):
		# for the highest resolution
		res = min(res, len(self.blocks))
		# get integer by floor
		eps = 1e-7
		n = max(int(res-eps), 0)
		# high resolution
		x_big = self.fromRGBs[n](x)
		x_big = self.blocks[n](x_big)
		if n==0:
			x = x_big
		else:
			# low resolution
			x_sml = F.adaptive_avg_pool2d(x, x_big.shape[2:4])
			x_sml = self.fromRGBs[n-1](x_sml)
			alpha = res - int(res-eps)
			x = (1-alpha)*x_sml + alpha*x_big
		for i in range(n):
			x = self.blocks[n-1-i](x)
		return x

