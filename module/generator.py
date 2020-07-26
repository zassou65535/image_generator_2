#encoding:utf-8

from .importer import *
from .base_module import *

class Generator(nn.Module):
	def __init__(self):
		super().__init__()
		#畳み込みモジュールの設定を1つずつしていく
		scale = 1
		inchs  = np.array([512,256,128,64,32, 16], dtype=np.uint32)*scale
		outchs = np.array([256,128, 64,32,16,  8], dtype=np.uint32)*scale
		sizes  = np.array([  4,  8, 16,32,64,128], dtype=np.uint32)
		#最初の層のみ、それを示すフラグをTrueにしておく
		firsts = np.array([True, False, False, False, False, False], dtype=np.bool)
		#blockには畳み込み層を格納、toRGBsは入力されたデータを出力画像(RGB3チャネル)に変換するための層を格納
		blocks, toRGBs = [], []
		for s, inch, outch, first in zip(sizes, inchs, outchs, firsts):
			blocks.append(ConvModuleG(s, inch, outch, first))
			toRGBs.append(nn.Conv2d(outch, 3, 1, padding=0))
		self.blocks = nn.ModuleList(blocks)
		self.toRGBs = nn.ModuleList(toRGBs)
	def forward(self, x, res, eps=1e-7):
		# to image
		n,c = x.shape
		x = x.reshape(n,c//16,4,4)
		# for the highest resolution
		res = min(res, len(self.blocks))
		# get integer by floor
		nlayer = max(int(res-eps), 0)
		for i in range(nlayer):
			x = self.blocks[i](x)
		# high resolution
		x_big = self.blocks[nlayer](x)
		dst_big = self.toRGBs[nlayer](x_big)
		if nlayer==0:
			x = dst_big
		else:
			# low resolution
			x_sml = F.interpolate(x, x_big.shape[2:4], mode='nearest')
			dst_sml = self.toRGBs[nlayer-1](x_sml)
			alpha = res - int(res-eps)
			x = (1-alpha)*dst_sml + alpha*dst_big
		#return x, n, res
		return torch.sigmoid(x)








