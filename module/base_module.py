#encoding:utf-8

from .importer import *

#Pixel normalizationをする関数
#ピクセルごとに特徴量を正規化する
#チャネル方向に2乗平均をとってsqrtをした値で割り算をする
#generatorのみに使われる
class PixelNorm(nn.Module):
	def forward(self, x):
		#0除算を防ぐために十分小さい数epsを用意する
		eps = 1e-7
		#チャネル方向(dim=1)に2乗(x**2)平均をとる
		mean = torch.mean(x**2,dim=1,keepdims=True)
		return x / (torch.sqrt(mean)+eps)

#学習率の平滑化をする関数
#各レイヤの重みを入力チャネルサイズで正規化する。Heの初期化と似た効果を期待するもの。
class EqualizeLearningRate(nn.Module):
	def forward(self, x, gain=2):
		scale = (gain/x.shape[1])**0.5
		return x * scale

#入力されたテンソルに対して、「各バッチの標準偏差」を求め、
#チャネルと縦横方向に平均化したテンソルを作る関数
#「各バッチの標準偏差」 = そのミニバッチ内の画像がどれだけ多様かを表す
#discriminatorは本物か偽物かを判断するときに
#ミニバッチ内の画像そのものだけでなく、
#ミニバッチ内の画像から得られる標準偏差も考慮に入れて判別に使う
#これがあまりにも小さすぎるならばdiscriminatorは画像を偽物と判断できる
#これによりモード崩壊を防止できる
class MiniBatchStd(nn.Module):
	def forward(self, x):
		std = torch.std(x, dim=0, keepdim=True)
		mean = torch.mean(std, dim=(1,2,3), keepdim=True)
		n,c,h,w = x.shape
		mean = torch.ones(n,1,h,w, dtype=x.dtype, device=x.device)*mean
		return torch.cat((x,mean), dim=1)

#畳み込み層のモジュール「Conv2d」
#処理を1まとめにして扱いやすくしておく
#層の途中にあるReflectionPad2dはzero paddingと似た役割をするが
#zero paddingと比べて元の入力に近い分布を実現できるため
#生成された画像の端付近にアーティファクトができにくくなる
class Conv2d(nn.Module):
	'''
	引数:
		inch: (int)  入力チャネル数
		outch: (int) 出力チャネル数
		kernel_size: (int) カーネルの大きさ
		padding: (int) パディング
	'''
	def __init__(self, inch, outch, kernel_size, padding=0):
		super().__init__()
		self.layers = nn.Sequential(
			EqualizeLearningRate(),
			nn.ReflectionPad2d(padding),
			nn.Conv2d(inch, outch, kernel_size, padding=0),
			PixelNorm(),
		)
		nn.init.kaiming_normal_(self.layers[2].weight)
	def forward(self, x):
		return self.layers(x)

#generator用畳み込み層
#generatorの最初の層のみUpsampleなし
class ConvModuleG(nn.Module):
	def __init__(self, out_size, inch, outch, first=False):
		super().__init__()
		if first:
			layers = [
				Conv2d(inch, outch, 3, padding=1),
				nn.LeakyReLU(0.2, inplace=False),
				Conv2d(outch, outch, 3, padding=1),
				nn.LeakyReLU(0.2, inplace=False),
			]
		else:
			layers = [
				nn.Upsample((out_size, out_size), mode='nearest'),
				Conv2d(inch, outch, 3, padding=1),
				nn.LeakyReLU(0.2, inplace=False),
				Conv2d(outch, outch, 3, padding=1),
				nn.LeakyReLU(0.2, inplace=False),
			]
		self.layers = nn.Sequential(*layers)
	def forward(self, x):
		return self.layers(x)

#discriminator用の畳み込み層
#discriminatorの最後の層のみMiniBatchStd（モード崩壊防止用モジュール）あり
class ConvModuleD(nn.Module):
	def __init__(self, out_size, inch, outch, final=False):
		super().__init__()
		if final:
			layers = [
				MiniBatchStd(),
				Conv2d(inch+1, outch, 3, padding=1),
				nn.LeakyReLU(0.2, inplace=False),
				Conv2d(outch, outch, 4, padding=0), 
				nn.LeakyReLU(0.2, inplace=False),
				nn.Conv2d(outch, 1, 1, padding=0), 
			]
		else:
			layers = [
				Conv2d(inch, outch, 3, padding=1),
				nn.LeakyReLU(0.2, inplace=False),
				Conv2d(outch, outch, 3, padding=1),
				nn.LeakyReLU(0.2, inplace=False),
				nn.AdaptiveAvgPool2d((out_size, out_size)),
			]
		self.layers = nn.Sequential(*layers)
	def forward(self, x):
		return self.layers(x)


