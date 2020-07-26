#encoding:utf-8

from module.importer import *
from module.base_module import *
from module.discriminator import *
from module.generator import *

#学習と誤差伝搬を行う関数
def gradient_penalty(netD, real, fake, res, batch_size, gamma=1):
	device = real.device
	#requires_gradが有効なTensorに対してはbackwardメソッドが呼べて、自動的に微分を計算できる
	alpha = torch.rand(batch_size, 1, 1, 1, requires_grad=True).to(device)
	#学習時、損失関数にはWGAN-GPを用いる
	x = alpha*real + (1-alpha)*fake
	#推論を実行、結果をd_とする
	d_ = netD.forward(x, res)
	g = torch.autograd.grad(outputs=d_, inputs=x,
							grad_outputs=torch.ones(d_.shape).to(device),
							create_graph=True, retain_graph=True,only_inputs=True)[0]
	g = g.reshape(batch_size, -1)
	return ((g.norm(2,dim=1)/gamma-1.0)**2).mean()

f __name__ == '__main__':

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	netG = models.Generator().to(device)
	netD = models.Discriminator().to(device)
	netG_mavg = models.Generator().to(device) # moving average
	#generator,discriminatorの誤差伝搬の最適化手法にはAdamを指定
	optG = torch.optim.Adam(netG.parameters(), lr=0.0005, betas=(0.0, 0.99))
	optD = torch.optim.Adam(netD.parameters(), lr=0.0005, betas=(0.0, 0.99))
	criterion = torch.nn.BCELoss()

	# dataset
	transform = transforms.Compose([transforms.CenterCrop(160),
	                                transforms.Resize((128,128)),
	                                transforms.ToTensor(), ])

	trainset = datasets.CelebA('~/data', download=True, split='train',
	                           transform=transform)

	bs = 8
	train_loader = DataLoader(trainset, batch_size=bs, shuffle=True)

	#学習開始
	#エポック数
	nepoch = 10
	losses = []
	res_step = 15000
	j = 0
	# constant random inputs
	z0 = torch.randn(16, 512*16).to(device)
	#z0はclampを用いて値の下限を-1、上限を1にしておく
	z0 = torch.clamp(z0, -1.,1.)
	for iepoch in range(nepoch):
		if j==res_step*6.5:
			optG.param_groups[0]['lr'] = 0.0001
			optD.param_groups[0]['lr'] = 0.0001

		for i, data in enumerate(train_loader):
			x, y = data
			x = x.to(device)
			res = j/res_step

			#generaorの学習
			#ノイズを生成
			z = torch.randn(bs, 512*16).to(x.device)
			#ノイズをgeneratorに入力、出力画像をx_とする
			x_ = netG.forward(z, res)
			#出力画像x_をdiscriminatorで推論　つまり偽画像の入力をする
			d_ = netD.forward(x_, res)
			# WGAN_GPではミニバッチ内の推論結果全てに対し平均を取り、それを誤差伝搬に使う
			lossG = -d_.mean()

			optG.zero_grad()
			lossG.backward()
			optG.step()

			# update netG_mavg by moving average
			momentum = 0.995 # remain momentum
			alpha = min(1.0-(1/(j+1)), momentum)
			for p_mavg, p in zip(netG_mavg.parameters(), netG.parameters()):
				p_mavg.data = alpha*p_mavg.data + (1.0-alpha)*p.data

			#discriminatorの学習
			z = torch.randn(x.shape[0], 512*16).to(x.device)
			x_ = netG.forward(z, res)
			x = F.adaptive_avg_pool2d(x, x_.shape[2:4])
			d = netD.forward(x, res)   # real
			d_ = netD.forward(x_, res) # fake
			loss_real = -d.mean()
			loss_fake = d_.mean()
			loss_gp = gradient_penalty(netD, x.data, x_.data, res, x.shape[0])
			loss_drift = (d**2).mean()

			beta_gp = 10.0
			beta_drift = 0.001
			lossD = loss_real + loss_fake + beta_gp*loss_gp + beta_drift*loss_drift

			optD.zero_grad()
			lossD.backward()
			optD.step()

			print('ep: %02d %04d %04d lossG=%.10f lossD=%.10f' %
				(iepoch, i, j, lossG.item(), lossD.item()))

			losses.append([lossG.item(), lossD.item()])
			j += 1

			if j%500==0:
				netG_mavg.eval()
				z = torch.randn(16, 512*16).to(x.device)
				x_0 = netG_mavg.forward(z0, res)
				x_ = netG_mavg.forward(z, res)

				dst = torch.cat((x_0, x_), dim=0)
				dst = F.interpolate(dst, (128, 128), mode='nearest')
				dst = dst.to('cpu').detach().numpy()
				n, c, h, w = dst.shape
				dst = dst.reshape(4,8,c,h,w)
				dst = dst.transpose(0,3,1,4,2)
				dst = dst.reshape(4*h,8*w,3)
				dst = np.clip(dst*255., 0, 255).astype(np.uint8)
				skio.imsave('out/img_%03d_%05d.png' % (iepoch, j), dst)

				losses_ = np.array(losses)
				niter = losses_.shape[0]//100*100
				x_iter = np.arange(100)*(niter//100) + niter//200
				plt.plot(x_iter, losses_[:niter,0].reshape(100,-1).mean(1))
				plt.plot(x_iter, losses_[:niter,1].reshape(100,-1).mean(1))
				plt.tight_layout()
				plt.savefig('out/loss_%03d_%05d.png' % (iepoch, j))
				plt.clf()

				netG_mavg.train()

			if j >= res_step*7:
				break

			if j%100==0:
				coolGPU()

		if j >= res_step*7:
			break


