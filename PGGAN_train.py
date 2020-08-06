#encoding:utf-8

from module.importer import *
from module.base_module import *
from module.discriminator import *
from module.generator import *
from module.dataloader import *

#discriminatorの損失関数の、勾配制約項の計算に必要な関数「gradient_penalty」
#損失関数にはWGAN-GPを用いる
#discriminatorでは、E[本物の画像の判定結果]-E[偽画像の判定結果]+勾配制約項　と表され、
#generatorでは、E[偽画像の判定結果]と表される
def gradient_penalty(netD, real, fake, res, batch_size, gamma=1):
	device = real.device
	#requires_gradが有効なTensorに対してはbackwardメソッドが呼べて、自動的に微分を計算できる
	alpha = torch.rand(batch_size, 1, 1, 1, requires_grad=True).to(device)
	#本物画像と偽画像を任意の割合で混ぜ合わせる
	x = alpha*real + (1-alpha)*fake
	#それをdiscriminatorに入れ、結果をd_とする
	d_ = netD.forward(x, res)
	#出力d_と入力xから傾きを求める
	#傾きから計算されるL2ノルムが1になると良い結果を生むことが知られている
	#よってこれが1に近づくような学習ができるようにgradient_penaltyを計算
	g = torch.autograd.grad(outputs=d_, inputs=x,
							grad_outputs=torch.ones(d_.shape).to(device),
							create_graph=True, retain_graph=True,only_inputs=True)[0]
	g = g.reshape(batch_size, -1)
	return ((g.norm(2,dim=1)/gamma-1.0)**2).mean()

if __name__ == '__main__':
	#GPUが使えるならGPUを使用
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print("device : "+device)

	netG = Generator().to(device)
	netD = Discriminator().to(device)
	netG_mavg = Generator().to(device) # moving average
	#generator,discriminatorの誤差伝搬の最適化手法にはAdamを指定
	optG = torch.optim.Adam(netG.parameters(), lr=0.0005, betas=(0.0, 0.99))
	optD = torch.optim.Adam(netD.parameters(), lr=0.0005, betas=(0.0, 0.99))
	#誤差関数の定義
	criterion = torch.nn.BCELoss()

	# dataset
	# transform = transforms.Compose([transforms.CenterCrop(160),
	#                                 transforms.Resize((128,128)),
	#                                 transforms.ToTensor(), ])

	# trainset = datasets.CelebA('~/data', download=True, split='train',
	#                            transform=transform)

	# bs = 8
	# train_loader = DataLoader(trainset, batch_size=bs, shuffle=True)

	#訓練データの読み込み、データセット作成
	train_img_list = make_datapath_list()
	train_dataset = GAN_Img_Dataset(file_list=train_img_list,transform=ImageTransform(resize_pixel=256))
	#データローダー作成
	batch_size = 8#ミニバッチあたりのサイズ
	train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

	#学習開始
	#lossesはグラフの出力のための記録用変数　学習には使わない
	losses = []
	#res_step回繰り返すごとに解像度を高める
	res_step = 7500
	#何回イテレーションしたかをiterationとする
	iteration = 0
	#generatorに入力する、動作検証用のノイズ　学習の様子を見る用
	z0 = torch.randn(16, 512*16).to(device)
	#z0はclampを用いて値の下限を-1、上限を1にしておく
	z0 = torch.clamp(z0, -1.,1.)
	#学習は合計res_step*8回行う
	#res_step回繰り返すごとに解像度が高まっていく
	while(iteration<res_step*8):
		#学習が終わりに近づいてきたら学習率を下げる
		if iteration==res_step*7.5:
			optG.param_groups[0]['lr'] = 0.0001
			optD.param_groups[0]['lr'] = 0.0001

		#for i, data in enumerate(train_loader):
		#データローダーからminibatchずつ取り出す
		for imgs in train_dataloader:
			x = imgs
			#取り出したミニバッチ数が1の場合勾配を求める過程でエラーとなるので処理を飛ばす
			if(x.shape[0]==1): continue
			#GPUが使えるならGPUへ転送
			x = x.to(device)
			#どれだけ学習が進んだかを表す変数「res」
			#generator,discriminatorの解像度の切り替えに用いる
			res = iteration/res_step

			#----------generaorの学習----------
			#ノイズを生成
			z = torch.randn(batch_size, 512*16).to(x.device)
			#ノイズをgeneratorに入力、出力画像をx_とする
			x_ = netG.forward(z, res)
			#出力画像x_をdiscriminatorで推論　つまり偽画像の入力をする
			d_ = netD.forward(x_, res)
			#損失関数にはWGAN-GPを用いる
			#discriminatorでは、E[本物の画像の判定結果]-E[偽画像の判定結果]+勾配制約項　と表され、
			#generatorでは、E[偽画像の判定結果]と表される

			# WGAN_GPではミニバッチ内の推論結果全てに対し平均を取り、それを誤差伝搬に使う
			lossG = -d_.mean()#E[偽画像の判定結果]を計算

			#前のイテレーションで計算した傾きが残ってしまっているのでそれをリセットしておく
			optG.zero_grad()
			#損失の傾きを計算して
			lossG.backward()
			#実際に誤差伝搬を行う
			optG.step()

			# update netG_mavg by moving average
			momentum = 0.995 # remain momentum
			alpha = min(1.0-(1/(iteration+1)), momentum)
			for p_mavg, p in zip(netG_mavg.parameters(), netG.parameters()):
				p_mavg.data = alpha*p_mavg.data + (1.0-alpha)*p.data

			#----------discriminatorの学習----------
			#ノイズを生成、zとする
			z = torch.randn(x.shape[0], 512*16).to(x.device)
			#generatorにノイズを入れ偽画像を生成、x_とする
			x_ = netG.forward(z, res)
			#平均を取ることでxの次元を変換
			x = F.adaptive_avg_pool2d(x, x_.shape[2:4])
			#本物の画像を判定、結果をdに格納
			d = netD.forward(x, res)
			#偽画像を判定、結果をd_に格納
			d_ = netD.forward(x_, res)

			#損失関数にはWGAN-GPを用いる
			#discriminatorでは、E[本物の画像の判定結果]-E[偽画像の判定結果]+勾配制約項　と表され、
			#generatorでは、E[偽画像の判定結果]と表される

			#ミニバッチごとの、判定結果の平均をそれぞれとる
			loss_real = -d.mean()#E[本物の画像の判定結果]を計算
			loss_fake = d_.mean()#-E[偽画像の判定結果]を計算
			#勾配制約項の計算
			loss_gp = gradient_penalty(netD, x.data, x_.data, res, x.shape[0])
			loss_drift = (d**2).mean()
			beta_gp = 10.0
			beta_drift = 0.001
			#E[本物の画像の判定結果]-E[偽画像の判定結果]+勾配制約項 を計算
			lossD = loss_real + loss_fake + beta_gp*loss_gp + beta_drift*loss_drift

			#前のイテレーションで計算した傾きが残ってしまっているのでそれをリセットしておく
			optD.zero_grad()
			#損失の傾きを計算して
			lossD.backward()
			#実際に誤差伝搬を行う
			optD.step()

			print('floor(res)=%02d iteration=%06d lossG=%.10f lossD=%.10f' %
				(max(int(res-1e-7),0), iteration, lossG.item(), lossD.item()))
			#ログ取る用
			losses.append([lossG.item(), lossD.item()])
			#イテレーションをカウント
			iteration += 1

			if (iteration%500==0 or iteration==res_step*8):
				#画像の出力を行う
				netG_mavg.eval()
				#ノイズを入力して16枚画像を生成
				z = torch.randn(16, 512*16).to(x.device)
				x_0 = netG_mavg.forward(z0, res)
				x_ = netG_mavg.forward(z, res)

				dst = torch.cat((x_0, x_), dim=0)
				dst = F.interpolate(dst, (256,256), mode='nearest')
				dst = dst.to('cpu').detach().numpy()
				#それぞれ生成された画像の枚数、チャネル数、高さpixel、幅pixel
				num_picture, channel, height, width = dst.shape
				dst = np.clip(dst*255., 0, 255).astype(np.uint8)

				output_fig = plt.figure(figsize=(25.6,19.2))
				for i in range(0,num_picture):
					#出力結果を順に配置
					#plt.subplot(行数、列数、画像の番号)という形式で指定する
					plt.subplot(4,8,i+1)
					#dst[i]はこの時点で次元が[channel,height,width]となっているが、
					#画像として表示するにはtranspose(1,2,0)とすることで
					#[height,width,channel]に変換する必要がある
					plt.imshow(dst[i].transpose(1,2,0))
				plt.subplots_adjust(wspace=0.8)#出力時の生成画像同士の余白を調整
				#画像出力用にディレクトリを作成
				os.makedirs("output_img/pggan_train",exist_ok=True)
				#保存の実行
				output_fig.savefig('output_img/pggan_train/img_%d_%06d.jpg' % (max(int(res-1e-7),0),iteration),dpi=300)

				plt.clf()

				plt.close()#作成したグラフがメモリに残り続けるのを防ぐ

				netG_mavg.train()

			#学習が終わるとループを抜けるが、その際にlossのグラフを出力する
			#学習済みモデルの出力も行う
			if iteration == res_step*8:
				#lossのグラフを出力する
				losses_ = np.array(losses)
				niter = losses_.shape[0]//100*100
				x_iter = np.arange(100)*(niter//100) + niter//200
				plt.plot(x_iter, losses_[:niter,0].reshape(100,-1).mean(1))
				plt.plot(x_iter, losses_[:niter,1].reshape(100,-1).mean(1))
				plt.tight_layout()
				#画像出力用にディレクトリを作成
				os.makedirs("output_img/pggan_train",exist_ok=True)
				#そこへ保存
				plt.savefig('output_img/pggan_train/loss.jpg',dpi=70)
				plt.clf()
				#学習済みモデル(generator「netG_mavg」)の出力を行う
				#学習済みモデル（CPU向け）を出力
				torch.save(netG_mavg.to('cpu').state_dict(),'generator_trained_model_cpu.pth')
				break


