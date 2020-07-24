#encoding:utf-8

from module.importer import *
from module.dataloader import *
from module.discriminator import *
from module.generator import *

pixel_size = 64#画像のピクセルサイズ

#ネットワークを初期化
def weights_init(m):
	classname = m.__class__.__name__
	if(classname.find('Conv')!=-1):
		#Conv2dとConvTranspose2dの初期化
		nn.init.normal_(m.weight.data,0.0,0.02)
		nn.init.constant_(m.bias.data,0)
	elif(classname.find("BatchNorm")!=-1):
		#BatchNorm2dの初期化
		nn.init.normal_(m.weight.data,1.0,0.02)
		nn.init.constant_(m.bias.data,0)

#初期化の実施
G = Generator(z_dim=20,image_size=pixel_size)
D = Discriminator(z_dim=20,image_size=pixel_size)
G.apply(weights_init)
D.apply(weights_init)
print("initalized networks")

#モデルを学習させる関数
def train_model(G,D,dataloader,num_epochs):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("device:",device)
	#最適化手法の設定
	g_lr,d_lr = 0.0001,0.0004
	beta1,beta2 = 0.0,0.9
	g_optimizer = torch.optim.Adam(G.parameters(),g_lr,[beta1,beta2])
	d_optimizer = torch.optim.Adam(D.parameters(),d_lr,[beta1,beta2])
	#誤差関数の定義
	#criterion = nn.BCEWithLogitsLoss(reduction="mean")
	#パラメータ類
	z_dim = 20
	mini_batch_size = 5
	#ネットワークをデバイスに移動
	G.to(device)
	D.to(device)
	G.train()#学習モードに設定
	D.train()
	#ネットワークがある程度一定ならば高速化
	torch.backends.cudnn.benchmark = True
	#画像の枚数
	num_train_imgs = len(dataloader.dataset)
	batch_size = dataloader.batch_size
	#イテレーションカウンタをセット
	iteration = 1
	logs = []
	#epochのループ
	for epoch in range(num_epochs):
		#開始時刻を保存
		t_epoch_start = time.time()
		epoch_g_loss = 0.0#epoch損失和
		epoch_d_loss = 0.0#epoch損失和
		print("--------------------")
		print("Epoch {}/{}".format(epoch,num_epochs))
		print("--------------------")
		print("(train)")
		#データローダーからminibatchずつ取り出す
		for imgs in dataloader:
			# print(imgs.size())
			# print(dataloader)
			# print(iteration)
			#-------------------------
			#discriminatorの学習
			#-------------------------
			if(imgs.size()[0]==1): continue#ミニバッチサイズ1だと正規化でエラーになるので避ける
			#GPUが使えるならGPUへ転送
			imgs = imgs.to(device)
			#正解ラベル、偽ラベルを作成
			#epochの最後のイテレーションはミニバッチの数が少なくなる
			mini_batch_size = imgs.size()[0]
			# label_real = torch.full((mini_batch_size,),1).to(device)
			# label_fake = torch.full((mini_batch_size,),0).to(device)
			#真の画像を判定
			d_out_real,_,_ = D(imgs)
			#偽の画像を生成して判定
			input_z = torch.randn(mini_batch_size,z_dim).to(device)
			input_z = input_z.view(input_z.size(0),input_z.size(1),1,1)
			fake_images,_,_ = G(input_z)
			d_out_fake,_,_ = D(fake_images)
			#誤差の計算
			# d_loss_real = criterion(d_out_real.view(-1),label_real)
			# d_loss_fake = criterion(d_out_fake.view(-1),label_fake)
			d_loss_real = torch.nn.ReLU()(1.0-d_out_real).mean()
			d_loss_fake = torch.nn.ReLU()(1.0+d_out_fake).mean()
			d_loss = d_loss_real + d_loss_fake
			#誤差を伝搬
			g_optimizer.zero_grad()
			d_optimizer.zero_grad()
			d_loss.backward()
			d_optimizer.step()

			#-------------------------
			#generatorの学習
			#-------------------------
			#偽の画像を生成して判定
			input_z = torch.randn(mini_batch_size,z_dim).to(device)
			input_z = input_z.view(input_z.size(0),input_z.size(1),1,1)
			fake_images,_,_ = G(input_z)
			d_out_fake,_,_ = D(fake_images)
			#誤差の計算
			#g_loss = criterion(d_out_fake.view(-1),label_real)
			g_loss =- d_out_fake.mean()
			#誤差を伝搬
			g_optimizer.zero_grad()
			d_optimizer.zero_grad()
			g_loss.backward()
			g_optimizer.step()

			#-------------------------
			#記録
			#-------------------------
			epoch_d_loss += d_loss.item()
			epoch_g_loss += g_loss.item()
			iteration += 1

		#epochのphaseごとのlossと正解率
		t_epoch_finish = time.time()
		print("--------------------")
		print("epoch {} || Epoch_D_Loss:{:.4f} ||Epoch_G_Loss:{:.4f}".format(
				epoch,epoch_d_loss/batch_size,epoch_g_loss/batch_size))
		print("timer: {:.4f} sec.".format(t_epoch_finish - t_epoch_start))
		t_epoch_start = time.time()
	return G,D

#訓練データの読み込み、データセット作成
train_img_list = make_datapath_list()
mean = (0.5,)
std = (0.5,)
train_dataset = GAN_Img_Dataset(file_list=train_img_list,transform=ImageTransform(mean,std,resize_width_height_pixel=pixel_size))
# for i in range(0,len(train_dataset)):
# 	print(str(i))
# 	print(train_dataset[i].size())
# print(":::::::::::::::::::::::::::::::::::::")
#データローダー作成
batch_size = 5
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

#epoch数指定
num_epochs = 3000;
#モデルを学習させる
G_update,D_update = train_model(G,D,dataloader=train_dataloader,num_epochs=num_epochs)

#生成された画像、訓練データを可視化する
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#入力乱数の生成
z_dim = 20
fixed_z = torch.randn(batch_size,z_dim)
fixed_z = fixed_z.view(fixed_z.size(0),fixed_z.size(1),1,1)
#画像生成
fake_images,am1,am2 = G_update(fixed_z.to(device))
#訓練データ
batch_iterator = iter(train_dataloader)#イテレータに変換
imges = next(batch_iterator)#1番目の要素を取り出す

#出力
fig = plt.figure(figsize=(15,6))
for i in range(0,5):
	#上段に訓練データを配置
	plt.subplot(2,5,i+1)
	plt.imshow(imges[i].cpu().detach().numpy().transpose(1,2,0))
	#下段に訓練データを配置
	plt.subplot(2,5,5+i+1)
	plt.imshow(fake_images[i].cpu().detach().numpy().transpose(1,2,0))
fig.savefig("img/img.png")

#もっと生成
generate_number = 15#(5*generate_number)枚追加で生成する
for i in range(0,generate_number):
	fig = plt.figure(figsize=(15,6))
	fixed_z = torch.randn(batch_size,z_dim)
	fixed_z = fixed_z.view(fixed_z.size(0),fixed_z.size(1),1,1)
	generated_images,am1,am2 = G_update(fixed_z.to(device))
	for k in range(0,5):
		plt.subplot(2,5,k+1)
		plt.imshow(generated_images[k].cpu().detach().numpy().transpose(1,2,0))
	fig.savefig("img/generated_{}.png".format(str(i+1)))



