#encoding:utf-8

from .importer import *
import os.path as osp

def make_datapath_list():
	#読み込むデータセットのパス
	#例えばディレクトリ「dataset」内に直接、画像データを入れているならば
	#target_path = "./dataset/*"とします
	#画像のファイル形式はpng
	target_path = "./dataset/*/*"
	path_list = []#画像ファイルパスのリストを作り、戻り値とする
	for path in glob.glob(target_path):
		path_list.append(path)
		##読み込むパスを全部表示　必要ならコメントアウトを外す
		#print(path)
	##読み込むパスを全部表示　必要ならコメントアウトを外す
	#print(str(len(path_list)))
	return path_list

class ImageTransform():
	#画像の前処理クラス
	def __init__(self,resize_pixel):
		self.data_transform = transforms.Compose([
				transforms.Resize(resize_pixel),
				transforms.ToTensor()
			])
	def __call__(self,img):
		return self.data_transform(img)

class GAN_Img_Dataset(data.Dataset):
	#画像のデータセットクラス
	def __init__(self,file_list,transform):
		self.file_list = file_list
		self.transform = transform
	#画像の枚数を返す
	def __len__(self):
		return len(self.file_list)
	#前処理済み画像の、Tensor形式のデータを取得
	def __getitem__(self,index):
		img_path = self.file_list[index]
		img = Image.open(img_path)#[RGB][高さ][幅]
		img = img.convert('RGB')#pngをjpg形式に変換
		img_transformed = self.transform(img)
		return img_transformed

#動作確認
# train_img_list = make_datapath_list()

# mean = (0.5,)
# std = (0.5,)
# train_dataset = GAN_Img_Dataset(file_list=train_img_list,transform=ImageTransform(mean,std,resize_width_height_pixel=256))

# batch_size = 1
# train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

# batch_iterator = iter(train_dataloader)
# imgs = next(batch_iterator)
# print(imgs.size())

# fig = plt.figure()
# img_transformed = imgs[0].detach().numpy().transpose(1,2,0)
# plt.imshow(img_transformed)
# fig.savefig("img/img.png")




