#encoding:utf-8

from module.importer import *
from module.base_module import *
from module.generator import *

generator = Generator()
generator.load_state_dict(torch.load('generator_trained_model_cpu.pth'))
#推論モードに切り替え
generator.eval()
#ノイズを入力して16枚画像を生成
z = torch.randn(16, 512*16)
picture = generator.forward(z,8)
picture = picture.detach().numpy()
picture = np.clip(picture*255., 0, 255).astype(np.uint8)
#画像を出力
for i in range(0,picture.shape[0]):
	output_fig = plt.figure()
	#dst[i]はこの時点で次元が[channel,height,width]となっているが、
	#画像として表示するにはtranspose(1,2,0)とすることで
	#[height,width,channel]に変換する必要がある
	plt.imshow(picture[i].transpose(1,2,0))
	#画像出力用にディレクトリを作成
	os.makedirs("output_img/pggan_inference",exist_ok=True)
	#そこへ保存
	output_fig.savefig('output_img/pggan_inference/img_%d.jpg' % (i+1),dpi=300)
	plt.close()




