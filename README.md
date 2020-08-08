# image_generator_2
## 概要
メルアイコン生成器 version2で用いたソースコードです。  
詳しい解説は<a href="https://qiita.com/zassou65535/items/57fe3bd677159cdb2528">こちら</a>。

## 想定環境
python 3.7.1  
`pip install -r requirements.txt`で環境を揃えることができます。 

## プログラム
* `PGGAN_train.py`は学習を実行し、学習の過程と学習結果を出力するプログラムです。  
* `PGGAN_inference.py`は`PGGAN_train.py`が出力した学習結果をGeneratorに読み込み推論を実行、生成画像を出力するプログラムです。 

## 使い方
1. `PGGAN_train.py`のあるディレクトリに`./dataset`ディレクトリを作成します
1. `./dataset`ディレクトリに、学習に使いたい画像を`./dataset/*/*`という形式で好きな数入れます(画像のファイル形式はpng)
1. `PGGAN_train.py`の置いてあるディレクトリで`python PGGAN_train.py`を実行して学習を開始します
	* 学習の過程が`./output_img/pggan_train/`以下に出力されます
	* 学習結果が`generator_trained_model_cpu.pth`として出力されます
1. `PGGAN_inference.py`の置いてあるディレクトリで`python PGGAN_inference.py`を実行して推論します
	* 推論結果が`./output_img/pggan_inference/`以下に出力されます
	* 注意点として、`PGGAN_inference.py`の置いてあるディレクトリに`generator_trained_model_cpu.pth`がなければエラーとなります

学習には環境によっては12時間以上要する場合があります。    
入力された画像は256×256にリサイズされた上で学習に使われます。出力画像も256×256です。 
