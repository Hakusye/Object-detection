from utils.data_augumentation import *

class DataTransform():
	def __init__(self,input_size,color_mean):
		self.data_transform = {
			'train': Compose([
					ConvertFromInts(),  # intをfloat32に変換
					ToAbsoluteCoords(),  # アノテーションデータの規格化を戻す
					PhotometricDistort(),  # 画像の色調などをランダムに変化
					Expand(color_mean),  # 画像のキャンバスを広げる
					RandomSampleCrop(),  # 画像内の部分をランダムに抜き出す
					RandomMirror(),  # 画像を反転させる
					ToPercentCoords(),  # アノテーションデータを0-1に規格化
					Resize(input_size),  # 画像サイズをinput_size×input_sizeに変形
					SubtractMeans(color_mean)  # BGRの色の平均値を引き算
			]),
			'val': Compose([
					ConvertFromInts(),  # intをfloatに変換
					Resize(input_size),  # 画像サイズをinput_size×input_sizeに変形
					SubtractMeans(color_mean)  # BGRの色の平均値を引き算
			])
		}
	
	def __call__(self,img,phase,boxes,labels):#phase:train or val
		return self.data_transform[phase](img,boxes,labels)
