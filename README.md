# Object-detection
猫と犬を検出できる（ようになる予定）
#############現状###################################
画像の型の変換やモデル作成はうまくいったが、
unit_test/val.pyで、入力の型がおかしいために画像を出力ができない。
モデル作成あたりがおかしいと時間がかかりすぎる可能性があるので
いったん置いておく。
####################################################

unit-test/train_val.pyで学習の実施
学習データはdata/modelに出力
unit-test/val.pyで物体検出画像の出力(予定)
dataにデータをできるだけ増やすこと（アノテーションも作成しておくこと）
