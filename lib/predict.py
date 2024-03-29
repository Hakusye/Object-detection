import torch
import cv2  # OpenCVライブラリ
import numpy as np

import matplotlib.pyplot as plt 
#%matplotlib inline

def ssd_predict(img_index, img_list, dataset, net=None, dataconfidence_level=0.5):
    """
    SSDで予測させる関数。

    Parameters
    ----------
    img_index:  int
        データセット内の予測対象画像のインデックス。
    img_list: list
        画像のファイルパスのリスト
    dataset: PyTorchのDataset
        画像のDataset
    net: PyTorchのNetwork
        学習させたSSDネットワーク
    dataconfidence_level: float
        予測で発見とする確信度の閾値

    Returns
    -------
    rgb_img, true_bbox, true_label_index, predict_bbox, pre_dict_label_index, scores
    """

    # rgbの画像データを取得
    image_file_path = img_list[img_index]
    img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
    height, width, channels = img.shape  # 画像のサイズを取得
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 正解のBBoxを取得
    im, gt = dataset.__getitem__(img_index)
    true_bbox = gt[:, 0:4] * [width, height, width, height]
    true_label_index = gt[:, 4].astype(int)

    # SSDで予測
    net.eval()  # ネットワークを推論モードへ
    x = im.unsqueeze(0)  # ミニバッチ化：torch.Size([1, 3, 300, 300])
    detections = net(x)
    # detectionsの形は、torch.Size([1, 21, 200, 5])  ※200はtop_kの値

    # confidence_levelが基準以上を取り出す
    predict_bbox = []
    pre_dict_label_index = []
    scores = []
    detections = detections.cpu().detach().numpy()

    # 条件以上の値を抽出
    find_index = np.where(detections[:, 0:, :, 0] >= dataconfidence_level)
    detections = detections[find_index]
    for i in range(len(find_index[1])):  # 抽出した物体数分ループを回す
        if (find_index[1][i]) > 0:  # 背景クラスでないもの
            sc = detections[i][0]  # 確信度
            bbox = detections[i][1:] * [width, height, width, height]
            lable_ind = find_index[1][i]-1  # find_indexはミニバッチ数、クラス、topのtuple
            # （注釈）
            # 背景クラスが0なので1を引く

            # 返り値のリストに追加
            predict_bbox.append(bbox)
            pre_dict_label_index.append(lable_ind)
            scores.append(sc)

    return rgb_img, true_bbox, true_label_index, predict_bbox, pre_dict_label_index, scores

def vis_bbox(rgb_img, bbox, label_index, scores, label_names):
    """
    物体検出の予測結果を画像で表示させる関数。

    Parameters
    ----------
    rgb_img:rgbの画像
        対象の画像データ
    bbox: list
        物体のBBoxのリスト
    label_index: list
        物体のラベルへのインデックス
    scores: list
        物体の確信度。
    label_names: list
        ラベル名の配列

    Returns
    -------
    なし。rgb_imgに物体検出結果が加わった画像が表示される。
    """

    # 枠の色の設定
    num_classes = len(label_names)  # クラス数（背景のぞく）
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    # 画像の表示
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_img)
    currentAxis = plt.gca()

    # BBox分のループ
    for i, bb in enumerate(bbox):

        # ラベル名
        label_name = label_names[label_index[i]]
        color = colors[label_index[i]]  # クラスごとに別の色の枠を与える

        # 枠につけるラベル　例：person;0.72　
        if scores is not None:
            sc = scores[i]
            display_txt = '%s: %.2f' % (label_name, sc)
        else:
            display_txt = '%s: ans' % (label_name)

        # 枠の座標
        xy = (bb[0], bb[1])
        width = bb[2] - bb[0]
        height = bb[3] - bb[1]

        # 長方形を描画する
        currentAxis.add_patch(plt.Rectangle(
            xy, width, height, fill=False, edgecolor=color, linewidth=2))

        # 長方形の枠の左上にラベルを描画する
        currentAxis.text(xy[0], xy[1], display_txt, bbox={
                         'facecolor': color, 'alpha': 0.5})

class SSDPredictShow():
    """SSDでの予測と画像の表示をまとめて行うクラス"""

    def __init__(self, img_list, dataset,  eval_categories, net=None, dataconfidence_level=0.6):
        self.img_list = img_list
        self.dataset = dataset
        self.net = net
        self.dataconfidence_level = dataconfidence_level
        self.eval_categories = eval_categories

    def show(self, img_index, predict_or_ans):
        """
        物体検出の予測と表示をする関数。

        Parameters
        ----------
        img_index:  int
            データセット内の予測対象画像のインデックス。
        predict_or_ans: text
            'precit'もしくは'ans'でBBoxの予測と正解のどちらを表示させるか指定する

        Returns
        -------
        なし。rgb_imgに物体検出結果が加わった画像が表示される。
        """
        rgb_img, true_bbox, true_label_index, predict_bbox, pre_dict_label_index, scores = ssd_predict(img_index, self.img_list,
                                                                 self.dataset,
                                                                 self.net,
                                                                 self.dataconfidence_level)

        if predict_or_ans == "predict":
            vis_bbox(rgb_img, bbox=predict_bbox, label_index=pre_dict_label_index,
                     scores=scores, label_names=self.eval_categories)

        elif predict_or_ans == "ans":
            vis_bbox(rgb_img, bbox=true_bbox, label_index=true_label_index,
                     scores=None, label_names=self.eval_categories)


