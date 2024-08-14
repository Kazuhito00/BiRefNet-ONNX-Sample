# BiRefNet-ONNX-Sample
DIS(Dichotomous Image Segmentation)モデルである[ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet)のPythonでのONNX推論サンプルです。<br>
変換自体を試したい方は、Google Colaboratory上で[Convert2ONNX.ipynb](Convert2ONNX.ipynb)を使用ください。<br><br>
deform_conv2dをPyTorchからONNXへ変換するために[masamitsu-murase/deform_conv2d_onnx_exporter](https://github.com/masamitsu-murase/deform_conv2d_onnx_exporter)を利用しています。<br>

![image](https://github.com/user-attachments/assets/0317d3ea-16e0-4d64-87ff-57f8f98e3930)

# Requirement 
* OpenCV 4.5.3.56 or later
* onnxruntime 1.11.0 or later  # 処理時間がかかるため、onnxruntime-gpu を推奨

# Convert
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kazuhito00/BiRefNet-ONNX-Sample/blob/main/Convert2ONNX.ipynb)<br>
Colaboratoryでノートブックを開き、上から順に実行してください。<br>

# Demo
デモの実行方法は以下です。
```bash
python demo_onnx.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --model<br>
ロードするモデルの格納パス<br>
デフォルト：model/birefnet_1024x1024.onnx
* --score_th<br>
マスク値の閾値 ※指定する場合は0.5など小数値を指定<br>
デフォルト：None

# Reference
* [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
* [masamitsu-murase/deform_conv2d_onnx_exporter](https://github.com/masamitsu-murase/deform_conv2d_onnx_exporter)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
BiRefNet-ONNX-Sample is under [MIT License](LICENSE).

# Note
サンプルの画像は[ぱくたそ](https://www.pakutaso.com/)様の「[攻撃ヘリコプターアパッチ（AH-64）](https://www.pakutaso.com/20171004291ah-64-1.html)」を使用しています。

