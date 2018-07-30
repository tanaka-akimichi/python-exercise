import cupy as cp
import numpy as np
import timeit
import warnings
warnings.filterwarnings('ignore')

from skimage import data,transform,color,io

A = cp.arange(9).reshape(3, 3).astype('f')  # cupy上で3*3の行列を生成
B = cp.arange(9).reshape(3, 3).astype('f')  # cupy上で3*3の行列を生成
print('A = \n', A)
print('B = \n', B)

C = A + B  # 行列の和
print('和：A + B = \n', C)

D = cp.dot(A, B)  # 行列の積
print('積：A・B = \n', D)

# 画像のロード
np_img = data.coffee()  # コーヒーカップ画像をロード

np_img = transform.resize(np_img, (4096,4096))  # 4096*4096にリサイズ
np_img = color.rgb2gray(np_img)  # グレースケール化
np_img = np_img.astype('f')

io.imshow(np_img)  # 表示
io.show()


# フーリエ変換
cp_img = cp.asarray(np_img)  # numpy配列 ⇒ cupy配列に変換
cp_fimg = cp.fft.fft2(cp_img)  # 【フーリエ変換】
cp_fimg = cp.fft.fftshift(cp_fimg)  # fftshiftを使ってシフト

# パワースペクトルで表示
cp_fabs = cp.absolute(cp_fimg)  # 絶対値をとる
cp_fabs[cp_fabs < 1] = 1  # スケール調整用に1以下を1にする
cp_fpow = cp.log10(cp_fabs)  # logをとってパワースペクトル化

np_fpow = cp.asnumpy(cp_fpow)  # cupy配列 ⇒ numpy配列に変換

io.imshow(np_fpow)  # 表示
io.show()

# フーリエ逆変換
cp_ffimg = cp.fft.ifftshift(cp_fimg)  # シフトを戻す
cp_ffimg = cp.fft.ifft2(cp_ffimg)  # 【フーリエ逆変換】
cp_ffimg = cp.absolute(cp_ffimg)  # 絶対値をとって虚数をなくす
np_ffimg = cp.asnumpy(cp_ffimg)  # cupy配列 ⇒ numpy配列に変換

io.imshow(cp.asnumpy(np_ffimg))  # 表示
io.show()


# cupy repeatセット(フーリエ変換⇒シフト⇒シフト⇒フーリエ逆変換)計算時間
def cupy_fourier(cp_img, repeat):
    for i in range(repeat):
        cp_fimg = cp.fft.fftshift(cp.fft.fft2(cp_img))
        cp_ffimg = cp.fft.ifft2(cp.fft.ifftshift(cp_fimg))

# numpy repeatセット(フーリエ変換⇒シフト⇒シフト⇒フーリエ逆変換)計算時間
def numpy_fourier(np__img, repeat):
    for i in range(repeat):
        np_fimg = np.fft.fftshift(np.fft.fft2(np_img))
        np_ffimg = np.fft.ifft2(np.fft.ifftshift(np_fimg))

# Measure processing time.
outside_loop = 10
inside_loop = 100
result_cupy = timeit.timeit('cupy_fourier(cp_img, inside_loop)',
                        globals=globals(), number=outside_loop)
print("elapsed time(cupy) = {}".format(result_cupy))

result_numpy = timeit.timeit('numpy_fourier(np_img, inside_loop)',
                        globals=globals(), number=outside_loop)
print("elapsed time(numpy) = {}".format(result_numpy))

"""
i7-7700 @3.60GHz
NVIDIA GeForce GT 730
outside_loop = 10
inside_loop = 100
elapsed time(cupy) = 316.5533458640628
elapsed time(numpy) = 3178.9956735987694

i7-6700HQ @2.60GHZ
NVIDIA GeForce GTX 950M
elapsed time(cupy) = 207.7212834454453
elapsed time(numpy) = 3698.0998210764724
"""

