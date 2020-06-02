numpy.ndarray型のデータセットを生成するスクリプト
===

[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

このスクリプトは、クラスごとにディレクトリを分けて格納されている画像ファイル等を参照し、`numpy.ndarray`型のデータセットを生成するスクリプトです。
読み込んだデータを適宜処理した後、`numpy.ndarray`型に変換して`*.joblib`ファイルとして保存します。

追記：National Instruments社のTDMS形式のデータにも対応させました。

## 必要なライブラリ等
* [numpy](https://github.com/numpy/numpy)
* [Pillow](https://github.com/python-pillow/Pillow)
* [pandas](https://github.com/pandas-dev/pandas)
* [joblib](https://github.com/joblib/joblib)
* [npTDMS](https://github.com/adamreeve/npTDMS)

 [pathlib](https://docs.python.org/ja/3.5/library/pathlib.html)を用いているので、Python3.4以上を推奨します。

## 使い方

#### ディレクトリ構成
以下のような構成で、分類したいクラスごとにディレクトリを分けて画像ファイル（やTDMS形式ファイル）を格納しておいてください。
ディレクトリの名称は任意ですが、クラス分けに使用したディレクトリ名は、データセット生成時に`target_names`として利用されますので留意してください。

```
│  dataset.joblib
│  data_set_maker.py
└─ strategy
│   └─ files_loader_strategy.py
│    
└─ data_src
    ├─0_black
    │      c_blk_01.jpg
    │      ...
    │
    ├─1_red
    │      c_red_01.jpg
    │      ...
    │
    └─2_blue
            c_blu_01.jpg
            ...
```



#### データセットの生成（joblibファイルの保存）

以下のように、`-o`に続けてデータセットの保存先となるjoblibファイル名、`-i`に続けて元画像を格納したディレクトリ（上記ディレクトリ構成の場合は`data_src`）等を指定します。  

```
$ python data_set_maker.py -h
usage: data_set_maker.py [-h] [-i INPUT] [-o OUTPUT] [-e EXT]

This script is ...

optional arguments:
  -h, --help         show this help message and exit
  -i INPUT           set the path of the original datas directory. e.g.
                     ./data_src
  -o OUTPUT          set the file name to save. e.g. hoge.joblib
  -e EXT, --ext EXT  set the extension of the original image files. e.g. jpg

$ python data_set_maker.py -i data_src -o dataset.joblib

data set shape:  (9, 90, 160, 3)
data set target:  [[1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
   …

```


より詳細な設定を行いたい場合は、以下のように実施できます。
（TDMS形式のデータを処理したい場合は、「TdmsFilesLoader」を使って同様の流れで処理してください）

以下は、元画像を幅160高さ90にリサイズした後平滑化しHSV色空間に変換したものを`dataset.joblib`というファイル名で保存する例です。
```
$ ipython

In [1]: from data_set_maker import ImageFilesLoader, FileLoadContext, DataSetMaker

In [2]: ctx = FileLoadContext(ImageFilesLoader(resize=(160, 90), blur_radius=1, hsv=True), file_ext='jpg')

In [3]: ds_maker = DataSetMaker(ctx, src_dir='data_src')

In [4]: ds_maker.create_and_save_data_set('dataset.joblib')
```

#### データセットの読み込み（joblibファイルの読み込み）
生成された`*.joblib`ファイルに、`data`、`target`、`target_names`というキーで`numpy.ndarray`が格納されているので、以下のように取り出してください。
```
$ ipython

In [1]: import joblib

In [2]: file_name = 'dataset.joblib'

In [3]: with open(file_name, mode='rb') as f:
   ...:     ds = joblib.load(f)
   ...:     data = ds.data
   ...:     target = ds.target
   ...:     target_names = ds.target_names
   ...: 

In [4]: print(data.shape)
(9, 90, 160, 3)

In [5]: print(target)
[[1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]]

In [6]: print(target_names)
['0_black' '1_red' '2_blue']

```

#### 【補足】データセットの加工
`DataSetLoader`を使用することで、`data`を1次元配列に平坦化したり、`target`をone-hot表現からラベル値に変更することができます。

```
$ ipython

In [1]: from data_set_maker import DataSetLoader

In [2]: loader = DataSetLoader('dataset.joblib')

In [3]: ds = loader.load(flatten=True, one_hot_label=False)

In [4]: data = ds.data

In [5]: target = ds.target

In [6]: target_names = ds.target_names

In [7]: print(target)
[0 0 0 1 1 1 2 2 2]

In [8]: print(target_names)
['0_black' '1_red' '2_blue']

```
