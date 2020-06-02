import os
import re
import random
from typing import List, Union
from pathlib import Path
import argparse
import dataclasses

import numpy as np
import joblib

from strategy.files_loader_strategy import TdmsFilesLoader, ImageFilesLoader, FileLoadContext


@dataclasses.dataclass
class DataSet(object):
    """
    python>=3.7

    Attributes
    ----------
    data : numpy.ndarray
        データ本体
    targe : numpy.ndarray
        one-hotで表現されたクラス
    target_names : numpy.ndarray
        クラスと名前を紐づける情報
    """
    data: np.ndarray
    target: np.ndarray
    target_names: np.ndarray


class DataSetMaker(object):
    """
    strategy_ctx: FileLoadContext
    FilesLoaderStrategyのClient（依頼人）の役
    """

    def __init__(self, strategy_ctx: FileLoadContext, src_dir: str = 'data_src'):
        self.__loader = strategy_ctx
        self.__src_dir = src_dir
        self.__data_set = ...

    @staticmethod
    def paths_sort(paths):
        #         return sorted(paths, key = lambda x: int(x.name))
        return sorted(paths, key=lambda x: x.name)

    @staticmethod
    def classified_dirs(src_dir: str):
        """
        classified_dirs_p: クラス分けのディレクトリのPathオブジェクト
        """
        # Pathオブジェクトを生成
        src_dir_p = Path(src_dir)

        # iterdir() にis_dir()条件をつけてディレクトリのみ抽出
        classified_dirs_p = [p for p in src_dir_p.iterdir() if p.is_dir()]

        if not classified_dirs_p:  # 配列が空（クラス分けされたディレクトリが無い）の場合
            print("There is no classified directory in '{0}'.".format(src_dir))
            classified_dirs_p = [src_dir_p]

        return classified_dirs_p

    def data_set(self) -> Union[DataSet, None]:
        return self.__data_set

    def create_data_set(self, src_dir: str) -> DataSet:
        # データセットに格納するデータ群
        data, target, target_names = [], [], []

        # クラス分けしたディレクトリのPathObjectを取得
        dirs_p = DataSetMaker.classified_dirs(src_dir)

        target_names = [p.name for p in dirs_p]

        label_num = len(target_names)

        file_ext = self.__loader.file_extension()

        for i, po in enumerate(dirs_p):
            ## data に格納するデータの生成
            # glob() で拡張子指定して抽出
            file_paths = [p for p in po.glob("*." + file_ext)]

            # ファイル名をソートしておく（不要かも？）
            DataSetMaker.paths_sort(file_paths)
            # print([p.name for p in file_paths])

            arrays_list = self.__loader.file_load(file_paths)

            # データの追加
            data.extend(arrays_list)

            ## target に格納するデータの生成
            # one_hotラベルの生成
            one_hot_label = np.zeros(label_num)
            one_hot_label[i] = 1

            # ファイルの個数分ラベルを生成
            labels = [one_hot_label for i in file_paths]

            # ラベルデータの追加
            target.extend(labels)

        self.__data_set = DataSet(
            data=np.asarray(data),
            target=np.asarray(target),
            target_names=np.asarray(target_names)
        )

        return self.__data_set

    def save_data_set(self, file_name: str):
        # '*.joblib'というファイル名で、dataclasses型のオブジェクトを一つ保存する
        if not re.search(r'\.JOBLIB$', file_name.upper()):
            print("The extension of the saved file should be '* .joblib'."
                  "Current file name：{0}".format(file_name))
            return

        if not self.__data_set:
            print('data set is empty!')
            return

        # # '*.joblib'でdataclassesを保存
        with open(file_name, mode='wb') as f:
            # バイナリ形式で圧縮率3にして保存
            joblib.dump(self.__data_set, f, compress=3)

    def create_and_save_data_set(self, file_name: str, src_dir: str = None):
        if src_dir:
            self.__src_dir = src_dir

        assert os.path.isdir(self.__src_dir), "The source data directory '{0}' does not exist!".format(self.__src_dir)

        self.create_data_set(self.__src_dir)

        self.save_data_set(file_name)


class DataSetLoader(object):
    """
    Attributes
    ----------
    __data_set : DataSetクラス
    """

    def __init__(self, load_file: str):
        self.__data_set = ...
        self.__load_file = load_file

        assert self.__load_file, "please input an argument 'load_file = {0}'!".format(load_file)
        self.load_joblib(self.__load_file)

    def load_joblib(self, load_file: str):
        assert os.path.exists(load_file), "{0} does not exist!".format(load_file)

        with open(load_file, mode='rb') as f:
            self.__data_set = joblib.load(f)

    def load(self, flatten=False, one_hot_label=True, shuffle=False, pick_size: int = None) -> DataSet:
        """
        flatten: boolean
            'data'を一次元に平坦化する。
        one_hot_label: boolean
            'target'をone-hot表現にする（Falseの場合、クラスのラベル値となる）。
        shuffle: boolean
            'data'と'target'の相対関係を維持したままランダムに並び替えたdatasetを出力する。
        pick_size: int
            取り出すデータの個数を指定する（サンプリングサイズ：データの総数より小さい必要がある）。
            Noneの場合はpick_sizeを無視して全てのデータを取り出す。

        Returns
        -------
        data_set : DataSet
        'data','target','target_names'有するDataSet(dataclass型)

        """
        if not self.__data_set:
            return None

        data = self.__data_set.data
        target = self.__data_set.target
        target_names = self.__data_set.target_names

        if data is not None and target is not None:
            if flatten:
                data = self.to_flatten(data)

            if not one_hot_label:
                target = self.no_hot_label(target)

            if pick_size:
                data, target = self.sampling(data, target, pick_size)

            if shuffle:
                data, target = self.to_shuffle(data, target)

        data_set = DataSet(
            data=np.asarray(data),
            target=np.asarray(target),
            target_names=np.asarray(target_names)
        )
        return data_set

    @staticmethod
    def to_flatten(data):
        """ 一次元配列化 """
        return [d.flatten() for d in data]

    @staticmethod
    def no_hot_label(target):
        """ one_hotからラベル値に変換 """
        return [t.argmax(axis=0) for t in target]

    @staticmethod
    def sampling(data, target, size=None):
        """ 指定した個数のデータをランダムに取り出す """
        #         if not size or len(data) <= size:
        if not size:
            return data, target

        assert isinstance(size, int), "specified size instance type must be 'int'. ({0})".format(type(size))
        assert len(data) > size, "value of data size({0}) must be greater than specified size({1})!".format(len(data),
                                                                                                            size)
        p = random.sample(range(len(data)), size)
        p.sort()
        data = np.asarray(data)[p]
        target = np.asarray(target)[p]
        return data, target

    @staticmethod
    def to_shuffle(data, target):
        """ 配列の相対関係を保ったままシャッフルする """
        np.random.seed(0)
        p = np.random.permutation(len(data))
        data = np.asarray(data)[p]
        target = np.asarray(target)[p]
        return data, target


def parse_option_for_data_set_maker():
    dc = 'This script is ...'
    parser = argparse.ArgumentParser(description=dc)

    parser.add_argument('-i', action='store', type=str, dest='input',
                        default=None,
                        help='set the path of the original datas directory.  e.g. ./data_src')
    parser.add_argument('-o', action='store', type=str, dest='output',
                        default='dataset.joblib',
                        help='set the file name to save.  e.g. hoge.joblib')
    parser.add_argument('-e', '--ext', action='store', type=str, dest='ext',
                        default='jpg',
                        help='set the extension of the original image files.  e.g. jpg')
    return parser.parse_args()


def main():
    args = parse_option_for_data_set_maker()
    save_file = args.output
    src_dir = args.input
    data_ext = args.ext

    # FileLoaderの生成
    file_loader = ImageFilesLoader(resize=(160, 90), blur_radius=0, hsv=False)
    # FileLoadContextの生成
    ctx = FileLoadContext(file_loader, file_ext=data_ext)

    # # 元データの格納先を指定してDataSetMakerを生成
    # path = Path('D:\\data_src')
    # print(path)

    # StrategyのクライアントであるDataSetMakerを生成
    # ds_maker = DataSetMaker(ctx, src_dir=path)
    ds_maker = DataSetMaker(ctx, src_dir=src_dir)

    # データセットを生成し、joblibファイルで保存
    ds_maker.create_and_save_data_set(save_file)

    # データ内容確認
    with open(save_file, mode='rb') as f:
        ds = joblib.load(f)
        data = ds.data
        target = ds.target
        target_names = ds.target_names

    print('data set shape: ', data.shape)
    print('data set target: ', target)
    print('data set target names: ', target_names)

    # データセットローダでデータを読み込む
    loader = DataSetLoader(save_file)
    dataset = loader.load(flatten=True, one_hot_label=False, shuffle=False)

    data = dataset.data
    target = dataset.target
    target_names = dataset.target_names

    print('data set shape: ', data.shape)
    print('data set target: ', target)
    print('data set target names: ', target_names)


if __name__ == '__main__':
    main()
