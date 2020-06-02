from abc import ABCMeta, abstractmethod
import functools
from typing import List, Union
import pathlib

import numpy

from PIL import Image, ImageFilter
from nptdms import TdmsFile


class FilesLoaderStrategy(metaclass=ABCMeta):
    """
    strategy
    """

    @abstractmethod
    def file_load(self, file_paths: List[pathlib.Path]) -> List[numpy.ndarray]:
        """
        file_paths: リストに格納したpathlibのPathオブジェクト
        複数のfailのPathオブジェクトをリストに格納したものを渡し、
        それらのファイルを読み込んで変換した複数のデータをリストに格納してまとめて返す。
        """
        pass


class ImageFilesLoader(FilesLoaderStrategy):
    """
    ConcreteStrategy
    画像ファイルの読み込み用

    resize: 画像のリサイズをタプルで指定する
    blur_radius: 平滑化の指定（0なら平滑化しない）
    hsv: hsvにするかどうか
    """

    def __init__(self, resize: tuple = (160, 90), blur_radius: int = 0, hsv: bool = False):
        self._resize = resize
        self._blur_radius = blur_radius
        self._hsv = hsv

    def file_load(self, file_paths: List[pathlib.Path]) -> List[numpy.ndarray]:
        # pillowで画像データを読み込み
        pil_imgs = [Image.open(path) for path in file_paths]

        # ガウシアン平滑化
        if self._blur_radius != 0:
            pil_imgs = [img.filter(ImageFilter.GaussianBlur(radius=self._blur_radius)) for img in pil_imgs]

        # リサイズ
        resize_filter = Image.BOX
        pil_imgs = [img.resize(self._resize, resize_filter) for img in pil_imgs]

        # HSV変換
        if self._hsv:
            pil_imgs = [img.convert("HSV") for img in pil_imgs]

        pil_imgs = [numpy.asarray(img) for img in pil_imgs]

        return pil_imgs


class TdmsFilesLoader(FilesLoaderStrategy):
    """
    ConcreteStrategy
    TDMS形式ファイルの読み込み用

    TMDS形式のファイルの任意の位置から、指定した長さのデータを切り取る
    numger: スライスして取り出すデータの個数
    key: スライスの開始位置の指定
    """

    def __init__(self, number: int, key: Union[str, int] = 'all'):
        # データのスライス方法を決めておく
        self._slice_method = functools.partial(self.sliced_array, key=key, number=number)

    @staticmethod
    def sliced_array(array: numpy.ndarray, key: Union[str, int], number: int = 1) -> List[numpy.ndarray]:
        """
        numpy.ndarrayを指定した範囲でスライスして切り出す
        array: スライス処理の対象となるnumpy.ndarray
        key:スライスの開始位置の指定 'all', 'head', 'middle', 'tail', int
        number:スライスして取り出すデータの個数(key='all'の場合は無視される)
        """
        total_number = len(array)
        assert total_number > number, "The specified number of data({0}) is more than the total \
                                        number of data({1})!".format(number, total_number)

        def position_all(key):
            start = 0
            stop = total_number - 1
            return start, stop

        def position_head(key):
            start = 0
            stop = start + number
            return start, stop

        def position_middle(key):
            start = int(total_number / 2)
            stop = start + number
            return start, stop

        def position_tail(key):
            stop = total_number - 1
            start = stop - number
            return start, stop

        def position_other(key):
            start = int(key)
            stop = start + number
            return start, stop

        key_dict = {
            'all': position_all,
            'head': position_head,
            'middle': position_middle,
            'tail': position_tail
        }

        # dict.get()は、None以外の値を返したい場合は、第二引数にキーが存在しない場合に返すデフォルト値を指定
        start, stop = key_dict.get(key, position_other)(key)

        return array[start: stop]

    def file_load(self, file_paths: List[pathlib.Path]) -> numpy.ndarray:
        # tdmsファイル をpandasのdataframeとして読み込む
        dfs = [TdmsFile(path).as_dataframe() for path in file_paths]

        # object型のndarrayとして取得し、float型に変換しておく
        arrs_list = [df.values.astype(float) for df in dfs]

        # 転置して横に並べ、flatten()で1次元化しておく
        '''
        こんなデータを
        [[0.1 0.2 0.3]
         [0.2 0.4 0.6]
         …
         [0.3 0.6 0.9]
         [0.4 0.8 1.2]]

        こんな感じに
        [[0.1 0.2 … 0.3 0.4]
         [0.2 0.4 … 0.6 0.8]
         [0.3 0.6 … 0.9 1.2]]
        '''
        arrs_list = [arr.T.flatten() for arr in arrs_list]

        # 指定した範囲でデータを切り取る
        arrs_list = [self._slice_method(arr) for arr in arrs_list]

        return arrs_list


class FileLoadContext(object):
    """
    Context
    Strategy役を利用する役
    ConcreteStrategy役のインスタンスを持っていて、必要に応じてそれを利用します。
    strategy: FilesLoaderStrategy
    file_ext: 扱うファイルの拡張子
    """

    def __init__(self, strategy: FilesLoaderStrategy, file_ext: str):
        self.__strategy = strategy
        self.__file_ext = file_ext

    def file_load(self, file_paths: List[pathlib.Path]) -> List[numpy.ndarray]:
        return self.__strategy.file_load(file_paths)

    def file_extension(self):
        return self.__file_ext
