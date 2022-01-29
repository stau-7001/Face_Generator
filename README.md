# README

### 1 环境配置

#### 1.1 face_recognition

下载[dilb](https://pan.baidu.com/s/15bQ2vEU9pJUgwjNiQTNMmw) 解压码：whms

```makefile
pip install dlib-19.17.99-cp37-cp37m-win_amd64.whl
pip install face_recognition
```

#### 1.2 skimage

```makefile
conda install scikit-image
```

#### 1.3 tqdm

```
pip install tqdm
```

#### 1.4 scipy

在[这里]([Python Extension Packages for Windows - Christoph Gohlke (uci.edu)](https://www.lfd.uci.edu/~gohlke/pythonlibs/))找到包含MKL库版本的numpy和scipy文件（对应自己的py版本）下载并安装。

```
pip install numpy-1.21.5+mkl-cp37-cp37m-win_amd64.whl
pip install scipy-1.7.3-cp37-cp37m-win_amd64.whl
```

#### 1.5 pytorch

按照[官网](https://pytorch.org/)步骤安装即可。

### 2 使用说明

将要视频文件放于video文件夹下，并把其中要读取的视频名称分行写入`List_of_testing_videos.txt`文件内。使用如下命令运行处理程序。

```makefile
python fuce_cut.py -r 512 -t 500 -m 0 -i 5 -s 60 # -b 10
```

`-r` ：采集出来人脸的分辨率

`-t` ：可以容忍的原人脸分辨率的范围（分辨率大于**r-t**的人脸会被采集并resize到r）

`-m` ：crop和align的方式，0-VGGface，1-FFHQ

`-i` ：初始人脸采集间隔

`-s` ：未检测到人脸时跳过的帧数量

`-b` ：失焦模糊检测的阈值（一般不需要调整）

其他：文件中两个参数Max_int = 40，Min_int = 5指动态采集间隔的最大和最小值，增加Min_int和Max_int会减少人脸采集数量，提高人脸图像之间的差异。反之，人脸采集数量增加，差异减小。

------

如有任何疑问，请与stau7001@sjtu.edu.cn联系。

