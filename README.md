# README

### 1 Getting Started

#### 1.1 face_recognition

download from [dilb](https://pan.baidu.com/s/15bQ2vEU9pJUgwjNiQTNMmw) ：whms

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

download numpy+mkl and scipy from [here]([Python Extension Packages for Windows - Christoph Gohlke (uci.edu)](https://www.lfd.uci.edu/~gohlke/pythonlibs/)).

```
pip install numpy-1.21.5+mkl-cp37-cp37m-win_amd64.whl
pip install scipy-1.7.3-cp37-cp37m-win_amd64.whl
```

#### 1.5 pytorch

Just follow the steps on the [official website](https://pytorch.org/) to install.
### 2 Usage

Place the video files in the video folder and write the names of the videos to be processed in the `List_of_testing_videos.txt` file.

```makefile
python fuce_cut.py -r 512 -t 500 -m 0 -i 5 -s 60 # -b 10
```

`-r` ：Resolution of the captured face

`-t` ：The range of tolerable original face resolution（faces with resolution greater than **r-t** will be captured and resize to r）

`-m` ：Crop and alignment，0-VGGface，1-FFHQ

`-i` ：Initial face capture interval

`-s` ：Number of skipped frames when no face is detected

`-b` ：Threshold value for out-of-focus blur detection

The two parameters Max_int = 40 and Min_int = 5 in the file refer to the maximum and minimum values of the dynamic acquisition interval. Increasing Min_int and Max_int decreases the number of face acquisitions and increases the difference between face images. 

### 2 ACK
[void_zxh](https://github.com/void-zxh)
