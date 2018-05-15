####  安装OpenCV库
- `brew install opencv3`，安装opencv库
- 创建virtualenv虚拟环境venv
- 进入 `venv/lib/python3.6/site-packages`目录下
- 执行以下命令建立软连接
```
# 实际中需要看下具体版本和路径
ln -s /usr/local/Cellar/opencv/3.4.1_5/lib/python3.6/site-packages/cv2.cpython-36m-darwin.so  cv2.so
```
- `pip install numpy`
- 进入Python环境下，输入`import cv2`，如果不报错说明已经安装成功


#### 安装pyqt5
```
brew install PyQt5
```
- 将/anaconda/lib/site-packages/下的PyQt5这个文件夹以及sip.so文件，复制到该虚拟环境下的/venv/site-packages下
