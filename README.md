用 AI 来抓宝可梦吧！

## 引言

前些天得到国行 switch 停运的消息，其实并不是很难受，反而因为能够免费得到四款游戏而感到欣喜。

按说正处于高三的关键时期，应该好好学习，游戏什么的不应该碰。可是压力愈积俞多，总要找到派遣的出口，正巧因为这次停运接触到宝可梦，它们可爱的样子带给我极大的治愈，能够填平伤病和压力的沟壑，给予我些许慰藉。

在下面晒两张我觉得很可爱的宝可梦的图片。

![](https://cdn.luogu.com.cn/upload/image_hosting/l4lf7mww.png)

![](https://cdn.luogu.com.cn/upload/image_hosting/9lt5bc6n.png)

后面那只名为 "鲤鱼王" 的宝可梦，是这篇博客的主角，它的异色形态通体金黄，符合传统文化中对“锦鲤”的描述，象征着好运。因为宝可梦能够在世代间传递，我希望捕获一只能陪伴我一辈子的最完美的金色鲤鱼王，让好运时时相伴。

说干就干，在 《精灵宝可梦 let's go》中（游戏画面如下图所示），不计连锁捕捉，在野外遇见闪光精灵的最高概率为 $1-(\dfrac{4095}{4096})^4 \approx \dfrac{1}{1024}$，而要遇见一只满个体值的宝可梦，在连锁捕捉 31 次之后，概率为 $\dfrac{1}{32}\times \dfrac{1}{32}=\dfrac{1}{1024}$ , 因此想要捕获到满个体值金色鲤鱼王的概率是 $\dfrac{1}{1048576}$ 。（**金鱼稀**）

![](https://cdn.luogu.com.cn/upload/image_hosting/stt4xery.png)
这个概率微乎其微，如果手动刷闪，将消耗极大的精力。于是，我想到了利用 AI 帮我实时监测游戏画面，一旦出现金色鲤鱼王，就发出声音提醒，我再接管游戏。虽然这样做不会改变概率，但这样我就不必时时刻刻守在屏幕前，能够在不干扰学习的情况下，慢慢等待奇迹的到来。

## 技术细节

yolo11 系列模型有着强大的目标检测能力，而我的需求也比较简单，即识别画面中的金色鲤鱼，所以我选择用 yolo11 来完成这次任务。

yolo11 官网 ： https://docs.ultralytics.com/zh/models/yolo11/

注：目标检测，即识别出图像中的对象并用方框框出，并给出置信度

### Part 1 环境部署

为了使用 GPU 加速模型的训练，首先要安装  CUDA 和 cuDNN (适用于 NVIDIA GPU)，这两个组件在网上都有详细的安装教程，这里不再赘述，需要提醒的是，务必安装和自己的 GPU 相匹配的 CUDA 和 cuDNN , 如果版本不兼容，大概率会报错。

之后使用 conda 创造一个虚拟环境，没有安装过 conda 的需要先安装。


```
conda create -n yolo python=3.12 (创建虚拟环境）
conda activate yolo (激活虚拟环境)
```

我这里用的是 python 3.12， 其他的版本有可能遇到兼容性问题 (我没试过)

接下来的步骤可以参考 Ultralytics 的官方教程 ：https://docs.ultralytics.com/zh/quickstart/

具体步骤是，首先安装对应你 CUDA 和 cuDNN 版本的 PyTorch，链接如下，https://pytorch.org/get-started/locally/

例如我的 CUDA 版本是 12.6，那么我就只需在命令行里输入

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

配置好上述环境后，就可以安装 Ultralytics 库了

```
pip install ultralytics
```

如果下载速度慢，在命令后面加上 `-i https://pypi.tuna.tsinghua.edu.cn/simple/`，就可以从清华源下载了。感谢清华开源（bushi

如果要发出提示音，可以选择 `playsound` 库

```
pip install playsound
```

这样最基本的环境就部署好了

### Part 2 数据集准备

因为 yolo11 是预训练模型，如果要识别自己特定的目标，就需要准备相关的数据集，并在这个特制的数据集上对模型进行训练

yolo 的数据集要求每一张图片对应一个标签文件，标签文件的内容大概是 `class x y w h` ，其中 `class` 是对象的类别，在我这个例子中，需要识别金色的和红色的鱼，所以用 `0` 代表金鱼，用 `1` 代表红鱼。`x y w h` 均为属于 $(0,1)$ 的实数，分别代表对象所在方框的横坐标、纵坐标、宽度、高度。例如对于图片 `img.jpg`，图中央有一条金鱼，那么我们需要根据金鱼的位置准备文件 `img.txt`，文件中内容为

```
0 0.515234 0.521528 0.194531 0.301389
```

#### 准备图片

- 方式一：在遇见金色鲤鱼王的时候，直接截图，就像这样
  ![](https://cdn.luogu.com.cn/upload/image_hosting/3bewczff.png)
  但是由于上述提到的低概率，用这种方式获取数据效率太低

- 方式二：带着抓到的金色鲤鱼王到处遛弯，就像这样
  ![](https://cdn.luogu.com.cn/upload/image_hosting/a3gq9gwd.png)
  这种方法就相对更高效一些

- 方式三：在图鉴里旋转金色鲤鱼王并截图

  ![](https://cdn.luogu.com.cn/upload/image_hosting/x7mqnfjv.png)

- 方式四：博主博主，你的三个方法虽然很好，但还是太依赖 switch 了，有没有那种更普适的方法？

  有的兄弟有的，P 图，美其名曰： **合成数据**

  ![](https://cdn.luogu.com.cn/upload/image_hosting/shk9dvh8.png)

#### 数据标注

为了更高效地获得标签文件，我们需要使用数据标注软件，这里我用的是 labelimg, 安装方法如下

```
conda create -n labelimg python=3.9
pip install labelimg
labelimg
```

具体的使用方法，可以参见网上的教程

#### 数据划分

在这一部分，数据集将被分为训练集和验证集，一般大小比例为 $8 : 2$，这一步我从网上找来了现成的代码，贴在下方自取


```python
# by CSDN 迪菲赫尔曼
import os
import random
import shutil

def copy_files(src_dir, dst_dir, filenames, extension):
    os.makedirs(dst_dir, exist_ok=True)
    missing_files = 0
    for filename in filenames:
        src_path = os.path.join(src_dir, filename + extension)
        dst_path = os.path.join(dst_dir, filename + extension)
        
        # Check if the file exists before copying
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: File not found for {filename}")
            missing_files += 1

    return missing_files

def split_and_copy_dataset(image_dir, label_dir, output_dir, train_ratio=0.8, valid_ratio=0.2, test_ratio=0):
    # 获取所有图像文件的文件名（不包括文件扩展名）
    image_filenames = [os.path.splitext(f)[0] for f in os.listdir(image_dir)]

    # 随机打乱文件名列表
    random.shuffle(image_filenames)

    # 计算训练集、验证集和测试集的数量
    total_count = len(image_filenames)
    train_count = int(total_count * train_ratio)
    valid_count = int(total_count * valid_ratio)
    test_count = total_count - train_count - valid_count

    # 定义输出文件夹路径
    train_image_dir = os.path.join(output_dir, 'train', 'images')
    train_label_dir = os.path.join(output_dir, 'train', 'labels')
    valid_image_dir = os.path.join(output_dir, 'valid', 'images')
    valid_label_dir = os.path.join(output_dir, 'valid', 'labels')
    test_image_dir = os.path.join(output_dir, 'test', 'images')
    test_label_dir = os.path.join(output_dir, 'test', 'labels')

    # 复制图像和标签文件到对应的文件夹
    train_missing_files = copy_files(image_dir, train_image_dir, image_filenames[:train_count], '.jpg')
    train_missing_files += copy_files(label_dir, train_label_dir, image_filenames[:train_count], '.txt')

    valid_missing_files = copy_files(image_dir, valid_image_dir, image_filenames[train_count:train_count + valid_count], '.jpg')
    valid_missing_files += copy_files(label_dir, valid_label_dir, image_filenames[train_count:train_count + valid_count], '.txt')

    test_missing_files = copy_files(image_dir, test_image_dir, image_filenames[train_count + valid_count:], '.jpg')
    test_missing_files += copy_files(label_dir, test_label_dir, image_filenames[train_count + valid_count:], '.txt')

    # Print the count of each dataset
    print(f"Train dataset count: {train_count}, Missing files: {train_missing_files}")
    print(f"Validation dataset count: {valid_count}, Missing files: {valid_missing_files}")
    print(f"Test dataset count: {test_count}, Missing files: {test_missing_files}")

# 使用例子，目录位置可根据需求更改
image_dir = './train'
label_dir = './test'
output_dir = './dataset'

split_and_copy_dataset(image_dir, label_dir, output_dir)
```

接下来还需要准备一个 `data.yaml` 文件，之后会用到，大致格式为：

```
train : dataset\train #路径根据实际情况进行修改
val : dataset\valid #路径根据实际情况进行修改
test : dateset\test #路径根据实际情况进行修改

#类型根据要识别的对象修改
names :
  0 : Golden
  1 : Red
```
到此为止数据集准备工作就结束了，大致目录长这样
![](https://cdn.luogu.com.cn/upload/image_hosting/6ku1753h.png)


### Part 3 模型训练

激活之前创建的虚拟环境后，就可以开始训练模型了

具体代码如下

``` python
from ultralytics import YOLO # 导入库

model = YOLO("yolo11n.pt") # 导入预训练模型

if __name__ == '__main__' :
    model.train(data = 'dataset-Golden/data.yaml', epochs = 1500, batch = 4, imgsz = 1280, name = 'GF', patience = 0) # 开始训练

```

可以注意到 `model.train` 函数有很多参数，具体含义参见 https://docs.ultralytics.com/zh/modes/train/ 中的表格，这些参数可以根据自己的情况自行调整。

'epochs' 代表训练轮数，`batch` 代表一次训练同时加载多少张图片（应根据显存大小确定），`imgsz` 代表训练时输入图像的分辨率（模型会自动调整，因此不用将数据集特意设置为相应大小），`patience=x` 代表训练中如果 x 个 epoch 都没有得到更优的模型，就停止训练，如果 x = 0 , 那么就一直训练下去。

结算画面大概长这样

![](https://cdn.luogu.com.cn/upload/image_hosting/dvrze662.png)

### Part 4 模型推理

可以参考 https://docs.ultralytics.com/zh/modes/predict/

为了获取 switch 的画面数据，首先需要一个采集卡，然后调用 `cv2.VideoCapture` 函数获取采集卡传来的视频数据，之后再利用每一帧的图片进行推理，我写的代码长这样

``` python
from ultralytics import YOLO
from playsound import playsound
import cv2
import time

model = YOLO('best.engine', task = 'detect', verbose = False) # 加载模型

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # 获取采集卡视频数据

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #设置分辨率
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 

while True :

    ret, frame = cap.read() # 获取当前帧

    if not ret :
        continue

    results = model.predict(frame, verbose = False, imgsz = 1280) # 模型预测

    cv2.imshow('switch', frame) # 实时显示画面，但因为下面代码中写了和提示音有关的代码，可以删去

    cv2.waitKey(1) # 同上

    boxes = results[0].boxes # 获取预测结果

    size = boxes.cls.size(0) # 获取检测到的鱼条数

    for i in range(0, size) :
        if (int)(boxes.cls[i]) == 0 and boxes.conf[i] > 0.8: # 这里置信度我取 0.8 了，可以按情况改
            x = (boxes.xyxyn[i][0] + boxes.xyxyn[i][2]) / 2
            y = (boxes.xyxyn[i][1] + boxes.xyxyn[i][3]) / 2
            x = (int)(1 + round((x.item() - 0.5) * 2))
            y = (int)(1 + round((y.item() - 0.5) * 2))
            print(boxes.conf[i].item())
            cv2.imwrite('capture/' + str((int)(time.time())) + '-' + str(int(boxes.conf[i].item() * 100)) + '.jpg', frame)
            ret, frame = cap.read()
            playsound('sound/' + str(y * 3 + x) + '.mp3')
            # 这里计算 x 和 y 是为了在金鱼出现在画面中的不同位置时发出不同提示音，例如，如果 x = 0, y = 0, 那么会发出 "左上" 的提示音
            break         
```

至此，整套系统就构建完毕了。

## 效果图展示
![](https://cdn.luogu.com.cn/upload/image_hosting/w2w6e7fp.png)


## 一些补充

- 额，你不一定非要用 yolo11 来抓金鱼，抓别的宝可梦也是可以的，但是数据集要自己准备
- 如果要加快模型的推理速度，可以将模型导出为 `.onnx` 或 `.engine` 格式，我采用了 `.engine` 格式，用 TensorRT 加速推理，详见 https://docs.ultralytics.com/zh/modes/export/
- 版本兼容是非常重要的，配置各种环境时一定要注意
