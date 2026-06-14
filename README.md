# YOLOv3 目标检测

基于 YOLOv3 (Darknet) 和 OpenCV DNN 的实时目标检测项目，支持图片、视频和摄像头三种推理模式，纯 CPU 运行。

## 功能

- **单张图片检测** — 输出带标注框、类别标签和置信度的结果图
- **实时摄像头** — 实时推理并显示 FPS
- **视频文件处理** — 输出带标注的 MP4 视频
- 80 类 COCO 数据集标签（人、车、自行车、狗、猫等）
- 基于 OpenCV DNN 的纯 CPU 推理，无需 GPU

## 环境要求

- Python 3.7+
- OpenCV（带 DNN 模块）

```bash
pip install opencv-python numpy
```

## 模型文件

| 文件 | 大小 | 获取方式 |
|------|------|----------|
| `yolov3.cfg` | 8.3 KB | 已包含在仓库中 |
| `coco.names` | 621 B | 已包含在仓库中 |
| `yolov3.weights` | 237 MB | [从 Darknet 下载](https://pjreddie.com/media/files/yolov3.weights) |

下载 `yolov3.weights` 后放在项目目录下，与 `yolov3.cfg` 同级即可。

## 使用方法

```bash
cd YoloModel

# 检测单张图片
python deploy_yolo.py --image 图片路径.jpg

# 检测图片并保存结果
python deploy_yolo.py --image 图片路径.jpg --save

# 打开摄像头实时检测
python deploy_yolo.py --webcam

# 检测视频文件
python deploy_yolo.py --video 视频路径.mp4
```

- 摄像头/视频模式下按 `q` 退出
- 图片模式下按任意键关闭窗口

## 参数配置

`deploy_yolo.py` 中的关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CONF_THRESHOLD` | 0.5 | 最低置信度阈值 |
| `NMS_THRESHOLD` | 0.4 | 非极大值抑制 IoU 阈值 |
| 输入尺寸 | 608×608 | 在 `yolov3.cfg` 和代码中统一配置 |

如需使用 GPU（需安装 CUDA 版 OpenCV），修改代码中的：

```python
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```

## 项目结构

```
YoloModel/
├── deploy_yolo.py      # 主部署脚本
├── yolov3.cfg          # Darknet 模型配置
├── coco.names          # 80 类 COCO 标签
├── yolov3.weights      # 预训练权重（需自行下载）
└── LICENSE             # 许可证
```

## 致谢

- YOLOv3 原作者 Joseph Redmon 和 Ali Farhadi：[pjreddie.com/darknet/yolo](https://pjreddie.com/darknet/yolo/)
- OpenCV DNN 模块提供 Darknet 模型加载支持
- COCO 数据集提供 80 类别标签
