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

## 人像检测结果

以下为 10 张含人物图片的 YOLOv3 检测结果（置信度阈值 0.5）。

| 图片 | 检出人数 | 置信度范围 |
|------|----------|-----------|
| photo_01 | 1 人 | 0.826 |
| photo_02 | 4 人 | 0.958 - 0.999 |
| photo_03 | 11 人 | 0.511 - 0.997 |
| photo_04 | 1 人 | 0.867 |
| photo_05 | 0 人 | - |
| photo_06 | 6 人 | 0.621 - 0.991 |
| photo_07 | 2 人 | 0.765 - 0.999 |
| photo_08 | 4 人 | 0.994 - 0.999 |
| photo_09 | 2 人 | 0.756 - 0.997 |
| photo_10 | 3 人 | 0.998 |

### photo_01
![photo_01](../detection_results/photo_01_detected.jpg)

### photo_02
![photo_02](../detection_results/photo_02_detected.jpg)

### photo_03
![photo_03](../detection_results/photo_03_detected.jpg)

### photo_04
![photo_04](../detection_results/photo_04_detected.jpg)

### photo_05
![photo_05](../detection_results/photo_05_detected.jpg)

### photo_06
![photo_06](../detection_results/photo_06_detected.jpg)

### photo_07
![photo_07](../detection_results/photo_07_detected.jpg)

### photo_08
![photo_08](../detection_results/photo_08_detected.jpg)

### photo_09
![photo_09](../detection_results/photo_09_detected.jpg)

### photo_10
![photo_10](../detection_results/photo_10_detected.jpg)

## 致谢

- YOLOv3 原作者 Joseph Redmon 和 Ali Farhadi：[pjreddie.com/darknet/yolo](https://pjreddie.com/darknet/yolo/)
- OpenCV DNN 模块提供 Darknet 模型加载支持
- COCO 数据集提供 80 类别标签