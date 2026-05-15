# HI3519DV500 人脸识别特征提取模型开发总结

## 1. 项目简介
本项目基于海思 SVP ACL (Ascend Computing Language) 接口，在 HI3519DV500 芯片上部署和运行基于 YOLOv8 架构的自定义人脸检测模型 (`yolov8n-face`)。项目实现了从 ONNX 模型转换、图片预处理、NPU 推理到 CPU 后处理（解码、NMS、画框）的完整端到端流程。

## 2. 环境与依赖库
本项目的开发与仿真运行环境位于 Ubuntu 18.04 的 Docker 容器内。
- **操作系统**: Ubuntu 18.04 x86_64 / aarch64 (Docker 环境)
- **目标芯片**: HI3519DV500
- **编译器**: g++ (仿真环境), aarch64-v01c01-linux-musl-gcc / aarch64-v01c01-linux-gnu-gcc (板端交叉编译)
- **核心依赖**:
  - **SVP ACL 库**: 海思/昇腾硬件推理接口库。
  - **OpenCV**: 用于 C++ 层的图片读取、后处理画框和保存。
  - **Python 3.7.5**: 用于运行模型转换脚本和数据预处理脚本。
  - **Pillow / NumPy / ONNX**: Python 侧的数据处理和模型检查依赖。

## 3. 项目目录结构
```text
yolo/
├── data/                  # 测试数据目录
│   ├── dog_bike_car.jpg       # 原始测试图片
│   ├── dog_bike_car_yolov8.bin # 预处理后的二进制图片输入
│   └── image_ref_list.txt     # ATC 转换所需的量化/校准参考列表
├── inc/                   # C++ 头文件目录
│   ├── model_process.h        # 模型加载、推理、后处理类声明
│   ├── sample_process.h       # ACL 资源初始化、管理类声明
│   └── utils.h                # 公共工具函数宏定义
├── src/                   # C++ 源代码目录
│   ├── CMakeLists.txt         # 源码目录的 CMake 编译脚本 (非常重要，不可缺失)
│   ├── main.cpp               # 程序入口，解析命令行参数
│   ├── model_process.cpp      # 核心逻辑：模型推理与 YOLOv8-Face CPU 后处理实现
│   ├── sample_process.cpp     # ACL Context/Stream 等资源的生命周期管理
│   └── utils.cpp              # 文件读取等辅助函数实现
├── script/                # 辅助脚本目录
│   ├── compile_model.sh       # ATC 模型转换脚本 (将 ONNX 转换为 OM)
│   └── transferPic.py         # 图片预处理脚本 (将 JPG 转为 NCHW 的 BIN 文件)
├── model/                 # 转换后的 OM 离线模型存放目录
├── onnx_model/            # 原始 ONNX 模型存放目录
│   └── yolov8n-face.onnx      # 自定义的 YOLOv8 人脸检测模型
├── out/                   # 编译产物及运行结果目录
│   ├── func_main              # 编译生成的功能仿真可执行文件
│   ├── yolov8_face_detResult.txt # 运行后输出的检测框坐标结果
│   └── out_img_yolov8_face.jpg   # 运行后输出的画框结果图片
├── CMakeLists.txt         # 顶层 CMake 构建脚本
├── build.sh               # 一键编译脚本
└── insert_op.cfg          # ATC 转换时的 AIPP (图像预处理) 配置文件
```

## 4. 核心代码逻辑说明
为了适配自定义的 `yolov8n-face` 模型，我们在官方样例的基础上进行了精简和修改，主要集中在 `8_face` 模式：
1. **模型输出解析**: 原始 `yolov8n-face.onnx` 的输出维度为 `(1, 20, 8400)`。其中 20 个通道分别代表：4个框坐标 + 1个类别置信度(人脸) + 15个关键点坐标。
2. **CPU 后处理 (`model_process.cpp`)**:
   - `FilterYolov8FaceBox`: 专门针对 Face 模型设计的解码函数。由于导出的 ONNX 在类别分支末端缺失 Sigmoid 激活，代码中手动添加了 `1.0f / (1.0f + exp(-val))` 进行概率转换。同时，严格限制只读取第 5 个通道作为唯一类别（classId = 0），避免了官方 80 类别硬编码导致的内存越界问题。
   - `OutputModelResultYoloV8Face`: 整合了解码、NMS (非极大值抑制) 和结果保存、画框的流程。
3. **硬件 RPN vs CPU 后处理**:
   - 本项目使用的是**纯 CPU 后处理**路线，因此**不需要**对 YOLO 源码打 `0001-yolov8-rpn.patch`。直接使用最原始的 ONNX 模型进行转换即可。

## 5. 快速启动指南 (Docker 环境)

### 第一步：准备数据
如果需要更换测试图片，请使用 Python 脚本将 JPG 转换为模型所需的 BIN 格式：
```bash
cd /workspace/1_classification/yolo/data
python3.7.5 ../script/transferPic.py 8_cpu  # 生成 640x640 分辨率的输入
```

### 第二步：模型转换 (ONNX -> OM)
使用 `atc` 工具将 ONNX 转换为适配 HI3519DV500 的离线模型。该过程会读取 `insert_op.cfg` 进行 AIPP 配置。
```bash
cd /workspace/1_classification/yolo
bash script/compile_model.sh
```
*成功后会在 `model/` 目录下生成 `yolov8_face_original.om`。*

### 第三步：编译 C++ 工程
运行一键编译脚本。如果遇到 `CMakeLists.txt` 缺失的报错，请确保 `src/CMakeLists.txt` 文件存在。
```bash
cd /workspace/1_classification/yolo
rm -rf out/*   # 清理旧产物
./build.sh     # 编译生成可执行文件
```

### 第四步：执行推理测试
进入产物目录，指定 `8_face` 参数运行仿真可执行文件：
```bash
cd /workspace/1_classification/yolo/out
./func_main 8_face
```

### 第五步：查看结果
运行成功后，当前 `out/` 目录下会生成：
- `yolov8_face_detResult.txt`: 包含检测到的 bounding box 坐标及置信度。
- `out_img_yolov8_face.jpg`: 带有检测框的可视化图片。