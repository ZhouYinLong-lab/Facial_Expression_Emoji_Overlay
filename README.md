# Emoji Mask · 表情人脸贴图

[![在线体验](https://img.shields.io/badge/GitHub%20Pages-在线体验-2856E8?logo=github)](https://zhouyinlong-lab.github.io/Facial_Expression_Emoji_Overlay/)
[![JavaScript](https://img.shields.io/badge/JavaScript-原生-F7DF1E?logo=javascript&logoColor=111)](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20Vision-00A67E)](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/web_js)
[![隐私](https://img.shields.io/badge/图片处理-仅在浏览器本地-16855B)](#隐私与模型加载)

上传一张或多张合影，自动找到人脸，并用 Unicode Emoji、模糊、像素或纯色遮住每张脸。贴图可以继续移动、缩放、替换，最终按原图尺寸导出 PNG。

**[打开在线版本](https://zhouyinlong-lab.github.io/Facial_Expression_Emoji_Overlay/)** · 无需安装，不会上传照片。

![Emoji Mask 工作台](assets/demo.png)

## 主要功能

- 检测近景人像和多人合影，单次最多保留 80 张脸，并显示检测人数与当前遮挡数量
- 对面积最大的 6 张脸进一步识别开心、悲伤、生气、惊讶和中性表情
- 未进入表情层或判断不确定的人脸默认使用 `😄`，保证检测到就会被遮住
- 支持“自动表情 / 全部开心 / 随机表情”三种自动模式
- 支持 Emoji、局部模糊、像素马赛克和纯色四种单脸遮挡效果
- 支持补脸模式：在画布拖拽矩形，为远处或被遮挡的人脸手动补一层遮挡
- 支持拖动、等比例缩放、删除和手动替换单个 Emoji
- 支持滚轮缩放、画布平移和右键手动添加贴图
- 支持多选照片，按顺序处理并逐张下载 `原文件名-emoji.png`
- 按原始图片分辨率导出 PNG，视图缩放不会影响导出位置
- Face Landmarker 不可用时仍可依靠 Face Detector 生成开心贴图

## 工作原理

项目使用两层识别流程，在“找全人脸”和“判断表情”之间取平衡：

```text
原始图片
   │
   ├─ Face Detector：整图 + 多尺度重叠分块
   │      └─ 合并重复框，为每张脸先生成 😄
   │
   └─ Face Landmarker：处理面积最大的前 6 张脸
          └─ 用关键点和 blendshape 更新表情、位置、大小与旋转
```

| 阶段 | 作用 | 限制 |
| --- | --- | --- |
| 人脸层 | 尽量找全合影中的人脸 | 最多 80 张 |
| 表情层 | 细化表情和贴图位置 | 面积最大的前 6 张 |
| 编辑层 | 手动修正识别结果 | Emoji / 模糊 / 像素 / 纯色 |

## 使用方法

1. 选择一张或多张照片，也可以把图片拖到画布中。
2. 选择自动模式后点击“识别并贴图”。大型合影会进行多尺度分块检测，通常需要数秒到十几秒。
3. 检测完成后查看“检测到 X 张脸 · 当前 Y 个遮挡”。远处、小脸或被遮挡区域漏检时，点击“补脸”，在画布拖拽矩形补充遮挡。
4. 选中任意贴图后，可以切换单张脸的遮挡效果、Emoji、位置和大小；按 `Delete` 或 `Backspace` 删除选中项。
5. 点击“导出 PNG”保存当前照片。选择了多张照片时，点击“批量识别并下载”，每张完成后会单独下载 PNG。

### 编辑操作

| 操作 | 效果 |
| --- | --- |
| 拖动 Emoji | 移动贴图 |
| 拖动选中边框 | 等比例缩放贴图 |
| 拖动画布空白处 | 平移视图 |
| 鼠标滚轮 | 以指针位置为中心缩放 |
| 图片区域右键 | 手动添加开心 Emoji |
| 补脸模式拖拽矩形 | 添加一个当前遮挡效果的手动遮挡 |
| 遮挡方式按钮 | 只切换当前选中贴图；未选中时设置新贴图默认效果 |
| `Delete` / `Backspace` | 删除选中贴图 |

### 三种自动模式

- **自动表情：** 人脸层先为所有检测结果生成 `😄`，再为面积最大的 6 张脸识别基础表情。
- **全部开心：** 跳过表情识别，适合大型合影和更快的批量处理。
- **随机表情：** 从五种 Unicode Emoji 中随机分配，仅供娱乐，不代表真实情绪。

### 四种遮挡效果

每张脸可以独立使用 Emoji、模糊、像素或纯色。预览和导出都使用同一个 Canvas 绘制逻辑，位置、旋转和尺寸保持一致。模糊、像素和纯色适合对隐私要求更高的照片；Unicode Emoji 的具体外观会随操作系统和浏览器字体变化。

### 批量处理

一次选择多张图片后，应用会按文件顺序逐张完成“加载 → 检测 → 遮挡 → 下载”。状态栏会显示 `正在处理 3/8`，单张失败不会阻塞后续文件。当前版本采用逐张下载，不生成 ZIP，也不处理视频。

## 本地运行

项目使用原生 HTML、CSS 和 JavaScript，不需要安装依赖或构建工具。ES Module 不能稳定地从 `file://` 直接加载，请通过静态服务器运行：

```powershell
git clone https://github.com/ZhouYinLong-lab/Facial_Expression_Emoji_Overlay.git
cd Facial_Expression_Emoji_Overlay
python -m http.server 5173 --bind 127.0.0.1
```

打开 [http://127.0.0.1:5173/](http://127.0.0.1:5173/)。推荐使用最新版 Chrome 或 Edge。

## 隐私与模型加载

- 图片、关键点和编辑结果只存在于当前浏览器页面，不会上传到项目服务器。
- Face Detector 和 Face Landmarker 模型存放在 `assets/models/`，由站点同源加载；“全部开心”模式不调用表情层。
- MediaPipe Tasks Vision 运行时通过 jsDelivr CDN 加载，因此首次打开和首次识别需要网络。
- 两个模型都不可用时，右键手动添加、调整和 PNG 导出仍可使用。

> 表情分类来自面部 blendshape 的规则组合，只适合趣味图片编辑，不代表医学或心理层面的情绪判断。

## 项目结构

```text
Facial_Expression_Emoji_Overlay/
├── index.html                 # 工作台结构与无障碍标签
├── styles.css                # 响应式界面与交互状态
├── app.js                    # 检测、表情规则、画布编辑与导出
├── assets/
│   ├── demo.png              # README 展示图
│   ├── favicon.png           # A1 金毛项目图标，网页与 favicon 共用
│   ├── emoji/                # 后续自定义表情资源预留目录
│   └── models/
│       ├── blaze_face_full_range.tflite
│       ├── face_landmarker.task
│       └── README.md          # 模型来源与校验值
└── README.md
```

## 技术细节

- **人脸检测：** MediaPipe Face Detector + BlazeFace full-range
- **表情与关键点：** MediaPipe Face Landmarker，478 个面部关键点和 blendshape
- **多人合影：** 整图检测后按 4 个尺度做 50% 重叠分块，再通过 IoU 和交集比例去重
- **渲染：** Canvas 2D 统一绘制 Unicode Emoji、局部模糊、像素和纯色遮挡
- **坐标：** 贴图始终保存在原图坐标系；缩放和平移只作用于预览
- **部署：** 纯静态 GitHub Pages，无后端

模型文件的官方来源与 SHA-256 校验值见 [assets/models/README.md](assets/models/README.md)。

## 已知限制

- 侧脸、严重遮挡、模糊或小于约 10–18px 的远处人脸可能漏检。
- Unicode Emoji 的具体外观由操作系统和浏览器决定，预览风格可能不同。
- 为控制耗时，表情层只处理面积最大的 6 张脸，其余人脸默认开心；人脸层最多保留 80 张。
- 当前版本不处理视频、摄像头实时画面、ZIP 打包、MP4 编码或 EXIF 清理。

## 路线图

- [ ] 增加原创或授权明确的“大黄脸”表情资源层
- [ ] 支持更多表情和单脸置信度展示
- [ ] 为触屏设备增加更直观的缩放手柄与删除按钮
- [ ] 支持视频导入、逐帧人脸追踪和 WebM 导出
- [ ] 将 MediaPipe 运行时也改为可选的同源离线加载

## 反馈与贡献

发现漏检、重复贴图或界面问题时，请在 [GitHub Issues](https://github.com/ZhouYinLong-lab/Facial_Expression_Emoji_Overlay/issues) 提交：

- 可复现的图片特征（人数、距离、遮挡情况）
- 浏览器与操作系统版本
- 实际结果和预期结果

欢迎通过 Fork 和 Pull Request 改进检测规则、交互体验或文档。

## 致谢

- [MediaPipe](https://github.com/google-ai-edge/mediapipe) 提供浏览器端视觉任务运行时与官方模型
- [GitHub Pages](https://pages.github.com/) 提供静态站点托管

本仓库目前未声明独立的项目许可证。MediaPipe 及模型文件适用其各自的许可证；复用项目前请先确认授权范围。
