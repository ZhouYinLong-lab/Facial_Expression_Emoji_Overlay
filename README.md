# Emoji Mask · 表情人脸贴图

[![在线体验](https://img.shields.io/badge/GitHub%20Pages-在线体验-2856E8?logo=github)](https://zhouyinlong-lab.github.io/Facial_Expression_Emoji_Overlay/)
[![JavaScript](https://img.shields.io/badge/JavaScript-原生-F7DF1E?logo=javascript&logoColor=111)](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20Vision-00A67E)](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/web_js)
[![隐私](https://img.shields.io/badge/图片处理-仅在浏览器本地-16855B)](#隐私与模型加载)

上传一张合影，自动找到人脸、判断基础表情，并用 Unicode Emoji 遮住每张脸。贴图可以继续移动、缩放、替换，最终按原图尺寸导出 PNG。

**[打开在线版本](https://zhouyinlong-lab.github.io/Facial_Expression_Emoji_Overlay/)** · 无需安装，不会上传照片。

![Emoji Mask 工作台](assets/demo.png)

## 主要功能

- 检测近景人像和多人合影，单次最多保留 30 张脸
- 对面积最大的 6 张脸进一步识别开心、悲伤、生气、惊讶和中性表情
- 未进入表情层或判断不确定的人脸默认使用 `😄`，保证检测到就会被遮住
- 支持拖动、等比例缩放、删除和手动替换单个 Emoji
- 支持滚轮缩放、画布平移和右键手动添加贴图
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
| 人脸层 | 尽量找全合影中的人脸 | 最多 30 张 |
| 表情层 | 细化表情和贴图位置 | 面积最大的前 6 张 |
| 编辑层 | 手动修正识别结果 | Unicode Emoji |

## 使用方法

1. 选择照片，或把图片拖到画布中。
2. 点击“识别并贴图”。多人合影通常需要几秒钟。
3. 点击任意 Emoji 进行选中，拖动可移动，拖动选中边框可缩放。
4. 在左侧表情栏中修正误判；按 `Delete` 或 `Backspace` 删除选中项。
5. 点击“导出 PNG”保存结果。

### 编辑操作

| 操作 | 效果 |
| --- | --- |
| 拖动 Emoji | 移动贴图 |
| 拖动选中边框 | 等比例缩放贴图 |
| 拖动画布空白处 | 平移视图 |
| 鼠标滚轮 | 以指针位置为中心缩放 |
| 图片区域右键 | 手动添加开心 Emoji |
| `Delete` / `Backspace` | 删除选中贴图 |

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
- Face Detector 和 Face Landmarker 模型存放在 `assets/models/`，由站点同源加载。
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
- **渲染：** Canvas 2D `fillText()` 绘制 Unicode Emoji
- **坐标：** 贴图始终保存在原图坐标系；缩放和平移只作用于预览
- **部署：** 纯静态 GitHub Pages，无后端

模型文件的官方来源与 SHA-256 校验值见 [assets/models/README.md](assets/models/README.md)。

## 已知限制

- 侧脸、严重遮挡、模糊或小于约 10–18px 的远处人脸可能漏检。
- Unicode Emoji 的具体外观由操作系统和浏览器决定，预览风格可能不同。
- 为控制耗时，表情层只处理面积最大的 6 张脸，其余人脸默认开心。
- 当前版本不处理视频或摄像头实时画面。

## 路线图

- [ ] 增加原创或授权明确的“大黄脸”表情资源层
- [ ] 支持更多表情和单脸置信度展示
- [ ] 为触屏设备增加更直观的缩放手柄与删除按钮
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
