# 论文组图工具 (Figer Paper Tool)

一个专为学术论文设计的图片组合排版工具，基于 PySide6 和 Pillow 开发，提供直观的 GUI 界面和强大的图片布局功能。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![PySide6](https://img.shields.io/badge/GUI-PySide6-brightgreen.svg)

## ✨ 主要特性

- 📐 **专业布局模板**：预设多种符合学术期刊要求的图片布局
- 🖼️ **可视化编辑**：实时预览图片排列效果
- 🎯 **精确控制**：支持毫米级尺寸设置和 DPI 调整
- 🏷️ **自动标注**：自动添加 (a)、(b)、(c) 等子图标注
- 📤 **多格式导出**：支持导出为高质量 PNG 和 PDF 格式
- 🎨 **字体管理**：支持宋体（中文）和 Times New Roman（英文）
- 🔄 **拖拽操作**：支持图片拖拽导入和排序

## 🎯 功能模块

### 默认布局模式
- **1x1**：单图布局
- **1x2**：横向双图布局
- **2x1**：纵向双图布局
- **2x2**：四宫格布局
- **1+2**：一大二小布局
- **2+1**：二小一大布局

### 自定义布局模式
- 网格式画布编辑器
- 支持拖拽、缩放和合并单元格
- 实时预览和精确控制

### 图片处理
- 自动处理 EXIF 旋转信息
- 智能图片缩放和裁剪
- 支持常见图片格式（PNG、JPG、BMP 等）

### 导出设置
- 可调整画布宽度（单位：mm）
- 可调整输出分辨率（DPI）
- 可调整边距和图片间距
- PNG/PDF 双格式导出

## 📥 下载使用（推荐）

**不懂编程？没有Python环境？** 直接下载编译好的可执行文件即可使用！

### Windows 用户
1. 前往 [Releases 页面](https://github.com/cat-dig/figer-paper-tool/releases)
2. 下载最新版本的 `figer-paper-tool.exe` 文件
3. 双击运行，无需安装（首次运行可能需要稍等片刻）

### 特别说明
- ✅ 无需安装 Python
- ✅ 无需安装任何依赖
- ✅ 下载即用
- ⚠️ Windows 可能提示"未知发布者"，请选择"仍要运行"

## 🚀 快速开始（开发者）

### 环境要求

- Python 3.8 或更高版本
- Windows / macOS / Linux

### 安装依赖

```bash
pip install PySide6 Pillow reportlab
```

### 运行程序

```bash
python figer.py
```

或运行论文增强版：

```bash
python figer_paper.py
```

## 📦 打包为可执行文件

项目包含 PowerShell 构建脚本，可将程序打包为独立可执行文件：

```powershell
.\build.ps1
```

构建脚本使用 Nuitka 进行编译，生成的 `.exe` 文件无需 Python 环境即可运行。

## 📖 使用说明

1. **导入图片**：点击"添加图片"或直接拖拽图片到左侧列表
2. **选择布局**：在右侧选择合适的布局模板
3. **调整参数**：设置画布宽度、DPI、边距等参数
4. **预览效果**：在中间区域实时查看组图效果
5. **导出文件**：点击"导出 PNG"或"导出 PDF"保存结果

## 🛠️ 技术栈

- **GUI 框架**：PySide6 (Qt for Python)
- **图像处理**：Pillow (PIL Fork)
- **PDF 生成**：ReportLab
- **打包工具**：Nuitka

## 📁 项目结构

```
figer-paper-tool/
├── figer.py              # 主程序（基础版）
├── figer_paper.py        # 主程序（论文增强版）
├── build.ps1             # Windows 构建脚本
├── README.md             # 项目说明文档
├── .gitignore            # Git 忽略文件配置
└── dist/                 # 构建输出目录（不包含在 Git 中）
```

## 🔧 开发说明

### 字体配置

程序默认使用以下字体：
- **英文**：Times New Roman
- **中文**：宋体 (SimSun)

字体文件会从系统字体目录自动加载。如找不到指定字体，会回退到 PIL 默认字体。

### 布局定义

所有布局通过 `CellSpec` 类定义，使用归一化坐标（0-1）表示位置和大小：

```python
@dataclass
class CellSpec:
    x: float          # 左上角 x 坐标（相对）
    y: float          # 左上角 y 坐标（相对）
    w: float          # 宽度（相对）
    h: float          # 高度（相对）
    # ... 其他属性
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

在提交 PR 之前，请确保：
- 代码符合 PEP 8 规范
- 添加了必要的注释和文档
- 测试了所有主要功能

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 👨‍💻 作者

本工具由研究人员开发，用于提高学术论文图片排版效率。

## 🙏 致谢

感谢以下开源项目：
- [PySide6](https://wiki.qt.io/Qt_for_Python) - Qt for Python
- [Pillow](https://python-pillow.org/) - Python Imaging Library
- [ReportLab](https://www.reportlab.com/) - PDF Generation

---

**如果这个工具对你的研究工作有帮助，请给个 ⭐️ Star！**
