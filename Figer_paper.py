# -*- coding: utf-8 -*-
"""
论文组图 GUI（PySide6 + Pillow）
- 左侧：图片列表（支持拖拽导入、拖拽排序）
- 中间：实时预览
- 右侧：常用布局 + 论文尺寸参数 + 标注(a)(b)(c) + 导出 PNG/PDF

字体要求：
- 中文：宋体（SimSun）
- 英文：Times New Roman

依赖安装：
pip install PySide6 pillow

运行：
python figure_montage_gui.py
"""

import os
import sys
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from PIL import Image, ImageDraw, ImageFont

from PySide6.QtCore import Qt, QSize, QMimeData, QTimer, QPointF, QRectF, QLineF, Signal, QDataStream, QIODevice
from PySide6.QtGui import QIcon, QPixmap, QImage, QAction, QPen, QBrush, QColor, QPainter
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QListWidget, QListWidgetItem,
    QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFileDialog,
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QMessageBox,
    QGroupBox, QFormLayout, QLineEdit, QSplitter, QToolBar, QMenu,
    QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsLineItem, QStackedWidget, QGraphicsObject,
    QSizePolicy, QRadioButton, QScrollArea
)

APP_TITLE = "论文组图工具（GUI）"


# ----------------------------
# 字体加载（宋体 / Times New Roman）
# ----------------------------
def _try_font(paths: List[str], size: int) -> Optional[ImageFont.FreeTypeFont]:
    for p in paths:
        if p and os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return None


def load_fonts(size_en: int, size_zh: int) -> Tuple[ImageFont.ImageFont, ImageFont.ImageFont, List[str]]:
    """
    返回 (英文字体, 中文字体, 警告信息列表)
    若找不到指定字体，会回退到 PIL 默认字体（会影响论文一致性）。
    """
    warnings = []

    # Windows 常见字体路径
    win_dir = os.environ.get("WINDIR", r"C:\Windows")
    fonts_dir = os.path.join(win_dir, "Fonts")

    # 宋体
    simsun_candidates = [
        os.path.join(fonts_dir, "simsun.ttc"),
        os.path.join(fonts_dir, "simsun.ttf"),
        os.path.join(fonts_dir, "SimSun.ttc"),
        os.path.join(fonts_dir, "SimSun.ttf"),
    ]
    # Times New Roman
    tnr_candidates = [
        os.path.join(fonts_dir, "times.ttf"),
        os.path.join(fonts_dir, "Times.ttf"),
        os.path.join(fonts_dir, "timesbd.ttf"),
        os.path.join(fonts_dir, "Times New Roman.ttf"),
        os.path.join(fonts_dir, "timesi.ttf"),
    ]

    font_en = _try_font(tnr_candidates, size=size_en)
    font_zh = _try_font(simsun_candidates, size=size_zh)

    # macOS / Linux 兜底（不保证一定存在）
    if font_en is None:
        font_en = _try_font(
            [
                "/Library/Fonts/Times New Roman.ttf",
                "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
                "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
                "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
            ],
            size=size_en,
        )

    if font_zh is None:
        font_zh = _try_font(
            [
                "/System/Library/Fonts/STHeiti Light.ttc",  # 仅兜底
                "/Library/Fonts/Songti.ttc",
                "/usr/share/fonts/truetype/arphic/uming.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            ],
            size=size_zh,
        )

    if font_en is None:
        warnings.append("未找到 Times New Roman 字体，将回退到默认字体（不建议用于终稿）。")
        font_en = ImageFont.load_default()

    if font_zh is None:
        warnings.append("未找到 宋体 字体，将回退到默认字体（不建议用于终稿）。")
        font_zh = ImageFont.load_default()

    return font_en, font_zh, warnings


# ----------------------------
# 工具函数
# ----------------------------
def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def pil_to_qpixmap(img: Image.Image) -> Optional[QPixmap]:
    """
    将 PIL Image 转换为 QPixmap。
    注意：必须正确处理内存管理，否则会导致程序闪退。
    """
    try:
        if img is None:
            return None
        
        # 确保图片尺寸有效
        if img.width <= 0 or img.height <= 0:
            return None
        
        # 统一转换为 RGBA 模式以避免格式兼容性问题
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        
        # 获取原始字节数据
        data = img.tobytes("raw", "RGBA")
        
        # 计算每行的字节数（bytes per line / stride）
        # RGBA 每像素 4 字节
        bytes_per_line = img.width * 4
        
        # 创建 QImage，必须传入 bytesPerLine 参数以确保正确的内存对齐
        qimg = QImage(data, img.width, img.height, bytes_per_line, QImage.Format_RGBA8888)
        
        # 关键：必须 copy() 以确保 Qt 拥有数据的副本
        # 否则当 data 被垃圾回收后，QImage 会引用无效内存，导致闪退
        qimg = qimg.copy()
        
        return QPixmap.fromImage(qimg)
    except Exception as e:
        print(f"[ERROR] pil_to_qpixmap failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def safe_open_image(path: str) -> Optional[Image.Image]:
    """
    安全地打开图片文件。
    - 处理 EXIF 旋转
    - 确保图片完全加载到内存
    - 返回 RGB 格式的图片
    """
    try:
        if not path or not os.path.exists(path):
            print(f"[WARN] Image path does not exist: {path}")
            return None
        
        img = Image.open(path)
        
        # 强制加载图片到内存（Image.open 是延迟加载的）
        # 这样可以在这里捕获到损坏图片的异常，而不是在后续操作中
        img.load()
        
        # 处理 EXIF 旋转
        try:
            from PIL import ImageOps
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        
        # 转换为 RGB 格式
        return img.convert("RGB")
    except Exception as e:
        print(f"[ERROR] Failed to open image {path}: {e}")
        return None


# ----------------------------
# 布局定义
# ----------------------------
@dataclass
class CellSpec:
    """一个“格子/区域”的归一化布局（相对于画布宽高 0~1），由渲染时换算成像素。"""
    x: float
    y: float
    w: float
    h: float
    image_offset_x: float = 0.0
    image_offset_y: float = 0.0
    image_scale: float = 1.0
    image_path: Optional[str] = None
    # 独立的标注位置和颜色（如果为None则使用全局设置）
    label_x: Optional[float] = None  # 标注X位置（0-1相对位置）
    label_y: Optional[float] = None  # 标注Y位置（0-1相对位置）
    label_color: Optional[str] = None  # 标注颜色："黑色" / "白色"


def generate_irregular_layout(rows_list: List[int]) -> List[CellSpec]:
    """
    生成不规则的行列布局，自动为每行的图片居中对齐。
    所有子图框的大小完全一致，基于最大列数计算。
    
    Args:
        rows_list: 每行的列数，如 [3, 2] 表示上面3个，下面2个
    
    Returns:
        List[CellSpec]: 布局规格列表
    
    示例:
        generate_irregular_layout([3, 2])  # 第一行3个，第二行2个（第二行居中）
        所有框的大小都是 1/3，高度都是 0.5
        第一行：[0, 0, 1/3, 0.5], [1/3, 0, 1/3, 0.5], [2/3, 0, 1/3, 0.5]
        第二行：[1/6, 0.5, 1/3, 0.5], [1/2, 0.5, 1/3, 0.5]  (居中对齐）
    """
    if not rows_list or all(c == 0 for c in rows_list):
        return []
    
    specs = []
    num_rows = len(rows_list)
    row_height = 1.0 / num_rows
    
    # 关键修改：使用所有行中的最大列数作为统一的列宽基准
    # 这样所有子图框的宽度都一样
    max_cols = max(rows_list)
    unified_col_width = 1.0 / max_cols
    
    for row_idx, num_cols in enumerate(rows_list):
        if num_cols <= 0:
            continue
        
        # 计算此行的起始X位置（居中对齐）
        # 此行实际占用的宽度
        row_used_width = unified_col_width * num_cols
        offset_x = (1.0 - row_used_width) / 2
        
        # 添加此行的各个格子（使用统一的列宽）
        for col_idx in range(num_cols):
            x = offset_x + col_idx * unified_col_width
            y = row_idx * row_height
            specs.append(CellSpec(x, y, unified_col_width, row_height))
    
    return specs


def parse_layout_string(layout_str: str) -> Optional[Tuple]:
    """
    解析用户输入的布局字符串，支持以下格式：
    - "2x2" 或 "2×2": 2行2列网格
    - "[3,2]" 或 "3,2": 不规则布局（第一行3个，第二行2个）
    
    Args:
        layout_str: 用户输入的布局字符串
    
    Returns:
        若为规则网格返回 ('grid', rows, cols)，不规则返回 ('irregular', rows_list)
        解析失败返回 None
    """
    layout_str = layout_str.strip()
    
    # 检查是否为网格格式 (如 "2x2", "2×2")
    if 'x' in layout_str.lower() or '×' in layout_str:
        parts = layout_str.replace('×', 'x').lower().split('x')
        try:
            rows = int(parts[0].strip())
            cols = int(parts[1].strip())
            if rows > 0 and cols > 0:
                return ('grid', rows, cols)
        except (ValueError, IndexError):
            pass
    
    # 检查是否为不规则格式 (如 "[3,2]", "3,2")
    layout_str = layout_str.strip('[]').strip()
    if ',' in layout_str:
        try:
            parts = [int(p.strip()) for p in layout_str.split(',')]
            if all(p > 0 for p in parts):
                return ('irregular', parts)
        except ValueError:
            pass
    
    return None


def build_layout(name: str) -> List[CellSpec]:
    """
    返回一组 CellSpec，顺序对应图片顺序（也对应标注 a,b,c...）
    所有布局都按"常用论文版式"预设，最终画布高度由布局+间距+边距决定。
    
    支持以下格式：
    - 预定义布局名称（如 "2×2 网格"）
    - 自定义布局字符串（如 "2x2", "[3,2]"）
    """
    # 尝试解析自定义布局字符串
    parsed = parse_layout_string(name)
    if parsed:
        if parsed[0] == 'grid':
            rows, cols = parsed[1], parsed[2]
            specs = []
            cell_w = 1.0 / cols
            cell_h = 1.0 / rows
            for r in range(rows):
                for c in range(cols):
                    specs.append(CellSpec(c * cell_w, r * cell_h, cell_w, cell_h))
            return specs
        elif parsed[0] == 'irregular':
            return generate_irregular_layout(parsed[1])
    
    # 自定义网格支持
    if name == "自定义网格":
        # 使用传入的 rows 和 cols 参数（需要在调用时传递）
        # 这里提供默认值，实际值由MainWindow传递
        rows = getattr(build_layout, '_custom_rows', 2)
        cols = getattr(build_layout, '_custom_cols', 2)
        specs = []
        cell_w = 1.0 / cols
        cell_h = 1.0 / rows
        for r in range(rows):
            for c in range(cols):
                specs.append(CellSpec(c * cell_w, r * cell_h, cell_w, cell_h))
        return specs
    
    if name == "2×2 网格":
        # 2行2列
        return [
            CellSpec(0.0, 0.0, 0.5, 0.5),
            CellSpec(0.5, 0.0, 0.5, 0.5),
            CellSpec(0.0, 0.5, 0.5, 0.5),
            CellSpec(0.5, 0.5, 0.5, 0.5),
        ]
    if name == "3×2 网格":
        # 2行3列（更常见：6张对比）
        return [
            CellSpec(0.0, 0.0, 1/3, 0.5),
            CellSpec(1/3, 0.0, 1/3, 0.5),
            CellSpec(2/3, 0.0, 1/3, 0.5),
            CellSpec(0.0, 0.5, 1/3, 0.5),
            CellSpec(1/3, 0.5, 1/3, 0.5),
            CellSpec(2/3, 0.5, 1/3, 0.5),
        ]
    if name == "1+2（上大下两小）":
        # 上面 1 张跨全宽；下面 2 张各占半宽
        return [
            CellSpec(0.0, 0.0, 1.0, 0.6),
            CellSpec(0.0, 0.6, 0.5, 0.4),
            CellSpec(0.5, 0.6, 0.5, 0.4),
        ]
    if name == "2+1（上两小下大）":
        return [
            CellSpec(0.0, 0.0, 0.5, 0.4),
            CellSpec(0.5, 0.0, 0.5, 0.4),
            CellSpec(0.0, 0.4, 1.0, 0.6),
        ]
    if name == "左大右两小":
        return [
            CellSpec(0.0, 0.0, 0.6, 1.0),
            CellSpec(0.6, 0.0, 0.4, 0.5),
            CellSpec(0.6, 0.5, 0.4, 0.5),
        ]
    if name == "右大左两小":
        return [
            CellSpec(0.4, 0.0, 0.6, 1.0),
            CellSpec(0.0, 0.0, 0.4, 0.5),
            CellSpec(0.0, 0.5, 0.4, 0.5),
        ]
    # 默认
    return build_layout("1+2（上大下两小）")


# ----------------------------
# 交互式画布编辑器
# ----------------------------
@dataclass
class PanelSpec:
    """子图框模型（单位：mm）"""
    id: str
    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float
    image_path: Optional[str] = None
    label_index: int = 0  # 用于 (a)(b)(c) 顺序
    lock_aspect: bool = True
    lock_position: bool = False
    # 图片在框内的调整参数
    image_offset_x: float = 0.0  # 图片在框内的水平偏移（相对于框宽的比例，-1~1）
    image_offset_y: float = 0.0  # 图片在框内的垂直偏移（相对于框高的比例，-1~1）
    image_scale: float = 1.0     # 图片在框内的缩放比例（>0，1.0为默认填满）
    # 独立的标注位置和颜色（如果为None则使用全局设置）
    label_x: Optional[float] = None  # 标注X位置（0-1相对位置）
    label_y: Optional[float] = None  # 标注Y位置（0-1相对位置）
    label_color: Optional[str] = None  # 标注颜色："黑色" / "白色"
    label_y: Optional[float] = None  # 标注Y位置（0-1相对位置）
    label_color: Optional[str] = None  # 标注颜色："黑色" / "白色"

@dataclass
class CanvasSpec:
    """画布模型（单位：mm）"""
    width_mm: float
    height_mm: float
    margin_mm: float
    gap_mm: float
    panels: List[PanelSpec]


@dataclass
class GridCell:
    """单元格数据"""
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    image_path: Optional[str] = None
    is_merged: bool = False


class GridManager:
    """网格管理器"""
    
    def __init__(self, rows: int, cols: int, width: float, height: float):
        self.rows = rows
        self.cols = cols
        self.width = width
        self.height = height
        self.cells: Dict[Tuple[int, int], GridCell] = {}
        self._init_cells()
    
    def _init_cells(self):
        self.cells.clear()
        for r in range(self.rows):
            for c in range(self.cols):
                self.cells[(r, c)] = GridCell(r, c)
    
    def set_grid_size(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self._init_cells()
    
    def get_cell_rect(self, row: int, col: int, row_span: int = 1, col_span: int = 1) -> QRectF:
        cell_w = self.width / self.cols
        cell_h = self.height / self.rows
        return QRectF(col * cell_w, row * cell_h, col_span * cell_w, row_span * cell_h)
    
    def find_cell_at(self, pos: QPointF) -> Optional[Tuple[int, int]]:
        col = int(pos.x() / (self.width / self.cols))
        row = int(pos.y() / (self.height / self.rows))
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return (row, col)
        return None
    
    def merge_cells(self, start_row: int, start_col: int, end_row: int, end_col: int) -> bool:
        if not (0 <= start_row <= end_row < self.rows and 0 <= start_col <= end_col < self.cols):
            return False
        for r in range(start_row, end_row + 1):
            for c in range(start_col, end_col + 1):
                if self.cells[(r, c)].is_merged:
                    return False
        main_cell = self.cells[(start_row, start_col)]
        main_cell.row_span = end_row - start_row + 1
        main_cell.col_span = end_col - start_col + 1
        for r in range(start_row, end_row + 1):
            for c in range(start_col, end_col + 1):
                if r != start_row or c != start_col:
                    self.cells[(r, c)].is_merged = True
        return True
    
    def unmerge_cell(self, row: int, col: int):
        cell = self.cells.get((row, col))
        if not cell:
            return
        for r in range(row, row + cell.row_span):
            for c in range(col, col + cell.col_span):
                self.cells[(r, c)].is_merged = False
        cell.row_span = 1
        cell.col_span = 1
    
    def export_to_cellspecs(self) -> List[CellSpec]:
        specs = []
        for (r, c), cell in self.cells.items():
            if cell.is_merged:
                continue
            rect = self.get_cell_rect(r, c, cell.row_span, cell.col_span)
            specs.append(CellSpec(
                x=rect.x() / self.width,
                y=rect.y() / self.height,
                w=rect.width() / self.width,
                h=rect.height() / self.height
            ))
        return specs


class ResizeHandle(QGraphicsRectItem):
    """8个方向的缩放句柄"""
    HANDLE_SIZE = 8
    
    def __init__(self, handle_pos: str, parent=None):
        super().__init__(-self.HANDLE_SIZE/2, -self.HANDLE_SIZE/2, self.HANDLE_SIZE, self.HANDLE_SIZE, parent)
        self.handle_pos = handle_pos  # 'tl', 'tr', 'bl', 'br', 't', 'b', 'l', 'r'
        self.setBrush(QBrush(QColor(100, 150, 255)))
        self.setPen(QPen(QColor(255, 255, 255), 1))
        self.setZValue(100)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setCursor(self._get_cursor())
    
    def _get_cursor(self):
        if self.handle_pos in ['tl', 'br']: return Qt.CursorShape.SizeFDiagCursor
        if self.handle_pos in ['tr', 'bl']: return Qt.CursorShape.SizeBDiagCursor
        if self.handle_pos in ['t', 'b']: return Qt.CursorShape.SizeVerCursor
        if self.handle_pos in ['l', 'r']: return Qt.CursorShape.SizeHorCursor
        return Qt.CursorShape.SizeAllCursor

    def mouseMoveEvent(self, event):
        # 实际缩放逻辑在父级 PanelItem 处理
        self.parentItem().handle_move(self, event.pos())

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        # 通知父级面板触发布局变更，驱动实时预览刷新
        parent = self.parentItem()
        if isinstance(parent, PanelItem):
            parent.layoutChanged.emit()

class SnapGuideLine(QGraphicsLineItem):
    """吸附辅助线"""
    def __init__(self, scene):
        super().__init__()
        self.setPen(QPen(QColor(255, 0, 0), 0.5, Qt.PenStyle.DashLine))
        self.setZValue(2000)
        scene.addItem(self)
        self.hide()

class LabelHandle(QGraphicsRectItem):
    """可拖拽的标注位置控制器"""
    HANDLE_SIZE = 6
    
    def __init__(self, parent=None):
        super().__init__(-self.HANDLE_SIZE/2, -self.HANDLE_SIZE/2, self.HANDLE_SIZE, self.HANDLE_SIZE, parent)
        self.setBrush(QBrush(QColor(255, 100, 100)))
        self.setPen(QPen(QColor(255, 255, 255), 1))
        self.setZValue(200)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setVisible(False)
    
    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        # 限制在父级范围内
        parent = self.parentItem()
        if parent and isinstance(parent, PanelItem):
            rect = parent.boundingRect()
            pos = self.pos()
            x = max(0, min(rect.width(), pos.x()))
            y = max(0, min(rect.height(), pos.y()))
            self.setPos(x, y)
            
            # 强制父级重绘以更新文字位置
            if parent:
                parent.update()
    
    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        # 通知主窗口同步标注位置
        parent = self.parentItem()
        if parent and isinstance(parent, PanelItem):
            scene = self.scene()
            if scene and hasattr(scene, 'views'):
                for view in scene.views():
                    widget = view.parent()
                    if widget and hasattr(widget, 'sync_label_position'):
                        widget.sync_label_position(self.pos().x() / parent.spec.w_mm, 
                                                   self.pos().y() / parent.spec.h_mm)


class PanelItem(QGraphicsObject):
    """可拖拽、缩放的子图框（单位为 mm）"""
    layoutChanged = Signal()
    
    def __init__(self, spec: PanelSpec):
        super().__init__()
        self.spec = spec
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)  # 启用悬停事件
        
        self.pixmap = None
        if spec.image_path:
            self._load_image(spec.image_path)
        
        # 图片编辑模式
        self.image_edit_mode = False  # 是否处于图片编辑模式
        self.dragging_image = False   # 是否正在拖拽图片
        self.last_mouse_pos = None
        
        self.handles = {}
        for pos in ['tl', 'tr', 'bl', 'br', 't', 'b', 'l', 'r']:
            h = ResizeHandle(pos, self)
            self.handles[pos] = h
        self._update_handles_pos()
        self._show_handles(False)
        
        # 添加标注位置控制器
        self.label_handle = LabelHandle(self)
        self._update_label_handle_pos()

    def _get_main_window(self):
        """向上查找主窗口，便于读取标注模式等全局配置。"""
        # 优先通过场景→视图→父级链条寻找
        if self.scene():
            for view in self.scene().views():
                widget = view.parent()
                while widget:
                    if hasattr(widget, "cmb_label_pos"):
                        return widget
                    widget = widget.parent()
        # 兜底：使用当前激活窗口
        try:
            from PySide6.QtWidgets import QApplication
            win = QApplication.activeWindow()
            if win and hasattr(win, "cmb_label_pos"):
                return win
        except Exception:
            pass
        return None

    def _load_image(self, path):
        """安全加载图片到 pixmap"""
        try:
            img = safe_open_image(path)
            if img:
                img.thumbnail((800, 800))
                self.pixmap = pil_to_qpixmap(img)
            else:
                self.pixmap = None
        except Exception as e:
            print(f"[ERROR] Failed to load image in PanelItem: {path}, error: {e}")
            self.pixmap = None

    def boundingRect(self):
        return QRectF(0, 0, self.spec.w_mm, self.spec.h_mm)

    def paint(self, painter, option, widget):
        rect = self.boundingRect()
        
        # 背景（白色）
        painter.fillRect(rect, Qt.GlobalColor.white)
        
        # 图片 - 使用裁剪填满模式
        if self.pixmap:
            pw, ph = self.pixmap.width(), self.pixmap.height()
            if pw > 0 and ph > 0:
                # 1. 计算填满框所需的基础缩放比例（取max确保填满）
                base_scale = max(rect.width()/pw, rect.height()/ph)
                
                # 2. 应用用户的额外缩放
                total_scale = base_scale * self.spec.image_scale
                
                # 3. 计算缩放后的图片尺寸
                nw, nh = pw * total_scale, ph * total_scale
                
                # 4. 默认居中，然后应用用户偏移
                # 偏移量是相对于框尺寸的比例
                center_x = (rect.width() - nw) / 2
                center_y = (rect.height() - nh) / 2
                
                offset_x = self.spec.image_offset_x * rect.width()
                offset_y = self.spec.image_offset_y * rect.height()
                
                x = center_x + offset_x
                y = center_y + offset_y
                
                # 5. 绘制图片（超出框的部分会被自动裁剪）
                painter.setClipRect(rect)  # 确保裁剪
                painter.drawPixmap(QRectF(x, y, nw, nh), self.pixmap, QRectF(self.pixmap.rect()))
                painter.setClipping(False)
        
        # 边框
        pen_color = QColor(100, 150, 255) if self.isSelected() else QColor(180, 180, 180)
        # 图片编辑模式时使用特殊颜色
        if self.image_edit_mode:
            pen_color = QColor(255, 100, 100)  # 红色表示编辑模式
            pen_width = 1.0
        else:
            pen_width = 0.5
        painter.setPen(QPen(pen_color, pen_width))
        painter.drawRect(rect)
        
        # 图片编辑模式提示
        if self.image_edit_mode:
            painter.setPen(QColor(255, 100, 100))
            font = painter.font()
            font.setPixelSize(3)
            painter.setFont(font)
            painter.drawText(rect.adjusted(1, 1, -1, -1),
                           Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                           "编辑模式")
        
        # 标签 (a, b, c...)
        if self.spec.label_index >= 0:
            label = chr(ord('a') + self.spec.label_index)
            
            # 获取标注颜色（优先使用子图独立设置，否则使用全局设置）
            label_color = self.spec.label_color
            if label_color is None:
                host = self._get_main_window()
                if host and hasattr(host, 'cmb_label_color'):
                    label_color = host.cmb_label_color.currentText()
                else:
                    label_color = "黑色"
            
            # 设置标注颜色
            if label_color == "白色":
                painter.setPen(Qt.GlobalColor.white)
            else:
                painter.setPen(Qt.GlobalColor.black)
            
            font = painter.font()
            font.setPixelSize(4) # 4mm 左右
            painter.setFont(font)
            
            # 获取当前标注位置模式
            label_mode = "左上"
            host = self._get_main_window()
            if host and hasattr(host, 'cmb_label_pos'):
                label_mode = host.cmb_label_pos.currentText()
            
            # 获取标注位置（优先使用子图独立设置）
            if self.spec.label_x is not None and self.spec.label_y is not None:
                # 使用子图独立的标注位置
                x_ratio = self.spec.label_x
                y_ratio = self.spec.label_y
                pos_x = rect.width() * x_ratio
                pos_y = rect.height() * y_ratio
                painter.drawText(QRectF(pos_x, pos_y, rect.width(), rect.height()), 
                                 Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, 
                                 f"({label})")
            elif label_mode == "自定义":
                # 自定义模式沿用主界面滑块参数
                x_ratio = 0.05
                y_ratio = 0.05
                if host and hasattr(host, 'slider_label_x') and hasattr(host, 'slider_label_y'):
                    x_ratio = host.slider_label_x.value() / 100.0
                    y_ratio = host.slider_label_y.value() / 100.0
                pos_x = rect.width() * x_ratio
                pos_y = rect.height() * y_ratio
                painter.drawText(QRectF(pos_x, pos_y, rect.width(), rect.height()), 
                                 Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, 
                                 f"({label})")
            elif label_mode == "左下":
                painter.drawText(rect.adjusted(1, 1, -1, -1), 
                                 Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom, 
                                 f"({label})")
            else:
                # 默认 左上
                painter.drawText(rect.adjusted(1, 1, -1, -1), 
                                 Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, 
                                 f"({label})")

    def _update_handles_pos(self):
        rect = self.boundingRect()
        self.handles['tl'].setPos(rect.topLeft())
        self.handles['tr'].setPos(rect.topRight())
        self.handles['bl'].setPos(rect.bottomLeft())
        self.handles['br'].setPos(rect.bottomRight())
        self.handles['t'].setPos(rect.center().x(), rect.top())
        self.handles['b'].setPos(rect.center().x(), rect.bottom())
        self.handles['l'].setPos(rect.left(), rect.center().y())
        self.handles['r'].setPos(rect.right(), rect.center().y())

    def _show_handles(self, show: bool):
        for h in self.handles.values():
            h.setVisible(show)
        # 标注手柄仅在选中且自定义模式下显示
        if hasattr(self, 'label_handle'):
            # 取消拖拽红色方块的交互，统一用数值调节
            self.label_handle.setVisible(False)

    def _hide_guides(self):
        if hasattr(self, '_guide_h'):
            self._guide_h.hide()
        if hasattr(self, '_guide_v'):
            self._guide_v.hide()
    
    def _is_label_custom_mode(self) -> bool:
        """检查是否为标注自定义模式"""
        host = self._get_main_window()
        if host and hasattr(host, 'cmb_label_pos'):
            return host.cmb_label_pos.currentText() == "自定义"
        return False
    
    def _update_label_handle_pos(self, x_ratio: float = 0.05, y_ratio: float = 0.05):
        """更新标注手柄位置"""
        if hasattr(self, 'label_handle'):
            rect = self.boundingRect()
            self.label_handle.setPos(rect.width() * x_ratio, rect.height() * y_ratio)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
            self._show_handles(value)
            if not value:
                self._hide_guides()
        elif change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # 1. 基础网格吸附
            new_pos = value
            x = round(new_pos.x())
            y = round(new_pos.y())
            
            # 2. 智能辅助线吸附
            if self.scene() and self.isSelected():
                x, y = self.perform_snap(x, y)
            
            # 同步更新 spec 的 x, y (mm)
            self.spec.x_mm = x
            self.spec.y_mm = y
            return QPointF(x, y)
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event):
        self._hide_guides()
        super().mouseReleaseEvent(event)
        self.layoutChanged.emit()
        
    def perform_snap(self, x, y):
        """执行吸附并显示辅助线"""
        SNAP_DIST = 1.0 # mm 吸附阈值
        
        # 延迟初始化辅助线
        if not hasattr(self, '_guide_h'):
            self._guide_h = SnapGuideLine(self.scene())
        if not hasattr(self, '_guide_v'):
            self._guide_v = SnapGuideLine(self.scene())
        
        self._guide_h.hide()
        self._guide_v.hide()
        
        # 收集其他 items 的关键坐标
        xs = set() # 垂线 X 坐标
        ys = set() # 水平线 Y 坐标
        
        # 画布边界 (尝试获取 MarginRect)
        # 通过 sceneRect 估算也可以，或者就不对齐画布边缘了，只对齐其他 item
        
        for item in self.scene().items():
            if isinstance(item, PanelItem) and item != self:
                ix = item.pos().x()
                iy = item.pos().y()
                iw = item.spec.w_mm
                ih = item.spec.h_mm
                xs.add(ix)
                xs.add(ix + iw)
                xs.add(ix + iw/2)
                ys.add(iy)
                ys.add(iy + ih)
                ys.add(iy + ih/2)

        w = self.spec.w_mm
        h = self.spec.h_mm
        
        # --- 检测 X 轴（垂线） ---
        snapped_x = x
        snap_target_x = None
        # 定义关键点偏移：(当前关键点相对于左上角的偏移, 自身宽度补偿)
        # 左边缘对齐: x 变 target -> offset=0
        # 右边缘对齐: x+w 变 target -> x 变 target-w -> offset=-w
        # 中轴对齐: x+w/2 变 target -> x 变 target-w/2 -> offset=-w/2
        candidates = [
            (x, 0),        
            (x + w, -w),   
            (x + w/2, -w/2)
        ]
        
        for curr, offset in candidates:
            # 找最近的吸附点
            best_dist = SNAP_DIST
            for target in xs:
                dist = abs(curr - target)
                if dist < best_dist:
                    best_dist = dist
                    snapped_x = target + offset
                    snap_target_x = target
            if snap_target_x is not None: break # 找到一个吸附点就停止
        
        # --- 检测 Y 轴（水平线） ---
        snapped_y = y
        snap_target_y = None
        candidates_y = [
            (y, 0),
            (y + h, -h),
            (y + h/2, -h/2)
        ]
        for curr, offset in candidates_y:
            best_dist = SNAP_DIST
            for target in ys:
                dist = abs(curr - target)
                if dist < best_dist:
                    best_dist = dist
                    snapped_y = target + offset
                    snap_target_y = target
            if snap_target_y is not None: break
            
        # 绘制辅助线 (延伸到整个 Scene)
        scene_rect = self.scene().sceneRect()
        if snap_target_x is not None:
             self._guide_v.setLine(snap_target_x, scene_rect.top(), snap_target_x, scene_rect.bottom())
             self._guide_v.show()
        
        if snap_target_y is not None:
             self._guide_h.setLine(scene_rect.left(), snap_target_y, scene_rect.right(), snap_target_y)
             self._guide_h.show()
             
        return snapped_x, snapped_y

    def handle_move(self, handle, pos):
        # 缩放逻辑
        rect = self.boundingRect()
        handle_pos = handle.handle_pos
        delta = handle.mapToParent(pos) - handle.pos()
        
        new_w, new_h = self.spec.w_mm, self.spec.h_mm
        new_x, new_y = self.spec.x_mm, self.spec.y_mm
        
        # 简单的缩放（先不考虑比例锁定，后续按 Alt 切换）
        if 'r' in handle_pos: new_w += delta.x()
        if 'l' in handle_pos: 
            new_w -= delta.x()
            new_x += delta.x()
        if 'b' in handle_pos: new_h += delta.y()
        if 't' in handle_pos:
            new_h -= delta.y()
            new_y += delta.y()
            
        # 最小尺寸限制 2mm
        if new_w < 2: new_w = 2
        if new_h < 2: new_h = 2
        
        self.prepareGeometryChange()
        self.spec.w_mm = new_w
        self.spec.h_mm = new_h
        self.setPos(new_x, new_y)
        self._update_handles_pos()
        self.update()
    
    def mouseDoubleClickEvent(self, event):
        """双击切换图片编辑模式"""
        if self.pixmap is None:
            super().mouseDoubleClickEvent(event)
            return
        
        self.image_edit_mode = not self.image_edit_mode
        
        if self.image_edit_mode:
            # 进入编辑模式：锁定框架移动
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
            self._show_handles(False)  # 隐藏缩放手柄
        else:
            # 退出编辑模式：恢复框架移动
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
            if self.isSelected():
                self._show_handles(True)
        
        self.update()
        event.accept()
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if self.image_edit_mode and event.button() == Qt.MouseButton.LeftButton:
            # 编辑模式下，开始拖拽图片
            self.dragging_image = True
            self.last_mouse_pos = event.pos()
            event.accept()
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.image_edit_mode and self.dragging_image and self.last_mouse_pos:
            # 计算鼠标移动距离
            delta = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()
            
            # 将像素移动转换为相对偏移比例
            # delta是在item坐标系下的，单位是mm
            rect = self.boundingRect()
            offset_delta_x = delta.x() / rect.width()
            offset_delta_y = delta.y() / rect.height()
            
            # 更新偏移
            self.spec.image_offset_x += offset_delta_x
            self.spec.image_offset_y += offset_delta_y
            
            # 限制偏移范围（可选）
            # self.spec.image_offset_x = max(-2.0, min(2.0, self.spec.image_offset_x))
            # self.spec.image_offset_y = max(-2.0, min(2.0, self.spec.image_offset_y))
            
            self.update()
            event.accept()
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if self.image_edit_mode and self.dragging_image:
            self.dragging_image = False
            self.last_mouse_pos = None
            event.accept()
        else:
            self._hide_guides()
            super().mouseReleaseEvent(event)
        self.layoutChanged.emit()
    
    def wheelEvent(self, event):
        """滚轮事件 - 缩放图片"""
        if self.image_edit_mode and self.pixmap:
            # 滚轮缩放
            delta = event.angleDelta().y()
            scale_factor = 1.1 if delta > 0 else 0.9
            
            new_scale = self.spec.image_scale * scale_factor
            # 限制缩放范围
            new_scale = max(0.1, min(10.0, new_scale))
            
            self.spec.image_scale = new_scale
            self.update()
            event.accept()
        else:
            super().wheelEvent(event)
    
    def hoverEnterEvent(self, event):
        """鼠标悬停进入"""
        if self.image_edit_mode:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        """鼠标悬停离开"""
        self.unsetCursor()
        super().hoverLeaveEvent(event)


class CanvasEditorWidget(QWidget):
    """真·交互式画布编辑器 (单位: mm)"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas_w = 180  # 默认双栏宽度
        self.canvas_h = 120
        self.margin = 5
        
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.view.setBackgroundBrush(QBrush(QColor(240, 240, 240)))
        self.view.setAcceptDrops(True)
        self.view.viewport().setAcceptDrops(True)
        self._paper_rect = None
        
        # 安装事件过滤器以拦截 view/viewport 的拖拽事件
        self.view.installEventFilter(self)
        self.view.viewport().installEventFilter(self)
        
        layout = QVBoxLayout(self)
        
        # 顶部工具栏 (简化版)
        self.toolbar = QToolBar()
        self.btn_add = self.toolbar.addAction("新建面板")
        self.btn_add.triggered.connect(self.add_new_panel)
        self.btn_del = self.toolbar.addAction("删除")
        self.btn_del.triggered.connect(self.delete_selected)
        self.toolbar.addSeparator()
        self.btn_align_left = self.toolbar.addAction("左对齐")
        self.btn_align_left.triggered.connect(lambda: self.align_selected("left"))
        self.btn_align_right = self.toolbar.addAction("右对齐")
        self.btn_align_right.triggered.connect(lambda: self.align_selected("right"))
        self.btn_align_top = self.toolbar.addAction("上对齐")
        self.btn_align_top.triggered.connect(lambda: self.align_selected("top"))
        self.btn_align_bottom = self.toolbar.addAction("下对齐")
        self.btn_align_bottom.triggered.connect(lambda: self.align_selected("bottom"))
        self.toolbar.addSeparator()
        self.btn_align_hcenter = self.toolbar.addAction("水平居中")
        self.btn_align_hcenter.triggered.connect(lambda: self.align_selected("hcenter"))
        self.btn_align_vcenter = self.toolbar.addAction("垂直居中")
        self.btn_align_vcenter.triggered.connect(lambda: self.align_selected("vcenter"))
        
        self.toolbar.addSeparator()
        self.btn_dist_h = self.toolbar.addAction("水平分布")
        self.btn_dist_h.triggered.connect(lambda: self.distribute_selected("h"))
        self.btn_dist_v = self.toolbar.addAction("垂直分布")
        self.btn_dist_v.triggered.connect(lambda: self.distribute_selected("v"))
        
        self.toolbar.addSeparator()
        
        self.btn_apply_gap = self.toolbar.addAction("应用间距")
        self.btn_apply_gap.setToolTip("将选中的面板按右侧配置的间距(gap)从左到右/从上到下排列")
        self.btn_apply_gap.triggered.connect(self.apply_gap_to_selected)

        self.btn_set_first = self.toolbar.addAction("设为 (a)")
        self.btn_set_first.setToolTip("选中一个面板并将其设为 (a)，其余自动按空间顺序递增")
        self.btn_set_first.triggered.connect(self.set_selected_as_first)

        self.btn_smart_h = self.toolbar.addAction("智能调整高度")
        self.btn_smart_h.triggered.connect(self.smart_adjust_height)

        # 第二行工具栏：图片取景调整（始终可见）
        self.toolbar2 = QToolBar()
        self.toolbar2.addWidget(QLabel("图片取景："))
        self.btn_img_left = self.toolbar2.addAction("←")
        self.btn_img_left.setToolTip("图片左移")
        self.btn_img_left.triggered.connect(lambda: self.adjust_image_offset(dx=-0.02, dy=0))
        self.btn_img_right = self.toolbar2.addAction("→")
        self.btn_img_right.setToolTip("图片右移")
        self.btn_img_right.triggered.connect(lambda: self.adjust_image_offset(dx=0.02, dy=0))
        self.btn_img_up = self.toolbar2.addAction("↑")
        self.btn_img_up.setToolTip("图片上移")
        self.btn_img_up.triggered.connect(lambda: self.adjust_image_offset(dx=0, dy=-0.02))
        self.btn_img_down = self.toolbar2.addAction("↓")
        self.btn_img_down.setToolTip("图片下移")
        self.btn_img_down.triggered.connect(lambda: self.adjust_image_offset(dx=0, dy=0.02))
        self.toolbar2.addSeparator()
        self.btn_zoom_in = self.toolbar2.addAction("放大")
        self.btn_zoom_in.setToolTip("图片放大")
        self.btn_zoom_in.triggered.connect(lambda: self.adjust_image_scale(1.1))
        self.btn_zoom_out = self.toolbar2.addAction("缩小")
        self.btn_zoom_out.setToolTip("图片缩小")
        self.btn_zoom_out.triggered.connect(lambda: self.adjust_image_scale(0.9))
        self.btn_reset_img = self.toolbar2.addAction("重置")
        self.btn_reset_img.setToolTip("重置图片位置和缩放")
        self.btn_reset_img.triggered.connect(self.reset_image_adjustment)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.toolbar2)
        layout.addWidget(self.view)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.setAcceptDrops(True)
        self._init_scene()
        self._fit_view()
    
    def eventFilter(self, watched, event):
        """拦截 QGraphicsView/Viewport 的拖拽事件并处理。"""
        from PySide6.QtCore import QEvent
        if watched in (self.view, self.view.viewport()):
            if event.type() == QEvent.Type.DragEnter:
                if event.mimeData().hasFormat("application/x-qabstractitemmodeldatalist") or event.mimeData().hasUrls():
                    event.acceptProposedAction()
                    return True
            elif event.type() == QEvent.Type.DragMove:
                event.acceptProposedAction()
                return True
            elif event.type() == QEvent.Type.Drop:
                self._handle_drop(event)
                return True
        return super().eventFilter(watched, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 视口尺寸变化后重新适配
        self._fit_view()
    
    def sync_label_position(self, x_ratio: float, y_ratio: float):
        """同步所有面板的标注位置，并更新主窗口配置"""
        # 更新所有面板的标注手柄位置
        for item in self.scene.items():
            if isinstance(item, PanelItem):
                item._update_label_handle_pos(x_ratio, y_ratio)
        
        # 更新主窗口的滑块值
        if hasattr(self.window(), 'slider_label_x'):
            self.window().slider_label_x.blockSignals(True)
            self.window().slider_label_x.setValue(x_ratio * 100)
            self.window().slider_label_x.blockSignals(False)
        
        if hasattr(self.window(), 'slider_label_y'):
            self.window().slider_label_y.blockSignals(True)
            self.window().slider_label_y.setValue(y_ratio * 100)
            self.window().slider_label_y.blockSignals(False)
        
        # 触发预览刷新
        if hasattr(self.window(), 'schedule_preview'):
            self.window().schedule_preview()

    def smart_adjust_height(self):
        """根据面板布局自动调整画布高度"""
        items = [i for i in self.scene.items() if isinstance(i, PanelItem)]
        if not items: return
        
        # 计算所有 panel 的外避范围
        max_y = 0
        for i in items:
            max_y = max(max_y, i.pos().y() + i.spec.h_mm)
            
        # 推荐高度 = 面板最大底部 + 下边距
        recommended_h = max_y + self.margin
        
        # 通知 MainWindow 更新 UI
        if hasattr(self.window(), "sp_height"):
            self.window().sp_height.setValue(recommended_h)
            # 这会触发 on_canvas_size_changed -> set_canvas_size

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-qabstractitemmodeldatalist") or event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        self._handle_drop(event)
    
    def _handle_drop(self, event):
        """统一处理拖拽放下事件。"""
        # 获取拖入的图片路径（支持文件拖拽和列表内部拖拽）
        path = self._extract_path_from_mime(event.mimeData())

        if not path or not os.path.exists(path):
            return

        # 获取相对于 viewport 的位置并映射到场景坐标
        if hasattr(event, 'position'):
            # Qt6 / PySide6 使用 position()
            local_pos = event.position().toPoint()
        else:
            local_pos = event.pos()
        scene_pos = self.view.mapToScene(local_pos)
        target_item = self.scene.itemAt(scene_pos, self.view.transform())
        
        # 寻找最近的 PanelItem (避开 handles)
        panel = None
        if isinstance(target_item, PanelItem):
            panel = target_item
        elif isinstance(target_item, ResizeHandle):
            panel = target_item.parentItem()
        
        if panel and isinstance(panel, PanelItem):
            # 替换图片
            panel.spec.image_path = path
            panel._load_image(path)
            panel.update()
        else:
            # 在空白处创建
            item = self.add_new_panel(path)
            # 对齐到 1mm 网格
            item.setPos(round(scene_pos.x()), round(scene_pos.y()))
            item.spec.x_mm = round(scene_pos.x())
            item.spec.y_mm = round(scene_pos.y())
            
            # 重排标签
            self.reorder_labels()
        
        event.acceptProposedAction()
        if hasattr(self.window(), 'schedule_preview'):
            self.window().schedule_preview()

    def _extract_path_from_mime(self, mime: QMimeData) -> Optional[str]:
        """从拖拽数据中提取首个本地图片路径，支持 URL 和 QListWidget 内部拖拽。"""
        # 1) 直接文件拖入
        if mime.hasUrls():
            urls = mime.urls()
            for u in urls:
                if u.isLocalFile():
                    return u.toLocalFile()
        
        # 2) 列表内部拖拽（application/x-qabstractitemmodeldatalist）
        fmt = "application/x-qabstractitemmodeldatalist"
        if mime.hasFormat(fmt):
            data = mime.data(fmt)
            stream = QDataStream(data, QIODevice.ReadOnly)
            while not stream.atEnd():
                row = stream.readInt32()
                col = stream.readInt32()
                map_items = stream.readInt32()
                for _ in range(map_items):
                    role = stream.readInt32()
                    value = stream.readQVariant()
                    if role == int(Qt.ItemDataRole.UserRole):
                        try:
                            text = str(value)
                        except Exception:
                            text = None
                        if text:
                            return text
        
        # 3) 兜底：当前列表选中项
        if hasattr(self.window(), "list_widget"):
            try:
                items = self.window().list_widget.selectedItems()
                if items:
                    return items[0].data(Qt.ItemDataRole.UserRole)
            except RuntimeError:
                pass
        return None

    def _init_scene(self):
        self.scene.clear()
        self.scene.setSceneRect(-10, -10, self.canvas_w + 20, self.canvas_h + 20)
        
        # 绘制纸张（白底）
        paper = self.scene.addRect(0, 0, self.canvas_w, self.canvas_h, QPen(Qt.GlobalColor.black, 0.2), QBrush(Qt.GlobalColor.white))
        paper.setZValue(-10)
        self._paper_rect = paper
        
        # 绘制页边距参考线
        margin_rect = self.scene.addRect(self.margin, self.margin, 
                                        self.canvas_w - 2*self.margin, 
                                        self.canvas_h - 2*self.margin,
                                        QPen(QColor(220, 220, 220), 0.2, Qt.PenStyle.DashLine))
        margin_rect.setZValue(-9)
        
        # 监听选中事件
        self.scene.selectionChanged.connect(self._on_selection_changed)
    
    def _on_selection_changed(self):
        """选中项改变时更新主窗口的单个子图标注控件"""
        selected_items = [item for item in self.scene.selectedItems() if isinstance(item, PanelItem)]
        
        # 获取主窗口
        main_window = self.window()
        if not main_window or not hasattr(main_window, 'chk_use_single_label'):
            return
        
        if len(selected_items) == 1:
            # 只有一个面板被选中时，启用复选框并更新UI
            panel = selected_items[0]
            
            # 启用复选框
            main_window.chk_use_single_label.setEnabled(True)
            main_window.chk_use_single_label.setToolTip("勾选后可以为此子图单独设置标注位置和颜色")
            
            # 检查是否有独立的标注设置
            has_custom_label = (panel.spec.label_x is not None and 
                              panel.spec.label_y is not None)
            
            # 更新复选框状态
            main_window.chk_use_single_label.blockSignals(True)
            main_window.chk_use_single_label.setChecked(has_custom_label)
            main_window.chk_use_single_label.blockSignals(False)
            
            # 更新位置和颜色控件的值
            if has_custom_label:
                main_window.sp_single_label_x.blockSignals(True)
                main_window.sp_single_label_x.setValue(panel.spec.label_x * 100)
                main_window.sp_single_label_x.blockSignals(False)
                
                main_window.sp_single_label_y.blockSignals(True)
                main_window.sp_single_label_y.setValue(panel.spec.label_y * 100)
                main_window.sp_single_label_y.blockSignals(False)
                
                if panel.spec.label_color:
                    main_window.cmb_single_label_color.blockSignals(True)
                    main_window.cmb_single_label_color.setCurrentText(panel.spec.label_color)
                    main_window.cmb_single_label_color.blockSignals(False)
            else:
                # 没有独立设置时，显示默认值
                main_window.sp_single_label_x.blockSignals(True)
                main_window.sp_single_label_x.setValue(5.0)
                main_window.sp_single_label_x.blockSignals(False)
                
                main_window.sp_single_label_y.blockSignals(True)
                main_window.sp_single_label_y.setValue(5.0)
                main_window.sp_single_label_y.blockSignals(False)
                
                main_window.cmb_single_label_color.blockSignals(True)
                main_window.cmb_single_label_color.setCurrentText("黑色")
                main_window.cmb_single_label_color.blockSignals(False)
            
            # 根据复选框状态启用/禁用控件
            main_window.lbl_single_label_x.setEnabled(has_custom_label)
            main_window.sp_single_label_x.setEnabled(has_custom_label)
            main_window.lbl_single_label_y.setEnabled(has_custom_label)
            main_window.sp_single_label_y.setEnabled(has_custom_label)
            main_window.lbl_single_label_color.setEnabled(has_custom_label)
            main_window.cmb_single_label_color.setEnabled(has_custom_label)
        else:
            # 多个或没有面板被选中，禁用复选框和所有控件
            main_window.chk_use_single_label.setEnabled(False)
            main_window.chk_use_single_label.setToolTip("请先在画布中选中一个子图")
            main_window.chk_use_single_label.blockSignals(True)
            main_window.chk_use_single_label.setChecked(False)
            main_window.chk_use_single_label.blockSignals(False)
            
            main_window.lbl_single_label_x.setEnabled(False)
            main_window.sp_single_label_x.setEnabled(False)
            main_window.lbl_single_label_y.setEnabled(False)
            main_window.sp_single_label_y.setEnabled(False)
            main_window.lbl_single_label_color.setEnabled(False)
            main_window.cmb_single_label_color.setEnabled(False)

    def _fit_view(self):
        """让画布初始填充视口，避免背景看起来无法缩放"""
        if self._paper_rect is None:
            return
        rect = self._paper_rect.rect()
        # 加一点留白，方便看到边界
        padded = rect.adjusted(-5, -5, 5, 5)
        # 先重置缩放，再按比例适配
        self.view.resetTransform()
        self.view.fitInView(padded, Qt.AspectRatioMode.KeepAspectRatio)

    def set_canvas_size(self, w_mm, h_mm, margin_mm):
        # 1. 保存现有面板数据
        saved_specs = []
        for item in self.scene.items():
            if isinstance(item, PanelItem):
                 saved_specs.append(item.spec)
        
        # 2. 更新属性
        self.canvas_w = w_mm
        self.canvas_h = h_mm
        self.margin = margin_mm
        
        # 3. 重置场景背景
        self._init_scene()
        self._fit_view()
        
        # 4. 恢复面板
        for spec in saved_specs:
            self._create_panel_from_spec(spec)
            
        if saved_specs:
            self.reorder_labels()

    def adjust_image_offset(self, dx: float, dy: float):
        """按钮控制图片平移，dx/dy 为相对框宽高的比例增量"""
        changed = False
        for item in self.scene.selectedItems():
            if isinstance(item, PanelItem):
                item.spec.image_offset_x = max(-2.0, min(2.0, item.spec.image_offset_x + dx))
                item.spec.image_offset_y = max(-2.0, min(2.0, item.spec.image_offset_y + dy))
                item.update()
                changed = True
        if changed and hasattr(self.window(), 'schedule_preview'):
            self.window().schedule_preview()

    def adjust_image_scale(self, factor: float):
        """按钮控制图片缩放"""
        changed = False
        for item in self.scene.selectedItems():
            if isinstance(item, PanelItem):
                new_scale = item.spec.image_scale * factor
                new_scale = max(0.1, min(10.0, new_scale))
                if abs(new_scale - item.spec.image_scale) > 1e-6:
                    item.spec.image_scale = new_scale
                    item.update()
                    changed = True
        if changed and hasattr(self.window(), 'schedule_preview'):
            self.window().schedule_preview()

    def reset_image_adjustment(self):
        """重置选中面板的图片位置和缩放"""
        changed = False
        for item in self.scene.selectedItems():
            if isinstance(item, PanelItem):
                item.spec.image_offset_x = 0.0
                item.spec.image_offset_y = 0.0
                item.spec.image_scale = 1.0
                item.update()
                changed = True
        if changed and hasattr(self.window(), 'schedule_preview'):
            self.window().schedule_preview()

    def add_new_panel(self, path=None):
        import uuid
        spec = PanelSpec(
            id=str(uuid.uuid4()),
            x_mm=self.margin + 5,
            y_mm=self.margin + 5,
            w_mm=60,
            h_mm=45,
            image_path=path,
            label_index=self.get_next_label_idx()
        )
        return self._create_panel_from_spec(spec)
        
    def _create_panel_from_spec(self, spec: PanelSpec) -> PanelItem:
        item = PanelItem(spec)
        item.setPos(spec.x_mm, spec.y_mm)
        # 连接信号自动刷新预览
        if hasattr(self.window(), "schedule_preview"):
            item.layoutChanged.connect(self.window().schedule_preview)
        self.scene.addItem(item)
        return item

    def get_next_label_idx(self):
        # 简单返回当前数量即可，因为后续会重排
        count = 0
        for item in self.scene.items():
            if isinstance(item, PanelItem):
                count += 1
        return count

    def reorder_labels(self, anchor: Optional['PanelItem'] = None):
        """重新分配标签：自动行聚类 + 可指定起始面板为 (a)。"""
        items = [i for i in self.scene.items() if isinstance(i, PanelItem)]
        if not items:
            return
        
        # 计算平均高度作为聚类阈值
        mean_h = sum(i.spec.h_mm for i in items) / len(items) if items else 50
        threshold = mean_h * 0.5
        
        # 按 Y 坐标初步排序
        items.sort(key=lambda i: i.pos().y())
        
        rows = []
        if items:
            current_row = [items[0]]
            # 使用当前行的平均中心 Y 作为参考
            current_row_y = items[0].pos().y() + items[0].spec.h_mm / 2
            
            for item in items[1:]:
                cy = item.pos().y() + item.spec.h_mm / 2
                if abs(cy - current_row_y) < threshold:
                    # 同一行
                    current_row.append(item)
                    # 更新行参考 Y (移动平均)
                    avg_y = sum(i.pos().y() + i.spec.h_mm/2 for i in current_row) / len(current_row)
                    current_row_y = avg_y
                else:
                    # 新行
                    rows.append(current_row)
                    current_row = [item]
                    current_row_y = cy
            rows.append(current_row)
            
        # 行内按 X 排序，并展平
        final_list = []
        for r in rows:
            r.sort(key=lambda i: i.pos().x())
            final_list.extend(r)

        # 如果指定了锚点，将其旋转到序列首位，后续顺次编号
        if anchor and anchor in final_list:
            idx = final_list.index(anchor)
            final_list = final_list[idx:] + final_list[:idx]
        
        # 应用序号
        changed = False
        for idx, item in enumerate(final_list):
            if item.spec.label_index != idx:
                item.spec.label_index = idx
                item.update()
                changed = True
        
        if changed and hasattr(self.window(), "schedule_preview"):
            # 避免死循环，这里只在必要时触发，且已经在 mouseRelease 触发过了，
            # 这里主要是配合 add/delete/resize 后的重排
            pass

    def set_selected_as_first(self):
        """将当前选中的面板设为 (a)，其余按空间顺序自动递增。"""
        selected = [i for i in self.scene.selectedItems() if isinstance(i, PanelItem)]
        if len(selected) != 1:
            return
        self.reorder_labels(anchor=selected[0])
        if hasattr(self.window(), "schedule_preview"):
            self.window().schedule_preview()

    def delete_selected(self):
        deleted = False
        for item in self.scene.selectedItems():
            if isinstance(item, PanelItem):
                self.scene.removeItem(item)
                deleted = True
        
        if deleted:
            self.reorder_labels()
            self.window().schedule_preview()

    def align_selected(self, mode):
        items = [i for i in self.scene.selectedItems() if isinstance(i, PanelItem)]
        if len(items) < 2: return
        
        if mode == "left":
            min_x = min(i.pos().x() for i in items)
            for i in items: i.setPos(min_x, i.pos().y())
        elif mode == "right":
            max_r = max(i.pos().x() + i.spec.w_mm for i in items)
            for i in items: i.setPos(max_r - i.spec.w_mm, i.pos().y())
        elif mode == "top":
            min_y = min(i.pos().y() for i in items)
            for i in items: i.setPos(i.pos().x(), min_y)
        elif mode == "bottom":
            max_b = max(i.pos().y() + i.spec.h_mm for i in items)
            for i in items: i.setPos(i.pos().x(), max_b - i.spec.h_mm)
        elif mode == "hcenter":
            center_x = sum(i.pos().x() + i.spec.w_mm/2 for i in items) / len(items)
            for i in items: i.setPos(center_x - i.spec.w_mm/2, i.pos().y())
        elif mode == "vcenter":
            center_y = sum(i.pos().y() + i.spec.h_mm/2 for i in items) / len(items)
            for i in items: i.setPos(i.pos().x(), center_y - i.spec.h_mm/2)

        if hasattr(self.window(), "schedule_preview"):
            self.window().schedule_preview()

    def distribute_selected(self, mode):
        items = [i for i in self.scene.selectedItems() if isinstance(i, PanelItem)]
        if len(items) < 3: return
        
        if mode == "h":
            items.sort(key=lambda i: i.pos().x())
            min_x = items[0].pos().x()
            max_x = items[-1].pos().x()
            # 这里的分布是指中心点等距，或者边缘等距？通常是等间距分布
            # 方案：总宽度分布
            total_w = sum(i.spec.w_mm for i in items)
            span = items[-1].pos().x() + items[-1].spec.w_mm - items[0].pos().x()
            gap = (span - total_w) / (len(items) - 1)
            
            curr_x = items[0].pos().x()
            for i in items:
                i.setPos(curr_x, i.pos().y())
                curr_x += i.spec.w_mm + gap
        elif mode == "v":
            items.sort(key=lambda i: i.pos().y())
            total_h = sum(i.spec.h_mm for i in items)
            span = items[-1].pos().y() + items[-1].spec.h_mm - items[0].pos().y()
            gap = (span - total_h) / (len(items) - 1)
            
            curr_y = items[0].pos().y()
            for i in items:
                i.setPos(i.pos().x(), curr_y)
                curr_y += i.spec.h_mm + gap

        if hasattr(self.window(), "schedule_preview"):
            self.window().schedule_preview()

    def apply_gap_to_selected(self):
        """应用右侧面板设置的间距"""
        items = [i for i in self.scene.selectedItems() if isinstance(i, PanelItem)]
        if len(items) < 2: return
        
        # 获取间距值 (从 MainWindow 获取)
        gap = 5.0 # default
        if hasattr(self.window(), "sp_gap"):
            gap = float(self.window().sp_gap.value())
            
        # 自动判断是水平排列还是垂直排列
        # 计算 x跨度和 y跨度
        min_x = min(i.pos().x() for i in items)
        max_x = max(i.pos().x() + i.spec.w_mm for i in items)
        span_x = max_x - min_x
        
        min_y = min(i.pos().y() for i in items)
        max_y = max(i.pos().y() + i.spec.h_mm for i in items)
        span_y = max_y - min_y
        
        # 简单的启发式：判定主轴
        if span_x > span_y:
            # 水平排列
            # 按X排序
            items.sort(key=lambda i: i.pos().x())
            curr_x = items[0].pos().x()
            # 保持Y不变，只调X
            for i in items:
                i.setPos(curr_x, i.pos().y())
                curr_x += i.spec.w_mm + gap
        else:
            # 垂直排列
            items.sort(key=lambda i: i.pos().y())
            curr_y = items[0].pos().y()
            for i in items:
                i.setPos(i.pos().x(), curr_y)
                curr_y += i.spec.h_mm + gap
        
        self.reorder_labels()
        # 触发刷新
        if hasattr(self.window(), "schedule_preview"):
            self.window().schedule_preview()

    def export_layout(self) -> List[CellSpec]:
        """将 mm 面板转换为归一化的 CellSpec 供渲染器使用"""
        specs = []
        # 内容区尺寸
        cw = self.canvas_w - 2*self.margin
        ch = self.canvas_h - 2*self.margin
        
        # 按 label_index 排序面板，确保 (a)(b)(c) 顺序一致
        panel_items = [i for i in self.scene.items() if isinstance(i, PanelItem)]
        panel_items.sort(key=lambda x: x.spec.label_index)
        
        for item in panel_items:
            # 这里的坐标相对于 content 区域
            x_rel = (item.spec.x_mm - self.margin) / cw
            y_rel = (item.spec.y_mm - self.margin) / ch
            w_rel = item.spec.w_mm / cw
            h_rel = item.spec.h_mm / ch
            specs.append(CellSpec(
                x=x_rel,
                y=y_rel,
                w=w_rel,
                h=h_rel,
                image_offset_x=item.spec.image_offset_x,
                image_offset_y=item.spec.image_offset_y,
                image_scale=item.spec.image_scale,
                image_path=item.spec.image_path,
                label_x=item.spec.label_x,
                label_y=item.spec.label_y,
                label_color=item.spec.label_color
            ))
        return specs

    def get_image_paths(self) -> List[str]:
        panel_items = [i for i in self.scene.items() if isinstance(i, PanelItem)]
        panel_items.sort(key=lambda x: x.spec.label_index)
        return [i.spec.image_path for i in panel_items if i.spec.image_path]



# ----------------------------
# 渲染器：按论文尺寸导出
# ----------------------------
@dataclass
class RenderConfig:
    layout_name: str
    canvas_width_mm: float
    dpi: int
    margin_mm: float
    gap_mm: float
    fit_mode: str  # "智能填充（推荐）" / "裁剪填满" / "等比缩放+留白" / "拉伸填满"
    auto_trim: bool
    label_enabled: bool
    label_pos: str  # "左上" / "左下" / "自定义"
    label_size_pt: int
    label_padding_mm: float
    title_text: str
    title_enabled: bool
    title_size_pt: int
    background_white: bool
    canvas_height_mm: float = 120.0  # 移动到带默认值的区域
    custom_specs: Optional[List[CellSpec]] = None  # 自定义模式的布局
    label_custom_x: float = 0.05  # 自定义标注X位置（0-1相对位置）
    label_custom_y: float = 0.05  # 自定义标注Y位置（0-1相对位置）
    label_color: str = "黑色"  # 标注颜色："黑色" / "白色"
    label_style: str = "无框"  # 标注样式："有框" / "无框" / "半透明背景"
    # 子图边框选项
    border_enabled: bool = True  # 是否显示子图边框
    border_width: float = 1.0  # 边框宽度（px）
    border_color: str = "黑色"  # 边框颜色："黑色" / "白色" / "灰色"
    border_style: str = "实线"  # 边框样式："实线" / "虚线"
    # 标注字体样式
    label_font_weight: str = "Regular"  # 字体粗细："Regular" / "Bold"
    label_font_italic: bool = False  # 是否斜体


def trim_whitespace(img: Image.Image, threshold: int = 245) -> Image.Image:
    """
    简单“去白边”：假设背景接近白色。
    对于科研图（白底为主）通常可用，但不是万能。
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    # 快速采样：转灰度，找非白区域
    gray = img.convert("L")
    w, h = gray.size
    pix = gray.load()

    left, right, top, bottom = w, 0, h, 0
    found = False
    step = 1  # 可加大加速，但会降低精度
    for y in range(0, h, step):
        for x in range(0, w, step):
            if pix[x, y] < threshold:
                found = True
                left = min(left, x)
                right = max(right, x)
                top = min(top, y)
                bottom = max(bottom, y)

    if not found:
        return img

    # 适当留一点边界
    pad = 2
    left = clamp_int(left - pad, 0, w - 1)
    right = clamp_int(right + pad, 0, w - 1)
    top = clamp_int(top - pad, 0, h - 1)
    bottom = clamp_int(bottom + pad, 0, h - 1)

    if right <= left or bottom <= top:
        return img

    return img.crop((left, top, right + 1, bottom + 1))



def fit_image_to_box(
    img: Image.Image,
    box_w: int,
    box_h: int,
    fit_mode: str,
) -> Image.Image:
    """
    将图片适配到目标 box，支持多种模式：
    - "裁剪填满"：先放大后居中裁剪填满（推荐，适合手动调整后的图片）
    - "等比缩放+留白"：先缩放后留白（不裁剪，可能产生对齐问题）
    - "拉伸填满"：直接拉伸（不保持比例，会变形）
    """
    try:
        if box_w <= 0 or box_h <= 0:
            return img

        iw, ih = img.size
        if iw == 0 or ih == 0:
            return img.resize((box_w, box_h), Image.LANCZOS)

        # 标准化模式名称（移除UI装饰文本）
        fit_mode = fit_mode.replace("（推荐）", "").replace("(推荐)", "")

        if fit_mode == "拉伸填满":
            # 直接拉伸
            return img.resize((box_w, box_h), Image.LANCZOS)
        
        elif fit_mode == "裁剪填满":
            # 取较大比例，保证填满，然后裁剪
            sx = box_w / iw
            sy = box_h / ih
            s = max(sx, sy)
            nw = max(1, int(round(iw * s)))
            nh = max(1, int(round(ih * s)))
            resized = img.resize((nw, nh), Image.LANCZOS)
            # 居中裁剪
            left = (nw - box_w) // 2
            top = (nh - box_h) // 2
            return resized.crop((left, top, left + box_w, top + box_h))
        
        else:  # "等比缩放+留白" 或其他
            # 取较小比例，保证放进，留白
            sx = box_w / iw
            sy = box_h / ih
            s = min(sx, sy)
            nw = max(1, int(round(iw * s)))
            nh = max(1, int(round(ih * s)))
            resized = img.resize((nw, nh), Image.LANCZOS)
            # 留白居中
            canvas = Image.new("RGB", (box_w, box_h), (255, 255, 255))
            left = (box_w - nw) // 2
            top = (box_h - nh) // 2
            canvas.paste(resized, (left, top))
            return canvas
    except Exception as e:
        print(f"[ERROR] fit_image_to_box failed: {e}")
        # 返回白色占位框
        return Image.new("RGB", (max(1, box_w), max(1, box_h)), (255, 255, 255))


def render_panel_with_adjustment(
    img: Image.Image,
    box_w: int,
    box_h: int,
    cell: CellSpec,
) -> Image.Image:
    """应用自定义面板的缩放/取景参数，输出与框同尺寸的图像。"""
    try:
        if box_w <= 0 or box_h <= 0:
            return Image.new("RGB", (1, 1), (255, 255, 255))

        iw, ih = img.size
        if iw == 0 or ih == 0:
            return Image.new("RGB", (box_w, box_h), (255, 255, 255))

        base_scale = max(box_w / iw, box_h / ih)
        extra_scale = cell.image_scale if cell.image_scale > 0 else 1.0
        total_scale = base_scale * extra_scale

        nw = max(1, int(round(iw * total_scale)))
        nh = max(1, int(round(ih * total_scale)))
        resized = img.resize((nw, nh), Image.LANCZOS)

        offset_x = cell.image_offset_x * box_w
        offset_y = cell.image_offset_y * box_h
        x = (box_w - nw) / 2 + offset_x
        y = (box_h - nh) / 2 + offset_y

        panel = Image.new("RGB", (box_w, box_h), (255, 255, 255))
        panel.paste(resized, (int(round(x)), int(round(y))))
        return panel
    except Exception as e:
        print(f"[ERROR] render_panel_with_adjustment failed: {e}")
        return Image.new("RGB", (max(1, box_w), max(1, box_h)), (255, 255, 255))


def render_montage(paths: List[str], cfg: RenderConfig) -> Tuple[Image.Image, List[str]]:
    """
    输出拼图 PIL.Image + 警告信息
    """
    warnings = []
    use_custom_layout = bool(cfg.custom_specs)
    # 优先使用配置中的自定义布局
    if use_custom_layout:
        layout = cfg.custom_specs
    else:
        layout = build_layout(cfg.layout_name)

    # 计算画布像素尺寸：宽固定，高根据布局相对比例计算（归一化布局高度=1）
    canvas_w = mm_to_px(cfg.canvas_width_mm, cfg.dpi)

    margin = mm_to_px(cfg.margin_mm, cfg.dpi)
    gap = mm_to_px(cfg.gap_mm, cfg.dpi)

    # 标题区高度
    title_h = 0
    if cfg.title_enabled and cfg.title_text.strip():
        # 经验：标题区高度 ~ 1.4 * 字号（px），字号 pt->px：pt * dpi / 72
        title_px = int(round(cfg.title_size_pt * cfg.dpi / 72))
        title_h = int(round(title_px * 1.6))

    # 内容区宽高
    content_w = max(1, canvas_w - 2 * margin)
    
    # 画布总像素高度
    canvas_h_full = mm_to_px(cfg.canvas_height_mm, cfg.dpi)
    # 内容区像素高度 = 总高 - 边距 - 标题
    content_h = max(1, canvas_h_full - 2 * margin - title_h)

    # 只有在非自定义模式下，如果高度没有被显式调整，才执行旧的比例估算（可选）
    # 但为了统一，我们现在让 canvas_height_mm 成为权威。
    canvas_h = canvas_h_full

    bg = (255, 255, 255) if cfg.background_white else (240, 240, 240)
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(canvas)

    # 字体
    font_en, font_zh, font_warnings = load_fonts(size_en=max(8, int(round(cfg.label_size_pt * cfg.dpi / 72))),
                                                 size_zh=max(8, int(round(cfg.title_size_pt * cfg.dpi / 72))))
    warnings.extend(font_warnings)

    # 标题
    if cfg.title_enabled and cfg.title_text.strip():
        title = cfg.title_text.strip()
        # 简单规则：包含非 ASCII 则用中文字体，否则用英文
        use_font = font_zh if any(ord(ch) > 127 for ch in title) else font_en
        # 居中
        bbox = draw.textbbox((0, 0), title, font=use_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = (canvas_w - tw) // 2
        ty = margin + (title_h - th) // 2 - 2
        draw.text((tx, ty), title, fill=(0, 0, 0), font=use_font)

    # 内容区左上角
    ox = margin
    oy = margin + title_h

    # 每个 cell 的像素 box
    def cell_box(cell: CellSpec) -> Tuple[int, int, int, int]:
        # 将归一化坐标映射到 content 区域
        x = ox + int(round(cell.x * content_w))
        y = oy + int(round(cell.y * content_h))
        w = int(round(cell.w * content_w))
        h = int(round(cell.h * content_h))

        # 考虑 gap：每个 cell 内收缩一半间距（避免相邻重叠）
        shrink = gap // 2
        x += shrink
        y += shrink
        w -= gap
        h -= gap

        w = max(1, w)
        h = max(1, h)
        return x, y, x + w, y + h

    # 渲染子图
    n_cells = len(layout)
    if not use_custom_layout:
        n_imgs = len(paths)
        if n_imgs < n_cells:
            warnings.append(f"当前布局需要 {n_cells} 张图，但只导入了 {n_imgs} 张：剩余区域将显示占位框。")
    else:
        n_imgs = len(paths)  # 仅为兼容旧逻辑，不再用于索引

    label_pad = mm_to_px(cfg.label_padding_mm, cfg.dpi)
    for i, cell in enumerate(layout):
        x1, y1, x2, y2 = cell_box(cell)
        box_w = x2 - x1
        box_h = y2 - y1

        path = None
        if use_custom_layout:
            path = cell.image_path
        elif i < len(paths):
            path = paths[i]

        placeholder_needed = False

        if path:
            img = safe_open_image(path)
            if img is None:
                warnings.append(f"无法读取图片：{path}")
                placeholder_needed = True
            else:
                try:
                    if cfg.auto_trim:
                        try:
                            img = trim_whitespace(img)
                        except Exception:
                            pass

                    if use_custom_layout:
                        fitted = render_panel_with_adjustment(img, box_w, box_h, cell)
                    else:
                        fitted = fit_image_to_box(img, box_w, box_h, cfg.fit_mode)
                    
                    if fitted is not None:
                        canvas.paste(fitted, (x1, y1))
                    else:
                        warnings.append(f"无法处理图片：{path}")
                        placeholder_needed = True
                except Exception as e:
                    warnings.append(f"渲染图片时出错 {path}: {e}")
                    placeholder_needed = True
        else:
            placeholder_needed = True
            if use_custom_layout:
                letter = chr(ord('a') + i)
                warnings.append(f"自定义模式下的面板 ({letter}) 未分配图片，将显示占位框。")

        if placeholder_needed:
            draw.rectangle([x1, y1, x2, y2], outline=(160, 160, 160), width=2)
            draw.line([x1, y1, x2, y2], fill=(200, 200, 200), width=2)
            draw.line([x1, y2, x2, y1], fill=(200, 200, 200), width=2)

        # 子图标注 (a)(b)...
        if cfg.label_enabled:
            letter = chr(ord('a') + i)
            label = f"({letter})"
            # 标注字体用英文（Times）
            bbox = draw.textbbox((0, 0), label, font=font_en)
            lw = bbox[2] - bbox[0]
            lh = bbox[3] - bbox[1]
            
            # 获取标注位置（优先使用cell独立设置，否则使用全局设置）
            if cell.label_x is not None and cell.label_y is not None:
                # 使用子图独立的标注位置
                lx = x1 + int(box_w * cell.label_x)
                ly = y1 + int(box_h * cell.label_y)
            elif cfg.label_pos == "左上":
                lx = x1 + label_pad
                ly = y1 + label_pad
            elif cfg.label_pos == "左下":
                lx = x1 + label_pad
                ly = y2 - lh - label_pad
            else:  # 自定义位置
                lx = x1 + int(box_w * cfg.label_custom_x)
                ly = y1 + int(box_h * cfg.label_custom_y)
            
            # 获取标注颜色（优先使用cell独立设置，否则使用全局设置）
            label_color = cell.label_color if cell.label_color else cfg.label_color
            if label_color == "白色":
                text_color = (255, 255, 255)
                bg_color = (0, 0, 0)  # 白色文字用黑色背景
            else:
                text_color = (0, 0, 0)
                bg_color = (255, 255, 255)  # 黑色文字用白色背景

            # 根据标注样式绘制
            label_style = cfg.label_style
            
            if label_style == "无框":
                # 纯文字，不带背景
                draw.text((lx, ly), label, fill=text_color, font=font_en)
            
            elif label_style == "有框":
                # 带背景框和边框
                pad_box = max(2, int(round(lh * 0.20)))
                rx1 = lx - pad_box
                ry1 = ly - pad_box
                rx2 = lx + lw + pad_box
                ry2 = ly + lh + pad_box
                draw.rectangle([rx1, ry1, rx2, ry2], fill=bg_color, outline=text_color, width=1)
                draw.text((lx, ly), label, fill=text_color, font=font_en)
            
            elif label_style == "半透明背景":
                # 创建半透明背景
                pad_box = max(2, int(round(lh * 0.20)))
                rx1 = lx - pad_box
                ry1 = ly - pad_box
                rx2 = lx + lw + pad_box
                ry2 = ly + lh + pad_box
                
                # 创建一个临时的RGBA图层用于半透明效果
                overlay = Image.new('RGBA', canvas.size, (255, 255, 255, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                
                # 绘制半透明背景（使用与文字相反的颜色，透明度70%）
                alpha = 178  # 70% 不透明度
                if label_color == "白色":
                    bg_rgba = (0, 0, 0, alpha)
                else:
                    bg_rgba = (255, 255, 255, alpha)
                
                overlay_draw.rectangle([rx1, ry1, rx2, ry2], fill=bg_rgba)
                
                # 合并半透明层
                canvas_rgba = canvas.convert('RGBA')
                canvas_rgba = Image.alpha_composite(canvas_rgba, overlay)
                canvas.paste(canvas_rgba.convert('RGB'))
                
                # 绘制文字
                draw.text((lx, ly), label, fill=text_color, font=font_en)
        
        # 子图边框
        if cfg.border_enabled:
            border_color_map = {
                "黑色": (0, 0, 0),
                "白色": (255, 255, 255),
                "灰色": (128, 128, 128)
            }
            border_color = border_color_map.get(cfg.border_color, (0, 0, 0))
            border_width = int(max(1, cfg.border_width))
            
            if cfg.border_style == "虚线":
                # 手动绘制虚线边框（PIL 不直接支持虚线矩形）
                dash_length = 10
                gap_length = 5
                # 顶边
                x = x1
                while x < x2:
                    end_x = min(x + dash_length, x2)
                    draw.line([(x, y1), (end_x, y1)], fill=border_color, width=border_width)
                    x = end_x + gap_length
                # 底边
                x = x1
                while x < x2:
                    end_x = min(x + dash_length, x2)
                    draw.line([(x, y2), (end_x, y2)], fill=border_color, width=border_width)
                    x = end_x + gap_length
                # 左边
                y = y1
                while y < y2:
                    end_y = min(y + dash_length, y2)
                    draw.line([(x1, y), (x1, end_y)], fill=border_color, width=border_width)
                    y = end_y + gap_length
                # 右边
                y = y1
                while y < y2:
                    end_y = min(y + dash_length, y2)
                    draw.line([(x2, y), (x2, end_y)], fill=border_color, width=border_width)
                    y = end_y + gap_length
            else:  # 实线
                draw.rectangle([x1, y1, x2, y2], outline=border_color, width=border_width)

    return canvas, warnings


# ----------------------------
# GUI
# ----------------------------
class ImageListWidget(QListWidget):
    """
    支持：
    - 外部文件/文件夹拖拽导入
    - 列表内拖拽排序（InternalMove）
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setIconSize(QSize(96, 72))
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            paths = [u.toLocalFile() for u in urls if u.isLocalFile()]
            self.parent().add_paths(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1200, 760)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self.update_preview)

        self.image_paths: List[str] = []
        self.last_render: Optional[Image.Image] = None
        
        # 撤销/重做系统
        self.undo_stack = []  # 历史状态栈
        self.redo_stack = []  # 重做栈
        self.max_undo_steps = 50
        
        # 布局管理器
        try:
            from layout_manager import LayoutManager
            self.layout_manager = LayoutManager()
        except ImportError:
            self.layout_manager = None
            print("[WARN] layout_manager.py not found, save/load disabled")

        self._build_ui()
        self._build_toolbar()
        
        # 保存初始状态
        self._save_state()

    # ---------- UI ----------
    def _build_toolbar(self):
        tb = QToolBar("工具")
        tb.setMovable(False)
        self.addToolBar(tb)

        # 文件操作
        act_add = QAction("导入图片", self)
        act_add.setShortcut("Ctrl+I")
        act_add.triggered.connect(self.on_add_files)
        tb.addAction(act_add)

        act_add_dir = QAction("导入文件夹", self)
        act_add_dir.triggered.connect(self.on_add_folder)
        tb.addAction(act_add_dir)

        act_remove = QAction("删除选中", self)
        act_remove.setShortcut("Delete")
        act_remove.triggered.connect(self.on_remove_selected)
        tb.addAction(act_remove)

        tb.addSeparator()
        
        # 布局操作
        if self.layout_manager:
            act_save_layout = QAction("保存布局", self)
            act_save_layout.setShortcut("Ctrl+S")
            act_save_layout.triggered.connect(self.on_save_layout)
            tb.addAction(act_save_layout)
            
            act_load_layout = QAction("加载布局", self)
            act_load_layout.setShortcut("Ctrl+O")
            act_load_layout.triggered.connect(self.on_load_layout)
            tb.addAction(act_load_layout)
            
            tb.addSeparator()
        
        # 撤销/重做
        self.act_undo = QAction("撤销", self)
        self.act_undo.setShortcut("Ctrl+Z")
        self.act_undo.setEnabled(False)
        self.act_undo.triggered.connect(self.on_undo)
        tb.addAction(self.act_undo)
        
        self.act_redo = QAction("重做", self)
        self.act_redo.setShortcut("Ctrl+Y")
        self.act_redo.setEnabled(False)
        self.act_redo.triggered.connect(self.on_redo)
        tb.addAction(self.act_redo)
        
        tb.addSeparator()

        # 导出
        act_export = QAction("导出", self)
        act_export.setShortcut("Ctrl+E")
        act_export.triggered.connect(self.on_export)
        tb.addAction(act_export)

    def _build_ui(self):
        # 左：图片列表
        self.list_widget = ImageListWidget(self)
        self.list_widget.model().rowsMoved.connect(self._on_list_reordered)
        self.list_widget.itemSelectionChanged.connect(lambda: None)
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self.on_list_menu)

        left_box = QGroupBox("图片列表（拖拽导入 / 拖拽排序）")
        left_layout = QVBoxLayout(left_box)
        left_layout.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("导入图片")
        self.btn_add.clicked.connect(self.on_add_files)
        self.btn_rm = QPushButton("删除选中")
        self.btn_rm.clicked.connect(self.on_remove_selected)
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_rm)
        left_layout.addLayout(btn_row)

        # 中：画布编辑器 + 渲染预览（所有模式统一使用）
        # 创建画布编辑器（支持图片编辑功能）
        self.canvas_editor = CanvasEditorWidget()
        
        # 渲染预览区域
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(0)
        
        # 预览标题
        preview_title = QLabel("预览区域")
        preview_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_title.setStyleSheet("QLabel{background:#e0e0e0; padding:5px; font-weight:bold; color:black;}")
        preview_layout.addWidget(preview_title)
        
        # 渲染预览标签
        self.preview_label = QLabel("渲染预览")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("QLabel{background:#f7f7f7; border:1px solid #cfcfcf;}")
        self.preview_label.setMinimumWidth(300)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        preview_layout.addWidget(self.preview_label)
        
        # 分割器（画布编辑器 + 渲染预览）
        self.canvas_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.canvas_splitter.addWidget(self.canvas_editor)
        self.canvas_splitter.addWidget(preview_container)
        self.canvas_splitter.setStretchFactor(0, 1)
        self.canvas_splitter.setStretchFactor(1, 1)

        mid_box = QGroupBox("画布编辑（双击面板进入图片编辑模式）")
        mid_layout = QVBoxLayout(mid_box)
        mid_layout.addWidget(self.canvas_splitter)

        # 右：参数（使用滚动区域）
        right_content = QWidget()
        form = QFormLayout(right_content)

        self.cmb_layout = QComboBox()
        self.cmb_layout.addItems([
            "1+2（上大下两小）", "2+1（上两小下大）", "左大右两小", "右大左两小", 
            "2×2 网格", "3×2 网格", "快捷自定义", "自定义模式"
        ])
        self.cmb_layout.currentTextChanged.connect(self.on_layout_changed)
        form.addRow("布局模板", self.cmb_layout)
        
        # ========== 快捷自定义布局区域（优化版） ==========
        # 标题行（带帮助按钮）
        quick_title_widget = QWidget()
        quick_title_layout = QHBoxLayout(quick_title_widget)
        quick_title_layout.setContentsMargins(0, 0, 0, 0)
        quick_title_layout.setSpacing(5)
        
        self.lbl_custom_layout = QLabel("布局类型")
        self.btn_layout_help = QPushButton("❓")
        self.btn_layout_help.setMaximumWidth(30)
        self.btn_layout_help.setToolTip("点击查看详细使用说明")
        self.btn_layout_help.setStyleSheet("QPushButton { font-size: 12pt; padding: 2px; }")
        self.btn_layout_help.clicked.connect(self.show_layout_help)
        
        quick_title_layout.addWidget(self.lbl_custom_layout)
        quick_title_layout.addWidget(self.btn_layout_help)
        quick_title_layout.addStretch()
        quick_title_widget.setVisible(False)
        self.quick_title_widget = quick_title_widget
        
        # 布局类型选择（规则网格 / 不规则布局）
        layout_type_widget = QWidget()
        layout_type_layout = QHBoxLayout(layout_type_widget)
        layout_type_layout.setContentsMargins(0, 0, 0, 0)
        layout_type_layout.setSpacing(10)
        
        self.rb_grid_layout = QRadioButton("规则网格")
        self.rb_irregular_layout = QRadioButton("不规则布局")
        self.rb_grid_layout.setChecked(True)
        self.rb_grid_layout.toggled.connect(self.on_layout_type_changed)
        
        layout_type_layout.addWidget(self.rb_grid_layout)
        layout_type_layout.addWidget(self.rb_irregular_layout)
        layout_type_layout.addStretch()
        
        layout_type_widget.setVisible(False)
        self.layout_type_widget = layout_type_widget
        
        # 规则网格参数
        grid_params_widget = QWidget()
        grid_params_layout = QHBoxLayout(grid_params_widget)
        grid_params_layout.setContentsMargins(0, 0, 0, 0)
        grid_params_layout.setSpacing(10)
        
        grid_params_layout.addWidget(QLabel("行数:"))
        self.sp_custom_rows = QSpinBox()
        self.sp_custom_rows.setRange(1, 10)
        self.sp_custom_rows.setValue(2)
        self.sp_custom_rows.setMaximumWidth(60)
        self.sp_custom_rows.valueChanged.connect(self.on_custom_grid_params_changed)
        grid_params_layout.addWidget(self.sp_custom_rows)
        
        grid_params_layout.addWidget(QLabel("列数:"))
        self.sp_custom_cols = QSpinBox()
        self.sp_custom_cols.setRange(1, 10)
        self.sp_custom_cols.setValue(2)
        self.sp_custom_cols.setMaximumWidth(60)
        self.sp_custom_cols.valueChanged.connect(self.on_custom_grid_params_changed)
        grid_params_layout.addWidget(self.sp_custom_cols)
        
        grid_params_layout.addStretch()
        grid_params_widget.setVisible(False)
        self.grid_params_widget = grid_params_widget
        
        # 不规则布局参数
        irregular_params_widget = QWidget()
        irregular_params_layout = QVBoxLayout(irregular_params_widget)
        irregular_params_layout.setContentsMargins(0, 0, 0, 0)
        irregular_params_layout.setSpacing(5)
        
        # 行数控制
        row_count_layout = QHBoxLayout()
        row_count_layout.addWidget(QLabel("行数:"))
        self.sp_irregular_rows = QSpinBox()
        self.sp_irregular_rows.setRange(1, 10)
        self.sp_irregular_rows.setValue(2)
        self.sp_irregular_rows.setMaximumWidth(60)
        self.sp_irregular_rows.valueChanged.connect(self.on_irregular_rows_changed)
        row_count_layout.addWidget(self.sp_irregular_rows)
        row_count_layout.addStretch()
        irregular_params_layout.addLayout(row_count_layout)
        
        # 每行列数控制（动态生成）
        self.irregular_cols_container = QWidget()
        self.irregular_cols_layout = QVBoxLayout(self.irregular_cols_container)
        self.irregular_cols_layout.setContentsMargins(0, 0, 0, 0)
        self.irregular_cols_layout.setSpacing(3)
        self.irregular_col_spinboxes = []
        
        irregular_params_layout.addWidget(self.irregular_cols_container)
        
        irregular_params_widget.setVisible(False)
        self.irregular_params_widget = irregular_params_widget
        
        # 保留原来的输入框（隐藏，用于内部存储布局字符串）
        # 必须先创建这个，因为后面的初始化会用到
        self.ed_custom_layout = QLineEdit()
        self.ed_custom_layout.setVisible(False)
        self.ed_custom_layout.textChanged.connect(self.on_custom_layout_input_changed)
        
        # 添加到表单
        form.addRow(quick_title_widget)
        form.addRow("", layout_type_widget)
        form.addRow("", grid_params_widget)
        form.addRow("", irregular_params_widget)
        
        # 初始化不规则布局的列数控制（必须在ed_custom_layout创建之后）
        self.on_irregular_rows_changed()
        
        # 自定义网格参数 (保留引用但隐藏，因为旧逻辑可能还在用)
        self.sp_grid_rows = QSpinBox()
        self.sp_grid_rows.setVisible(False)
        self.sp_grid_cols = QSpinBox()
        self.sp_grid_cols.setVisible(False)
        self.lbl_grid_rows = QLabel("网格行数")
        self.lbl_grid_rows.setVisible(False)
        self.lbl_grid_cols = QLabel("网格列数")
        self.lbl_grid_cols.setVisible(False)
        form.addRow(self.lbl_grid_cols, self.sp_grid_cols)
        
        # 默认隐藏网格参数
        self.lbl_grid_rows.setVisible(False)
        self.sp_grid_rows.setVisible(False)
        self.lbl_grid_cols.setVisible(False)
        self.sp_grid_cols.setVisible(False)

        self.sp_width = QDoubleSpinBox()
        self.sp_width.setRange(40.0, 250.0)
        self.sp_width.setDecimals(1)
        self.sp_width.setValue(180.0)  # 双栏常用宽
        self.sp_width.valueChanged.connect(self.on_canvas_size_changed)
        form.addRow("输出宽度（mm）", self.sp_width)

        self.sp_height = QDoubleSpinBox()
        self.sp_height.setRange(20.0, 500.0)
        self.sp_height.setDecimals(1)
        self.sp_height.setValue(120.0)
        self.sp_height.valueChanged.connect(self.on_canvas_size_changed)
        form.addRow("输出高度（mm）", self.sp_height)

        self.sp_dpi = QComboBox()
        self.sp_dpi.addItems(["300", "600", "900"])
        self.sp_dpi.setCurrentText("600")
        self.sp_dpi.currentTextChanged.connect(lambda: self.schedule_preview())
        form.addRow("DPI", self.sp_dpi)

        self.sp_margin = QDoubleSpinBox()
        self.sp_margin.setRange(0.0, 20.0)
        self.sp_margin.setDecimals(1)
        self.sp_margin.setValue(2.5)
        self.sp_margin.valueChanged.connect(lambda: self.schedule_preview())
        form.addRow("外边距（mm）", self.sp_margin)
        
        self.sp_gap = QDoubleSpinBox()
        self.sp_gap.setRange(0.0, 20.0)
        self.sp_gap.setDecimals(1)
        self.sp_gap.setValue(2.0)
        self.sp_gap.valueChanged.connect(lambda: self.schedule_preview())
        form.addRow("子图间距（mm）", self.sp_gap)
        
        # 子图宽高比设置
        self.cmb_aspect_ratio = QComboBox()
        self.cmb_aspect_ratio.addItems([
            "自动（默认）",
            "1:1（正方形）",
            "4:3（横向）",
            "16:9（横向）",
            "3:4（竖向）",
            "9:16（竖向）",
            "自定义比例"
        ])
        self.cmb_aspect_ratio.setCurrentText("自动（默认）")
        self.cmb_aspect_ratio.setToolTip("调整子图框的宽高比例\n自动：根据布局自动计算\n其他：所有子图框使用统一比例")
        self.cmb_aspect_ratio.currentTextChanged.connect(self.on_aspect_ratio_changed)
        form.addRow("子图宽高比", self.cmb_aspect_ratio)
        
        # 自定义宽高比控件
        custom_ratio_widget = QWidget()
        custom_ratio_layout = QHBoxLayout(custom_ratio_widget)
        custom_ratio_layout.setContentsMargins(0, 0, 0, 0)
        custom_ratio_layout.setSpacing(5)
        
        custom_ratio_layout.addWidget(QLabel("宽:"))
        self.sp_aspect_width = QDoubleSpinBox()
        self.sp_aspect_width.setRange(0.1, 10.0)
        self.sp_aspect_width.setDecimals(1)
        self.sp_aspect_width.setValue(1.0)
        self.sp_aspect_width.setMaximumWidth(60)
        self.sp_aspect_width.valueChanged.connect(self.on_custom_aspect_changed)
        custom_ratio_layout.addWidget(self.sp_aspect_width)
        
        custom_ratio_layout.addWidget(QLabel("高:"))
        self.sp_aspect_height = QDoubleSpinBox()
        self.sp_aspect_height.setRange(0.1, 10.0)
        self.sp_aspect_height.setDecimals(1)
        self.sp_aspect_height.setValue(1.0)
        self.sp_aspect_height.setMaximumWidth(60)
        self.sp_aspect_height.valueChanged.connect(self.on_custom_aspect_changed)
        custom_ratio_layout.addWidget(self.sp_aspect_height)
        
        custom_ratio_layout.addStretch()
        custom_ratio_widget.setVisible(False)
        self.custom_ratio_widget = custom_ratio_widget
        form.addRow("", custom_ratio_widget)
        
        # 图片适配模式
        self.cmb_fit_mode = QComboBox()
        self.cmb_fit_mode.addItems([
            "裁剪填满（推荐）",
            "等比缩放+留白",
            "拉伸填满"
        ])
        self.cmb_fit_mode.setCurrentIndex(0)  # 默认裁剪填满
        self.cmb_fit_mode.currentTextChanged.connect(lambda: self.schedule_preview())
        form.addRow("图片适配模式", self.cmb_fit_mode)
        
        self.chk_trim = QCheckBox("自动去白边（谨慎开启）")
        self.chk_trim.setChecked(False)
        self.chk_trim.stateChanged.connect(lambda: self.schedule_preview())
        form.addRow(self.chk_trim)

        self.chk_label = QCheckBox("显示子图标注 (a)(b)(c)…")
        self.chk_label.setChecked(True)
        self.chk_label.stateChanged.connect(lambda: self.schedule_preview())
        form.addRow(self.chk_label)

        self.cmb_label_pos = QComboBox()
        self.cmb_label_pos.addItems(["左上", "左下", "自定义"])
        self.cmb_label_pos.currentTextChanged.connect(self.on_label_pos_changed)
        form.addRow("标注位置", self.cmb_label_pos)
        
        # 自定义标注位置控件（初始隐藏）
        self.lbl_label_x = QLabel("X位置（%）")
        self.slider_label_x = QDoubleSpinBox()
        self.slider_label_x.setRange(0.0, 100.0)
        self.slider_label_x.setDecimals(1)
        self.slider_label_x.setValue(5.0)
        self.slider_label_x.setSuffix("%")
        self.slider_label_x.valueChanged.connect(self.on_custom_label_changed)
        self.slider_label_x.valueChanged.connect(lambda: self.schedule_preview())
        form.addRow(self.lbl_label_x, self.slider_label_x)
        
        self.lbl_label_y = QLabel("Y位置（%）")
        self.slider_label_y = QDoubleSpinBox()
        self.slider_label_y.setRange(0.0, 100.0)
        self.slider_label_y.setDecimals(1)
        self.slider_label_y.setValue(5.0)
        self.slider_label_y.setSuffix("%")
        self.slider_label_y.valueChanged.connect(self.on_custom_label_changed)
        self.slider_label_y.valueChanged.connect(lambda: self.schedule_preview())
        form.addRow(self.lbl_label_y, self.slider_label_y)
        
        # 默认隐藏自定义位置控件
        self.lbl_label_x.setVisible(False)
        self.slider_label_x.setVisible(False)
        self.lbl_label_y.setVisible(False)
        self.slider_label_y.setVisible(False)

        self.sp_label_pt = QSpinBox()
        self.sp_label_pt.setRange(6, 24)
        self.sp_label_pt.setValue(10)
        self.sp_label_pt.valueChanged.connect(lambda: self.schedule_preview())
        form.addRow("标注字号（pt）", self.sp_label_pt)

        self.sp_label_pad = QDoubleSpinBox()
        self.sp_label_pad.setRange(0.0, 10.0)
        self.sp_label_pad.setDecimals(1)
        self.sp_label_pad.setValue(1.5)
        self.sp_label_pad.valueChanged.connect(lambda: self.schedule_preview())
        form.addRow("标注内边距（mm）", self.sp_label_pad)
        
        # 标注颜色（全局）
        self.lbl_label_color = QLabel("标注颜色")
        self.cmb_label_color = QComboBox()
        self.cmb_label_color.addItems(["黑色", "白色"])
        self.cmb_label_color.setCurrentText("黑色")
        self.cmb_label_color.currentTextChanged.connect(lambda: self.schedule_preview())
        form.addRow(self.lbl_label_color, self.cmb_label_color)
        
        # 标注样式（全局）
        self.lbl_label_style = QLabel("标注样式")
        self.cmb_label_style = QComboBox()
        self.cmb_label_style.addItems(["无框", "有框", "半透明背景"])
        self.cmb_label_style.setCurrentText("无框")
        self.cmb_label_style.setToolTip("无框：纯文字，简洁\n有框：带背景框和边框，醒目\n半透明背景：半透明背景，美观且清晰")
        self.cmb_label_style.currentTextChanged.connect(lambda: self.schedule_preview())
        form.addRow(self.lbl_label_style, self.cmb_label_style)
        
        # 单个子图标注独立控制
        form.addRow(QLabel(""))  # 分隔
        self.lbl_single_label = QLabel("【选中子图的标注设置】")
        form.addRow(self.lbl_single_label)
        
        self.chk_use_single_label = QCheckBox("使用独立的标注位置")
        self.chk_use_single_label.setChecked(False)
        self.chk_use_single_label.stateChanged.connect(self.on_single_label_toggle)
        form.addRow(self.chk_use_single_label)
        
        self.lbl_single_label_x = QLabel("X位置（%）")
        self.sp_single_label_x = QDoubleSpinBox()
        self.sp_single_label_x.setRange(0.0, 100.0)
        self.sp_single_label_x.setDecimals(1)
        self.sp_single_label_x.setValue(5.0)
        self.sp_single_label_x.setSuffix("%")
        self.sp_single_label_x.valueChanged.connect(self.on_single_label_pos_changed)
        form.addRow(self.lbl_single_label_x, self.sp_single_label_x)
        
        self.lbl_single_label_y = QLabel("Y位置（%）")
        self.sp_single_label_y = QDoubleSpinBox()
        self.sp_single_label_y.setRange(0.0, 100.0)
        self.sp_single_label_y.setDecimals(1)
        self.sp_single_label_y.setValue(5.0)
        self.sp_single_label_y.setSuffix("%")
        self.sp_single_label_y.valueChanged.connect(self.on_single_label_pos_changed)
        form.addRow(self.lbl_single_label_y, self.sp_single_label_y)
        
        self.lbl_single_label_color = QLabel("标注颜色")
        self.cmb_single_label_color = QComboBox()
        self.cmb_single_label_color.addItems(["黑色", "白色"])
        self.cmb_single_label_color.setCurrentText("黑色")
        self.cmb_single_label_color.currentTextChanged.connect(self.on_single_label_color_changed)
        form.addRow(self.lbl_single_label_color, self.cmb_single_label_color)
        
        # 默认禁用单个子图标注控件
        self.lbl_single_label_x.setEnabled(False)
        self.sp_single_label_x.setEnabled(False)
        self.lbl_single_label_y.setEnabled(False)
        self.sp_single_label_y.setEnabled(False)
        self.lbl_single_label_color.setEnabled(False)
        self.cmb_single_label_color.setEnabled(False)

        # 子图边框选项
        self.chk_border = QCheckBox("显示子图边框")
        self.chk_border.setChecked(True)
        self.chk_border.stateChanged.connect(self.on_border_changed)
        form.addRow(self.chk_border)
        
        self.lbl_border_width = QLabel("边框宽度（px）")
        self.sp_border_width = QDoubleSpinBox()
        self.sp_border_width.setRange(0.1, 10.0)
        self.sp_border_width.setDecimals(1)
        self.sp_border_width.setValue(1.0)
        self.sp_border_width.valueChanged.connect(lambda: self.schedule_preview())
        form.addRow(self.lbl_border_width, self.sp_border_width)
        
        self.lbl_border_color = QLabel("边框颜色")
        self.cmb_border_color = QComboBox()
        self.cmb_border_color.addItems(["黑色", "白色", "灰色"])
        self.cmb_border_color.setCurrentText("黑色")
        self.cmb_border_color.currentTextChanged.connect(lambda: self.schedule_preview())
        form.addRow(self.lbl_border_color, self.cmb_border_color)
        
        self.lbl_border_style = QLabel("边框样式")
        self.cmb_border_style = QComboBox()
        self.cmb_border_style.addItems(["实线", "虚线"])
        self.cmb_border_style.setCurrentText("实线")
        self.cmb_border_style.currentTextChanged.connect(lambda: self.schedule_preview())
        form.addRow(self.lbl_border_style, self.cmb_border_style)
        
        # 初始同步边框控件可见性
        self.on_border_changed()

        self.chk_title = QCheckBox("添加标题（可中文/英文）")
        self.chk_title.setChecked(False)
        self.chk_title.stateChanged.connect(lambda: self.schedule_preview())
        form.addRow(self.chk_title)

        self.ed_title = QLineEdit()
        self.ed_title.setPlaceholderText("例如：图1-1 不同参数下的速度对比")
        self.ed_title.textChanged.connect(lambda: self.schedule_preview())
        form.addRow("标题文本", self.ed_title)

        self.sp_title_pt = QSpinBox()
        self.sp_title_pt.setRange(8, 30)
        self.sp_title_pt.setValue(12)
        self.sp_title_pt.valueChanged.connect(lambda: self.schedule_preview())
        form.addRow("标题字号（pt）", self.sp_title_pt)
        
        # 导出格式选择
        self.cmb_export_format = QComboBox()
        self.cmb_export_format.addItems(["PNG", "JPEG", "TIFF"])
        self.cmb_export_format.setCurrentText("PNG")
        form.addRow("导出格式", self.cmb_export_format)

        self.btn_export = QPushButton("导出图片")
        self.btn_export.clicked.connect(self.on_export)
        form.addRow(self.btn_export)

        # 创建滚动区域包装右侧参数面板
        scroll_area = QScrollArea()
        scroll_area.setWidget(right_content)
        scroll_area.setWidgetResizable(True)  # 允许内容自适应宽度
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # 禁用横向滚动
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # 需要时显示纵向滚动条
        
        # 包装在GroupBox中
        right_box = QGroupBox("论文参数 / 布局 / 标注")
        scroll_layout = QVBoxLayout(right_box)
        scroll_layout.setContentsMargins(5, 5, 5, 5)
        scroll_layout.addWidget(scroll_area)

        # 主布局：三栏 splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_box)
        splitter.addWidget(mid_box)
        splitter.addWidget(right_box)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 2)  # 右侧参数面板宽度与中间相同

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.addWidget(splitter)
        self.setCentralWidget(root)

        # 初始化时禁用单个子图标注控件（没有选中任何面板）
        self.chk_use_single_label.setEnabled(False)
        self.chk_use_single_label.setToolTip("请先在画布中选中一个子图")

        # 初始化画布尺寸并应用初始布局
        self.on_layout_changed(self.cmb_layout.currentText())

    # ---------- List context menu ----------
    def on_list_menu(self, pos):
        menu = QMenu(self)
        act_open = QAction("打开所在文件夹", self)
        act_open.triggered.connect(self.open_selected_folder)
        menu.addAction(act_open)

        act_remove = QAction("删除选中", self)
        act_remove.triggered.connect(self.on_remove_selected)
        menu.addAction(act_remove)

        menu.exec(self.list_widget.mapToGlobal(pos))

    def _on_list_reordered(self, *args):
        """列表排序变化时同步画布与预览"""
        self._refresh_canvas_layout()
        self.schedule_preview()

    def open_selected_folder(self):
        items = self.list_widget.selectedItems()
        if not items:
            return
        path = items[0].data(Qt.ItemDataRole.UserRole)
        if path and os.path.exists(path):
            folder = os.path.dirname(path)
            try:
                if sys.platform.startswith("win"):
                    os.startfile(folder)  # type: ignore
                elif sys.platform.startswith("darwin"):
                    os.system(f'open "{folder}"')
                else:
                    os.system(f'xdg-open "{folder}"')
            except Exception:
                pass

    # ---------- Import ----------
    def on_add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
        )
        if files:
            self.add_paths(files)

    def on_add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if not folder:
            return
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        files = []
        for fn in sorted(os.listdir(folder)):
            p = os.path.join(folder, fn)
            if os.path.isfile(p) and os.path.splitext(fn.lower())[1] in exts:
                files.append(p)
        if files:
            self.add_paths(files)

    def add_paths(self, paths: List[str]):
        # 递归处理文件夹
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        collected = []
        for p in paths:
            if not p:
                continue
            if os.path.isdir(p):
                for fn in sorted(os.listdir(p)):
                    fp = os.path.join(p, fn)
                    if os.path.isfile(fp) and os.path.splitext(fn.lower())[1] in exts:
                        collected.append(fp)
            elif os.path.isfile(p) and os.path.splitext(p.lower())[1] in exts:
                collected.append(p)

        if not collected:
            print(f"[INFO] No valid image files found in provided paths")
            return

        failed_files = []
        for p in collected:
            if p in self.image_paths:
                print(f"[INFO] Image already added: {p}")
                continue
            try:
                # 验证文件是否可以打开
                test_img = safe_open_image(p)
                if test_img is None:
                    failed_files.append(p)
                    print(f"[ERROR] Failed to open image (unsupported format?): {p}")
                    continue
                self.image_paths.append(p)
                self._add_item(p)
            except Exception as e:
                failed_files.append(p)
                print(f"[ERROR] Exception adding image {p}: {e}")

        if failed_files:
            print(f"[WARN] {len(failed_files)} image(s) failed to load:")
            for f in failed_files:
                print(f"  - {f}")

        # 刷新画布布局
        self._refresh_canvas_layout()
        self.schedule_preview()
        
        # 保存状态（用于撤销）
        self._save_state()

    def _add_item(self, path: str):
        item = QListWidgetItem()
        item.setText(os.path.basename(path))
        item.setData(Qt.ItemDataRole.UserRole, path)

        # 缩略图
        try:
            img = safe_open_image(path)
            if img is not None:
                thumb = img.copy()
                thumb.thumbnail((256, 192))
                pix = pil_to_qpixmap(thumb)
                if pix is not None:
                    item.setIcon(QIcon(pix))
                else:
                    print(f"[WARN] Failed to convert thumbnail for: {path}")
        except Exception as e:
            print(f"[WARN] Failed to generate thumbnail for {path}: {e}")
        
        self.list_widget.addItem(item)

    def on_remove_selected(self):
        items = self.list_widget.selectedItems()
        if not items:
            return
        for it in items:
            path = it.data(Qt.ItemDataRole.UserRole)
            row = self.list_widget.row(it)
            self.list_widget.takeItem(row)
            if path in self.image_paths:
                self.image_paths.remove(path)
        # 刷新画布布局
        self._refresh_canvas_layout()
        self.schedule_preview()
        
        # 保存状态（用于撤销）
        self._save_state()

    # ---------- 布局切换 ----------
    def on_layout_changed(self, layout_name: str):
        """布局模板切换"""
        is_custom_mode = (layout_name == "自定义模式")
        is_quick_custom = (layout_name == "快捷自定义")
        
        # 控制快捷自定义布局UI的可见性
        self.quick_title_widget.setVisible(is_quick_custom)
        self.layout_type_widget.setVisible(is_quick_custom)
        
        # 根据布局类型显示对应的参数控件
        if is_quick_custom:
            is_grid = self.rb_grid_layout.isChecked()
            self.grid_params_widget.setVisible(is_grid)
            self.irregular_params_widget.setVisible(not is_grid)
        
        # 同步画布尺寸
        self.canvas_editor.set_canvas_size(
            float(self.sp_width.value()),
            float(self.sp_height.value()),
            float(self.sp_margin.value())
        )
        
        # 控制工具栏各部分的可见性
        # 布局调整按钮（仅自定义模式可用）
        layout_buttons = [
            self.canvas_editor.btn_add,
            self.canvas_editor.btn_del,
            self.canvas_editor.btn_align_left,
            self.canvas_editor.btn_align_right,
            self.canvas_editor.btn_align_top,
            self.canvas_editor.btn_align_bottom,
            self.canvas_editor.btn_align_hcenter,
            self.canvas_editor.btn_align_vcenter,
            self.canvas_editor.btn_dist_h,
            self.canvas_editor.btn_dist_v,
            self.canvas_editor.btn_apply_gap,
            self.canvas_editor.btn_set_first,
            self.canvas_editor.btn_smart_h,
        ]
        for btn in layout_buttons:
            btn.setVisible(is_custom_mode)
        
        # 图片取景按钮始终可见
        image_buttons = [
            self.canvas_editor.btn_img_left,
            self.canvas_editor.btn_img_right,
            self.canvas_editor.btn_img_up,
            self.canvas_editor.btn_img_down,
            self.canvas_editor.btn_zoom_in,
            self.canvas_editor.btn_zoom_out,
        ]
        for btn in image_buttons:
            btn.setVisible(True)
        
        # 工具栏始终可见
        self.canvas_editor.toolbar.setVisible(True)
        
        # 控制宽高比控件的启用状态（自定义模式下禁用）
        aspect_enabled = not is_custom_mode
        self.cmb_aspect_ratio.setEnabled(aspect_enabled)
        if aspect_enabled:
            self.cmb_aspect_ratio.setToolTip("调整子图框的宽高比例\n自动：根据布局自动计算\n其他：所有子图框使用统一比例")
        else:
            self.cmb_aspect_ratio.setToolTip("自定义模式下不可用\n在自定义模式中，您可以手动调整每个框的尺寸")
        
        # 自定义比例控件也随之禁用
        if hasattr(self, 'custom_ratio_widget'):
            if not aspect_enabled and self.custom_ratio_widget.isVisible():
                self.custom_ratio_widget.setVisible(False)
        
        if is_custom_mode:
            # 自定义模式：可自由布局
            # 只有当画布为空时才自动导入列表图片
            if not any(isinstance(i, PanelItem) for i in self.canvas_editor.scene.items()):
                self._update_canvas_from_images()
        elif is_quick_custom:
            # 快捷自定义：使用输入框内容作为布局模板
            # 如果输入框为空，先设置默认值
            if not self.ed_custom_layout.text().strip():
                self.ed_custom_layout.setText("2x2")
            else:
                # 应用输入框中的布局
                self._apply_template_layout(self.ed_custom_layout.text().strip())
        else:
            # 预定义模板模式：自动按模板布局
            self._apply_template_layout(layout_name)
        
        self.schedule_preview()
    
    def on_custom_layout_input_changed(self):
        """用户修改自定义布局输入框时触发"""
        layout_str = self.ed_custom_layout.text().strip()
        if not layout_str:
            return
        
        # 尝试解析布局字符串
        parsed = parse_layout_string(layout_str)
        if not parsed:
            # 输入无效，不做任何操作
            return
        
        # 应用新的布局
        self._apply_template_layout(layout_str)
    
    
    # ========== 快捷自定义布局事件处理 ==========
    def show_layout_help(self):
        """显示布局帮助对话框"""
        from PySide6.QtWidgets import QMessageBox
        
        help_text = """
<h3>📐 快捷自定义布局使用说明</h3>

<h4>🔹 规则网格</h4>
<p>适用于所有子图大小相同的情况</p>
<ul>
<li><b>行数</b>：网格的行数（1-10）</li>
<li><b>列数</b>：网格的列数（1-10）</li>
<li><b>示例</b>：2行×2列 = 4个相同大小的子图</li>
</ul>

<h4>🔹 不规则布局</h4>
<p>适用于每行子图数量不同的情况，下方行自动居中对齐</p>
<ul>
<li><b>行数</b>：设定总行数</li>
<li><b>每行列数</b>：分别设定每一行有几个子图</li>
<li><b>自动居中</b>：较少子图的行会自动居中对齐</li>
</ul>

<h4>📝 使用示例</h4>
<table border="1" cellpadding="5" cellspacing="0">
<tr><th>需求</th><th>设置方法</th></tr>
<tr><td>4个图，2×2排列</td><td>规则网格：2行×2列</td></tr>
<tr><td>6个图，3×2排列</td><td>规则网格：2行×3列</td></tr>
<tr><td>5个图，上3下2</td><td>不规则：2行，第1行3个，第2行2个</td></tr>
<tr><td>7个图，上4下3</td><td>不规则：2行，第1行4个，第2行3个</td></tr>
<tr><td>9个图，上4中3下2</td><td>不规则：3行，各行分别4、3、2个</td></tr>
</table>

<h4>💡 提示</h4>
<p>• 布局设定后会自动应用<br>
• 可以随时切换布局类型<br>
• 不规则布局会自动计算居中对齐</p>
"""
        
        msg = QMessageBox(self)
        msg.setWindowTitle("快捷自定义布局帮助")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(help_text)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
    
    def on_layout_type_changed(self):
        """布局类型切换（规则网格 vs 不规则布局）"""
        is_grid = self.rb_grid_layout.isChecked()
        self.grid_params_widget.setVisible(is_grid)
        self.irregular_params_widget.setVisible(not is_grid)
        
        # 更新布局字符串
        if is_grid:
            self.on_custom_grid_params_changed()
        else:
            self.on_irregular_layout_changed()
    
    def on_custom_grid_params_changed(self):
        """规则网格参数变化"""
        rows = self.sp_custom_rows.value()
        cols = self.sp_custom_cols.value()
        layout_str = f"{rows}x{cols}"
        self.ed_custom_layout.setText(layout_str)
    
    def on_irregular_rows_changed(self):
        """不规则布局行数变化，动态生成每行的列数控制"""
        rows = self.sp_irregular_rows.value()
        
        # 清空现有的列数控件
        while self.irregular_cols_layout.count():
            item = self.irregular_cols_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.irregular_col_spinboxes = []
        
        # 为每一行创建列数控件
        for i in range(rows):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(10)
            
            label = QLabel(f"第{i+1}行列数:")
            row_layout.addWidget(label)
            
            spinbox = QSpinBox()
            spinbox.setRange(1, 10)
            spinbox.setValue(3 if i == 0 else 2)  # 第一行默认3个，其他行默认2个
            spinbox.setMaximumWidth(60)
            spinbox.valueChanged.connect(self.on_irregular_layout_changed)
            row_layout.addWidget(spinbox)
            
            row_layout.addStretch()
            
            self.irregular_cols_layout.addWidget(row_widget)
            self.irregular_col_spinboxes.append(spinbox)
        
        # 更新布局
        self.on_irregular_layout_changed()
    
    def on_irregular_layout_changed(self):
        """不规则布局列数变化"""
        if not self.irregular_col_spinboxes:
            return
        
        cols_list = [sp.value() for sp in self.irregular_col_spinboxes]
        layout_str = str(cols_list)  # 生成 [3,2] 格式
        self.ed_custom_layout.setText(layout_str)
    
    def on_aspect_ratio_changed(self):
        """子图宽高比选择改变"""
        ratio_text = self.cmb_aspect_ratio.currentText()
        
        # 显示/隐藏自定义比例控件
        is_custom = (ratio_text == "自定义比例")
        self.custom_ratio_widget.setVisible(is_custom)
        
        # 如果是自定义模式，不自动应用
        if not is_custom:
            self.on_custom_aspect_changed()
    
    def on_custom_aspect_changed(self):
        """自定义宽高比或预设比例改变时，重新应用布局"""
        # 只在模板模式下有效，自定义模式不受影响
        layout_name = self.cmb_layout.currentText()
        if layout_name == "自定义模式":
            return
        
        # 重新应用当前布局
        if layout_name == "快捷自定义":
            layout_str = self.ed_custom_layout.text().strip()
            if layout_str:
                self._apply_template_layout(layout_str)
        else:
            self._apply_template_layout(layout_name)
    
    def _get_aspect_ratio(self):
        """获取当前设置的宽高比"""
        ratio_text = self.cmb_aspect_ratio.currentText()
        
        if ratio_text == "自动（默认）":
            return None  # 使用默认布局
        elif ratio_text == "1:1（正方形）":
            return 1.0
        elif ratio_text == "4:3（横向）":
            return 4.0 / 3.0
        elif ratio_text == "16:9（横向）":
            return 16.0 / 9.0
        elif ratio_text == "3:4（竖向）":
            return 3.0 / 4.0
        elif ratio_text == "9:16（竖向）":
            return 9.0 / 16.0
        elif ratio_text == "自定义比例":
            width = self.sp_aspect_width.value()
            height = self.sp_aspect_height.value()
            return width / height if height > 0 else 1.0
        
        return None
    
    def _apply_template_layout(self, layout_name: str):
        """根据布局模板自动创建面板，保留已有的图片编辑参数"""
        # 检查UI是否已完全初始化（避免初始化阶段调用）
        if not hasattr(self, 'sp_gap') or not hasattr(self, 'canvas_editor'):
            return
        
        try:
            layout = build_layout(layout_name)
            paths = self._collect_paths_from_list()
            
            # 保存现有面板的编辑参数（按图片路径索引）
            existing_adjustments = {}
            for item in self.canvas_editor.scene.items():
                if isinstance(item, PanelItem) and item.spec.image_path:
                    existing_adjustments[item.spec.image_path] = {
                        'offset_x': item.spec.image_offset_x,
                        'offset_y': item.spec.image_offset_y,
                        'scale': item.spec.image_scale
                    }
            
            # 清空现有面板
            for item in list(self.canvas_editor.scene.items()):
                if isinstance(item, PanelItem):
                    self.canvas_editor.scene.removeItem(item)
            
            # 计算内容区尺寸
            margin = self.canvas_editor.margin
            content_w = self.canvas_editor.canvas_w - 2 * margin
            content_h = self.canvas_editor.canvas_h - 2 * margin
            gap = float(self.sp_gap.value())
            
            # 获取宽高比设置
            aspect_ratio = self._get_aspect_ratio()
            
            # 如果设置了宽高比，先计算理想的画布高度
            if aspect_ratio is not None:
                # 计算每个cell的实际高度需求
                # 找出最大列数（用于计算单个框的宽度）
                max_cols = max(len([c for c in layout if c.y == row_y]) 
                              for row_y in set(c.y for c in layout))
                cell_width = content_w / max_cols - gap
                
                # 根据宽高比计算理想的单个框高度
                ideal_cell_height = cell_width / aspect_ratio
                
                # 计算行数
                num_rows = len(set(c.y for c in layout))
                
                # 计算理想的内容高度（所有行的高度 + 行间间距）
                ideal_content_h = ideal_cell_height * num_rows + gap * (num_rows - 1)
                
                # 计算理想的画布总高度
                ideal_canvas_h = ideal_content_h + 2 * margin
                
                # 更新画布和UI显示
                self.canvas_editor.set_canvas_size(
                    self.canvas_editor.canvas_w,
                    ideal_canvas_h,
                    margin
                )
                self.sp_height.blockSignals(True)
                self.sp_height.setValue(ideal_canvas_h)
                self.sp_height.blockSignals(False)
                
                # 重新计算content_h
                content_h = ideal_canvas_h - 2 * margin
            
            # 根据布局模板创建面板
            import uuid
            for i, cell in enumerate(layout):
                # 计算面板位置和尺寸（考虑间距）
                x = margin + cell.x * content_w + gap / 2
                y = margin + cell.y * content_h + gap / 2
                w = cell.w * content_w - gap
                h = cell.h * content_h - gap
                
                # 应用宽高比调整
                if aspect_ratio is not None:
                    # 根据宽高比调整框的尺寸
                    # 保持宽度，调整高度
                    h = w / aspect_ratio
                    # 如果调整后的高度超出了cell的高度范围，则调整方案
                    cell_max_h = cell.h * content_h - gap
                    if h > cell_max_h:
                        # 保持高度，调整宽度
                        h = cell_max_h
                        w = h * aspect_ratio
                        # 重新居中x位置
                        cell_center_x = margin + (cell.x + cell.w / 2) * content_w
                        x = cell_center_x - w / 2
                
                path = paths[i] if i < len(paths) else None
                
                spec = PanelSpec(
                    id=str(uuid.uuid4()),
                    x_mm=x,
                    y_mm=y,
                    w_mm=max(10, w),
                    h_mm=max(10, h),
                    image_path=path,
                    label_index=i
                )
                
                # 恢复之前的编辑参数
                if path and path in existing_adjustments:
                    adj = existing_adjustments[path]
                    spec.image_offset_x = adj['offset_x']
                    spec.image_offset_y = adj['offset_y']
                    spec.image_scale = adj['scale']
                
                item = self.canvas_editor._create_panel_from_spec(spec)
                # 模板模式下锁定面板位置（但允许图片编辑）
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
            
        except Exception as e:
            print(f"Error applying template layout: {e}")
        
        # 触发预览更新
        self.schedule_preview()
    
    def _refresh_canvas_layout(self):
        """根据当前布局模式刷新画布"""
        layout_name = self.cmb_layout.currentText()
        if layout_name == "自定义模式":
            self._update_canvas_from_images()
        else:
            # 模板模式：只更新图片分配，不重建布局
            self._update_panel_images()
    
    def _update_panel_images(self):
        """更新面板中的图片，保持布局不变"""
        paths = self._collect_paths_from_list()
        
        # 获取现有面板（按label_index排序）
        panels = []
        for item in self.canvas_editor.scene.items():
            if isinstance(item, PanelItem):
                panels.append(item)
        
        # 如果没有面板，需要创建新布局
        if not panels:
            self._apply_template_layout(self.cmb_layout.currentText())
            return
        
        # 按label_index排序
        panels.sort(key=lambda p: p.spec.label_index)
        
        # 更新每个面板的图片路径
        for i, panel in enumerate(panels):
            if i < len(paths):
                panel.spec.image_path = paths[i]
                # 重新加载图片
                panel._load_image(paths[i])
            else:
                panel.spec.image_path = None
                panel.pixmap = None
            panel.update()
        
        # 如果图片数量多于面板数量，需要应用新布局
        if len(paths) > len(panels):
            self._apply_template_layout(self.cmb_layout.currentText())
        else:
            # 触发预览更新
            self.schedule_preview()
    
    
    def _update_canvas_from_images(self):
        """使用图片列表更新画布编辑器（用于自定义模式）"""
        try:
            paths = self._collect_paths_from_list()
            # 清空现有面板
            for item in list(self.canvas_editor.scene.items()):
                if isinstance(item, PanelItem):
                    self.canvas_editor.scene.removeItem(item)
            
            # 如果没有图片，创建默认的空面板供用户拖入图片
            if not paths:
                # 创建 2x2 默认面板
                margin = self.canvas_editor.margin
                content_w = self.canvas_editor.canvas_w - 2 * margin
                content_h = self.canvas_editor.canvas_h - 2 * margin
                gap = float(self.sp_gap.value())
                
                panel_w = (content_w - gap) / 2
                panel_h = (content_h - gap) / 2
                
                positions = [
                    (margin, margin),
                    (margin + panel_w + gap, margin),
                    (margin, margin + panel_h + gap),
                    (margin + panel_w + gap, margin + panel_h + gap),
                ]
                
                for idx, (x, y) in enumerate(positions):
                    item = self.canvas_editor.add_new_panel(None)
                    item.spec.w_mm = panel_w
                    item.spec.h_mm = panel_h
                    item.spec.x_mm = x
                    item.spec.y_mm = y
                    item.setPos(x, y)
                    item.prepareGeometryChange()
                    item._update_handles_pos()
            else:
                # 简单排列：每行 2 个
                cols = 2
                for idx, p in enumerate(paths):
                    row = idx // cols
                    col = idx % cols
                    item = self.canvas_editor.add_new_panel(p)
                    # 简单位移
                    item.setPos(self.canvas_editor.margin + col * 70, 
                                self.canvas_editor.margin + row * 55)
            
            # 重排标签
            self.canvas_editor.reorder_labels()
        except Exception as e:
            print(f"Error updating canvas: {e}")

    # ---------- Preview & Render ----------
    def schedule_preview(self):
        # 聚合高频 UI 修改，减少反复渲染
        self._preview_timer.start(120)

    def _collect_paths_from_list(self) -> List[str]:
        paths = []
        for i in range(self.list_widget.count()):
            it = self.list_widget.item(i)
            p = it.data(Qt.ItemDataRole.UserRole)
            if p:
                paths.append(p)
        # 同步内部路径
        self.image_paths = paths
        return paths

    def on_canvas_size_changed(self):
        """当宽度或高度手动改变时"""
        if self.cmb_layout.currentText() == "自定义模式":
            self.canvas_editor.set_canvas_size(
                float(self.sp_width.value()),
                float(self.sp_height.value()),
                float(self.sp_margin.value())
            )
        self.schedule_preview()

    def on_label_pos_changed(self, pos: str):
        """标注位置模式切换"""
        is_custom = (pos == "自定义")
        self.lbl_label_x.setVisible(is_custom)
        self.slider_label_x.setVisible(is_custom)
        self.lbl_label_y.setVisible(is_custom)
        self.slider_label_y.setVisible(is_custom)
        
        # 更新画布编辑器中所有面板的标注手柄可见性
        if hasattr(self, 'canvas_editor'):
            for item in self.canvas_editor.scene.items():
                if isinstance(item, PanelItem) and item.isSelected():
                    item._show_handles(True)
        
        self.schedule_preview()
    
    def on_custom_label_changed(self):
        """自定义标注位置滑块变化时同步到画布"""
        if hasattr(self, 'canvas_editor') and self.cmb_label_pos.currentText() == "自定义":
            x_ratio = self.slider_label_x.value() / 100.0
            y_ratio = self.slider_label_y.value() / 100.0
            for item in self.canvas_editor.scene.items():
                if isinstance(item, PanelItem):
                    item._update_label_handle_pos(x_ratio, y_ratio)
    
    def on_border_changed(self):
        """边框复选框状态改变时，切换相关控件的可见性"""
        enabled = self.chk_border.isChecked()
        self.lbl_border_width.setVisible(enabled)
        self.sp_border_width.setVisible(enabled)
        self.lbl_border_color.setVisible(enabled)
        self.cmb_border_color.setVisible(enabled)
        self.lbl_border_style.setVisible(enabled)
        self.cmb_border_style.setVisible(enabled)
        self.schedule_preview()
    
    def on_single_label_toggle(self):
        """单个子图标注独立控制开关"""
        enabled = self.chk_use_single_label.isChecked()
        self.lbl_single_label_x.setEnabled(enabled)
        self.sp_single_label_x.setEnabled(enabled)
        self.lbl_single_label_y.setEnabled(enabled)
        self.sp_single_label_y.setEnabled(enabled)
        self.lbl_single_label_color.setEnabled(enabled)
        self.cmb_single_label_color.setEnabled(enabled)
        
        # 更新当前选中的面板
        if hasattr(self, 'canvas_editor'):
            selected_items = [item for item in self.canvas_editor.scene.selectedItems() 
                            if isinstance(item, PanelItem)]
            if selected_items:
                panel = selected_items[0]
                if enabled:
                    # 启用独立标注，设置初始值
                    panel.spec.label_x = self.sp_single_label_x.value() / 100.0
                    panel.spec.label_y = self.sp_single_label_y.value() / 100.0
                    panel.spec.label_color = self.cmb_single_label_color.currentText()
                else:
                    # 禁用独立标注，清除设置
                    panel.spec.label_x = None
                    panel.spec.label_y = None
                    panel.spec.label_color = None
                panel.update()
                self.schedule_preview()
    
    def on_single_label_pos_changed(self):
        """单个子图标注位置改变"""
        if not self.chk_use_single_label.isChecked():
            return
        
        if hasattr(self, 'canvas_editor'):
            selected_items = [item for item in self.canvas_editor.scene.selectedItems() 
                            if isinstance(item, PanelItem)]
            if selected_items:
                panel = selected_items[0]
                panel.spec.label_x = self.sp_single_label_x.value() / 100.0
                panel.spec.label_y = self.sp_single_label_y.value() / 100.0
                panel.update()
                self.schedule_preview()
    
    def on_single_label_color_changed(self):
        """单个子图标注颜色改变"""
        if not self.chk_use_single_label.isChecked():
            return
        
        if hasattr(self, 'canvas_editor'):
            selected_items = [item for item in self.canvas_editor.scene.selectedItems() 
                            if isinstance(item, PanelItem)]
            if selected_items:
                panel = selected_items[0]
                panel.spec.label_color = self.cmb_single_label_color.currentText()
                panel.update()
                self.schedule_preview()
    
    def _get_config(self) -> RenderConfig:
        layout_name = self.cmb_layout.currentText()
        width = float(self.sp_width.value())
        height = float(self.sp_height.value())
        
        # 模板布局的智能高度建议
        if layout_name != "自定义模式":
            if layout_name in ("3×2 网格",):
                height = width * 0.70
            elif layout_name in ("2×2 网格",):
                height = width * 0.90
            elif layout_name in ("1+2（上大下两小）", "2+1（上两小下大）", "左大右两小", "右大左两小"):
                height = width * 0.85
            
            # 更新 UI 显示推荐高度
            self.sp_height.blockSignals(True)
            self.sp_height.setValue(height)
            self.sp_height.blockSignals(False)

        custom_specs = None
        # 始终从画布编辑器导出布局（包含图片编辑参数）
        custom_specs = self.canvas_editor.export_layout()
        
        return RenderConfig(
            layout_name=layout_name,
            canvas_width_mm=width,
            canvas_height_mm=height,
            dpi=int(self.sp_dpi.currentText()),
            margin_mm=float(self.sp_margin.value()),
            gap_mm=float(self.sp_gap.value()),
            fit_mode=self.cmb_fit_mode.currentText(),
            auto_trim=bool(self.chk_trim.isChecked()),
            label_enabled=bool(self.chk_label.isChecked()),
            label_pos=self.cmb_label_pos.currentText(),
            label_size_pt=int(self.sp_label_pt.value()),
            label_padding_mm=float(self.sp_label_pad.value()),
            title_text=self.ed_title.text(),
            title_enabled=bool(self.chk_title.isChecked()),
            title_size_pt=int(self.sp_title_pt.value()),
            background_white=True,
            custom_specs=custom_specs,
            label_custom_x=self.slider_label_x.value() / 100.0,
            label_custom_y=self.slider_label_y.value() / 100.0,
            label_color=self.cmb_label_color.currentText(),
            label_style=self.cmb_label_style.currentText(),
            border_enabled=bool(self.chk_border.isChecked()),
            border_width=float(self.sp_border_width.value()),
            border_color=self.cmb_border_color.currentText(),
            border_style=self.cmb_border_style.currentText(),
            label_font_weight="Regular",
            label_font_italic=False
        )

    def update_preview(self):
        cfg = self._get_config()
        paths = self._collect_paths_from_list()

        try:
            img, warnings = render_montage(paths, cfg)
        except Exception as e:
            self.preview_label.setText(f"渲染失败：{e}")
            self.last_render = None
            return

        # 预览缩放显示（不改变导出分辨率）
        self.last_render = img
        pix = pil_to_qpixmap(img)
        
        # 检查 pixmap 是否有效
        if pix is None or pix.isNull():
            self.preview_label.setText("渲染预览失败")
            return
        
        # 更新预览标签
        target = self.preview_label.size() - QSize(20, 20)
        # 确保目标尺寸有效
        if target.width() <= 0 or target.height() <= 0:
            target = QSize(300, 200)  # 使用默认最小尺寸
        pix2 = pix.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.preview_label.setPixmap(pix2)
        
        # 刷新画布编辑器面板显示
        for item in self.canvas_editor.scene.items():
            if isinstance(item, PanelItem):
                item.update()

        if warnings:
            # 只在标题显示第一条，避免打扰；导出时会再提示
            self.statusBar().showMessage(warnings[0], 6000)
        else:
            self.statusBar().clearMessage()

    # ========== 撤销/重做功能 ==========
    def _save_state(self):
        """保存当前状态到撤销栈"""
        state = {
            'image_paths': self.image_paths.copy(),
            'layout': self.canvas_editor.export_layout(),
            'settings': {
                'layout_name': self.cmb_layout.currentText(),
                'width': self.sp_width.value(),
                'height': self.sp_height.value(),
                'dpi': self.sp_dpi.currentText(),
                'margin': self.sp_margin.value(),
                'gap': self.sp_gap.value(),
                'fit_mode': self.cmb_fit_mode.currentText(),
                'auto_trim': self.chk_trim.isChecked(),
                'label_enabled': self.chk_label.isChecked(),
                'label_pos': self.cmb_label_pos.currentText(),
                'label_size': self.sp_label_pt.value(),
                'label_padding': self.sp_label_pad.value(),
                'title_enabled': self.chk_title.isChecked(),
                'title_text': self.ed_title.text(),
                'title_size': self.sp_title_pt.value(),
                'border_enabled': self.chk_border.isChecked() if hasattr(self, 'chk_border') else False,
                'border_width': self.sp_border_width.value() if hasattr(self, 'sp_border_width') else 1.0,
                'border_color': self.cmb_border_color.currentText() if hasattr(self, 'cmb_border_color') else "黑色",
                'border_style': self.cmb_border_style.currentText() if hasattr(self, 'cmb_border_style') else "实线",
            }
        }
        
        self.undo_stack.append(state)
        if len(self.undo_stack) > self.max_undo_steps:
            self.undo_stack.pop(0)
        
        # 保存新状态后清空重做栈
        self.redo_stack.clear()
        
        # 更新按钮状态
        self.act_undo.setEnabled(len(self.undo_stack) > 1)
        self.act_redo.setEnabled(False)
    
    def _restore_state(self, state):
        """恢复状态"""
        # 恢复图片列表
        self.image_paths = state['image_paths'].copy()
        self.list_widget.clear()
        for path in self.image_paths:
            self._add_to_list(path)
        
        # 恢复布局
        self.canvas_editor.import_layout(state['layout'])
        
        # 恢复设置
        settings = state['settings']
        self.cmb_layout.setCurrentText(settings.get('layout_name', '2x2'))
        self.sp_width.setValue(settings.get('width', 180))
        self.sp_height.setValue(settings.get('height', 120))
        self.sp_dpi.setCurrentText(settings.get('dpi', '300'))
        self.sp_margin.setValue(settings.get('margin', 5.0))
        self.sp_gap.setValue(settings.get('gap', 2.0))
        self.cmb_fit_mode.setCurrentText(settings.get('fit_mode', '智能填充（推荐）'))
        self.chk_trim.setChecked(settings.get('auto_trim', False))
        self.chk_label.setChecked(settings.get('label_enabled', True))
        self.cmb_label_pos.setCurrentText(settings.get('label_pos', '左上'))
        self.sp_label_pt.setValue(settings.get('label_size', 10))
        self.sp_label_pad.setValue(settings.get('label_padding', 1.5))
        self.chk_title.setChecked(settings.get('title_enabled', False))
        self.ed_title.setText(settings.get('title_text', ''))
        self.sp_title_pt.setValue(settings.get('title_size', 12))
        
        if hasattr(self, 'chk_border'):
            self.chk_border.setChecked(settings.get('border_enabled', True))
            self.sp_border_width.setValue(settings.get('border_width', 1.0))
            self.cmb_border_color.setCurrentText(settings.get('border_color', '黑色'))
            self.cmb_border_style.setCurrentText(settings.get('border_style', '实线'))
        
        self.schedule_preview()
    
    def on_undo(self):
        """撤销"""
        if len(self.undo_stack) <= 1:
            return
        
        # 将当前状态移到重做栈
        current_state = self.undo_stack.pop()
        self.redo_stack.append(current_state)
        
        # 恢复上一个状态
        previous_state = self.undo_stack[-1]
        self._restore_state(previous_state)
        
        # 更新按钮状态
        self.act_undo.setEnabled(len(self.undo_stack) > 1)
        self.act_redo.setEnabled(len(self.redo_stack) > 0)
    
    def on_redo(self):
        """重做"""
        if not self.redo_stack:
            return
        
        # 从重做栈取出状态
        state = self.redo_stack.pop()
        self.undo_stack.append(state)
        
        # 恢复状态
        self._restore_state(state)
        
        # 更新按钮状态
        self.act_undo.setEnabled(len(self.undo_stack) > 1)
        self.act_redo.setEnabled(len(self.redo_stack) > 0)
    
    # ========== 布局保存/加载功能 ==========
    def on_save_layout(self):
        """保存布局"""
        if not self.layout_manager:
            QMessageBox.warning(self, "功能不可用", "布局管理器未加载")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "保存布局", self.layout_manager.layouts_dir, 
            "布局文件 (*.layout);;所有文件 (*)"
        )
        
        if not filepath:
            return
        
        # 收集当前布局数据
        layout_data = {
            'image_paths': self.image_paths,
            'panels': self.canvas_editor.export_layout(),
            'settings': {
                'layout_name': self.cmb_layout.currentText(),
                'canvas_width_mm': self.sp_width.value(),
                'canvas_height_mm': self.sp_height.value(),
                'dpi': int(self.sp_dpi.currentText()),
                'margin_mm': self.sp_margin.value(),
                'gap_mm': self.sp_gap.value(),
                'fit_mode': self.cmb_fit_mode.currentText(),
                'auto_trim': self.chk_trim.isChecked(),
                'label_enabled': self.chk_label.isChecked(),
                'label_pos': self.cmb_label_pos.currentText(),
                'label_size_pt': self.sp_label_pt.value(),
                'label_padding_mm': self.sp_label_pad.value(),
                'title_text': self.ed_title.text(),
                'title_enabled': self.chk_title.isChecked(),
                'title_size_pt': self.sp_title_pt.value(),
            }
        }
        
        # 添加边框设置
        if hasattr(self, 'chk_border'):
            layout_data['settings']['border_enabled'] = self.chk_border.isChecked()
            layout_data['settings']['border_width'] = self.sp_border_width.value()
            layout_data['settings']['border_color'] = self.cmb_border_color.currentText()
            layout_data['settings']['border_style'] = self.cmb_border_style.currentText()
        
        # 保存
        if self.layout_manager.save_layout(filepath, layout_data):
            QMessageBox.information(self, "保存成功", f"布局已保存到：\n{filepath}")
        else:
            QMessageBox.critical(self, "保存失败", "布局保存失败，请检查文件权限")
    
    def on_load_layout(self):
        """加载布局"""
        if not self.layout_manager:
            QMessageBox.warning(self, "功能不可用", "布局管理器未加载")
            return
        
        filepath, _ = QFileDialog.getOpenFileName(
            self, "加载布局", self.layout_manager.layouts_dir,
            "布局文件 (*.layout);;所有文件 (*)"
        )
        
        if not filepath:
            return
        
        # 加载
        layout_data = self.layout_manager.load_layout(filepath)
        if not layout_data:
            QMessageBox.critical(self, "加载失败", "无法加载布局文件")
            return
        
        # 恢复图片列表
        self.image_paths = layout_data.get('image_paths', [])
        self.list_widget.clear()
        for path in self.image_paths:
            self._add_to_list(path)
        
        # 恢复布局
        panels = layout_data.get('panels', [])
        self.canvas_editor.import_layout(panels)
        
        # 恢复设置
        settings = layout_data.get('settings', {})
        if settings:
            self.cmb_layout.setCurrentText(settings.get('layout_name', '2x2'))
            self.sp_width.setValue(settings.get('canvas_width_mm', 180))
            self.sp_height.setValue(settings.get('canvas_height_mm', 120))
            self.sp_dpi.setCurrentText(str(settings.get('dpi', 300)))
            self.sp_margin.setValue(settings.get('margin_mm', 5.0))
            self.sp_gap.setValue(settings.get('gap_mm', 2.0))
            self.cmb_fit_mode.setCurrentText(settings.get('fit_mode', '智能填充（推荐）'))
            self.chk_trim.setChecked(settings.get('auto_trim', False))
            self.chk_label.setChecked(settings.get('label_enabled', True))
            self.cmb_label_pos.setCurrentText(settings.get('label_pos', '左上'))
            self.sp_label_pt.setValue(settings.get('label_size_pt', 10))
            self.sp_label_pad.setValue(settings.get('label_padding_mm', 1.5))
            self.chk_title.setChecked(settings.get('title_enabled', False))
            self.ed_title.setText(settings.get('title_text', ''))
            self.sp_title_pt.setValue(settings.get('title_size_pt', 12))
            
            if hasattr(self, 'chk_border'):
                self.chk_border.setChecked(settings.get('border_enabled', True))
                self.sp_border_width.setValue(settings.get('border_width', 1.0))
                self.cmb_border_color.setCurrentText(settings.get('border_color', '黑色'))
                self.cmb_border_style.setCurrentText(settings.get('border_style', '实线'))
        
        self.schedule_preview()
        QMessageBox.information(self, "加载成功", f"布局已从以下文件加载：\n{filepath}")
    
    # ========== 快捷键支持 ==========
    def keyPressEvent(self, event):
        """处理快捷键"""
        key = event.key()
        modifiers = event.modifiers()
        
        # Ctrl+A: 全选面板
        if modifiers == Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_A:
            for item in self.canvas_editor.scene.items():
                if isinstance(item, PanelItem):
                    item.setSelected(True)
            event.accept()
            return
        
        # Esc: 取消选择
        if key == Qt.Key.Key_Escape:
            self.canvas_editor.scene.clearSelection()
            event.accept()
            return
        
        super().keyPressEvent(event)

    # ---------- Export ----------
    def on_export(self):
        cfg = self._get_config()
        paths = self._collect_paths_from_list()

        if not paths and not cfg.custom_specs:
            QMessageBox.warning(self, "提示", "请先导入图片。")
            return

        try:
            img, warnings = render_montage(paths, cfg)
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))
            return

        # 获取选择的格式
        export_format = self.cmb_export_format.currentText()
        
        # 设置文件扩展名和过滤器
        format_map = {
            "PNG": ("png", "PNG 图片 (*.png)"),
            "JPEG": ("jpg", "JPEG 图片 (*.jpg *.jpeg)"),
            "TIFF": ("tif", "TIFF 图片 (*.tif *.tiff)")
        }
        
        ext, file_filter = format_map.get(export_format, ("png", "PNG 图片 (*.png)"))
        default_name = f"figure_montage.{ext}"
        
        # 选择导出路径
        out_file, _ = QFileDialog.getSaveFileName(
            self, f"导出 {export_format}", default_name, file_filter
        )
        if not out_file:
            return
        
        # 确保文件扩展名正确
        if not out_file.lower().endswith(f".{ext}") and not (
            export_format == "JPEG" and out_file.lower().endswith(".jpeg")
        ) and not (
            export_format == "TIFF" and out_file.lower().endswith(".tiff")
        ):
            out_file += f".{ext}"

        # 保存图片
        try:
            if export_format == "PNG":
                img.save(out_file, format="PNG", dpi=(cfg.dpi, cfg.dpi))
            elif export_format == "JPEG":
                # JPEG 不支持透明度，需要转换为 RGB
                if img.mode in ("RGBA", "LA", "P"):
                    # 创建白色背景
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode == "P":
                        img = img.convert("RGBA")
                    background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(out_file, format="JPEG", dpi=(cfg.dpi, cfg.dpi), quality=95)
            elif export_format == "TIFF":
                img.save(out_file, format="TIFF", dpi=(cfg.dpi, cfg.dpi), compression="tiff_deflate")
        except Exception as e:
            QMessageBox.critical(self, f"{export_format} 导出失败", str(e))
            return

        msg = f"已导出：\n{out_file}\n\n格式：{export_format}\nDPI：{cfg.dpi}"
        if warnings:
            msg += "\n\n注意：\n- " + "\n- ".join(warnings)
        QMessageBox.information(self, "导出完成", msg)



def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

