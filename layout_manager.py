# -*- coding: utf-8 -*-
"""
布局管理器 - 负责保存和加载布局配置
"""

import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any


class LayoutManager:
    """布局配置管理器"""
    
    def __init__(self):
        self.layouts_dir = "layouts"
        self.recent_file = "config/recent_layouts.json"
        self.max_recent = 10
        
        # 确保目录存在
        os.makedirs(self.layouts_dir, exist_ok=True)
        os.makedirs("config", exist_ok=True)
    
    def save_layout(self, filepath: str, layout_data: Dict[str, Any]) -> bool:
        """
        保存布局到文件
        
        Args:
            filepath: 保存路径（.layout 文件）
            layout_data: 布局数据字典
        
        Returns:
            bool: 是否保存成功
        """
        try:
            # 确保扩展名
            if not filepath.endswith('.layout'):
                filepath += '.layout'
            
            # 添加元数据
            layout_data['metadata'] = {
                'version': '2.0',
                'saved_at': datetime.now().isoformat(),
                'app': '论文组图工具'
            }
            
            # 保存为JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(layout_data, f, ensure_ascii=False, indent=2)
            
            # 更新最近使用列表
            self._add_to_recent(filepath)
            
            return True
        except Exception as e:
            print(f"Error saving layout: {e}")
            return False
    
    def load_layout(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        从文件加载布局
        
        Args:
            filepath: 文件路径
        
        Returns:
            Dict: 布局数据，失败返回None
        """
        try:
            if not os.path.exists(filepath):
                print(f"Layout file not found: {filepath}")
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                layout_data = json.load(f)
            
            # 验证版本（可选）
            version = layout_data.get('metadata', {}).get('version', '1.0')
            if version != '2.0':
                print(f"Warning: Layout version {version} may not be fully compatible")
            
            # 更新最近使用列表
            self._add_to_recent(filepath)
            
            return layout_data
        except Exception as e:
            print(f"Error loading layout: {e}")
            return None
    
    def get_recent_layouts(self, limit: int = None) -> List[str]:
        """
        获取最近使用的布局文件列表
        
        Args:
            limit: 限制数量，默认使用 max_recent
        
        Returns:
            List[str]: 文件路径列表
        """
        if limit is None:
            limit = self.max_recent
        
        try:
            if os.path.exists(self.recent_file):
                with open(self.recent_file, 'r', encoding='utf-8') as f:
                    recent = json.load(f)
                
                # 过滤掉不存在的文件
                recent = [f for f in recent if os.path.exists(f)]
                
                return recent[:limit]
        except Exception as e:
            print(f"Error loading recent layouts: {e}")
        
        return []
    
    def _add_to_recent(self, filepath: str):
        """添加文件到最近使用列表"""
        try:
            recent = []
            if os.path.exists(self.recent_file):
                with open(self.recent_file, 'r', encoding='utf-8') as f:
                    recent = json.load(f)
            
            # 移除重复项
            if filepath in recent:
                recent.remove(filepath)
            
            # 添加到开头
            recent.insert(0, filepath)
            
            # 限制数量
            recent = recent[:self.max_recent]
            
            # 保存
            with open(self.recent_file, 'w', encoding='utf-8') as f:
                json.dump(recent, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error updating recent layouts: {e}")
    
    def clear_recent(self):
        """清空最近使用列表"""
        try:
            if os.path.exists(self.recent_file):
                os.remove(self.recent_file)
        except Exception as e:
            print(f"Error clearing recent layouts: {e}")
    
    def export_layout_data(self, canvas_editor, config) -> Dict[str, Any]:
        """
        从画布编辑器导出布局数据
        
        Args:
            canvas_editor: CanvasEditorWidget 实例
            config: 当前的配置参数
        
        Returns:
            Dict: 布局数据
        """
        # 1. 获取所有面板数据
        from Figure_paper import PanelItem
        
        panels = []
        panel_items = [i for i in canvas_editor.scene.items() if isinstance(i, PanelItem)]
        panel_items.sort(key=lambda x: x.spec.label_index)
        
        for item in panel_items:
            spec = item.spec
            panel_data = {
                'id': spec.id,
                'x_mm': spec.x_mm,
                'y_mm': spec.y_mm,
                'w_mm': spec.w_mm,
                'h_mm': spec.h_mm,
                'image_path': spec.image_path,
                'label_index': spec.label_index,
                'image_offset_x': spec.image_offset_x,
                'image_offset_y': spec.image_offset_y,
                'image_scale': spec.image_scale,
                'label_offset_x': spec.label_offset_x,
                'label_offset_y': spec.label_offset_y,
            }
            panels.append(panel_data)

        # 2. 构建完整的布局数据（移除图形标注）
        layout_data = {
            'canvas': {
                'width_mm': canvas_editor.canvas_w,
                'height_mm': canvas_editor.canvas_h,
                'margin_mm': canvas_editor.margin
            },
            'panels': panels,
            'settings': config  # 保存所有设置
        }
        
        return layout_data
