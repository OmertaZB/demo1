import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
from sort_algorithms import bubble_sort, selection_sort, insertion_sort, quick_sort, merge_sort
import matplotlib

"""
排序算法可视化工具
用于动态展示排序算法的执行过程，帮助理解算法原理
"""

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class SortingVisualizer:
    def __init__(self, array_size=50, max_val=100, interval=50):
        """
        初始化可视化工具
        
        参数:
        array_size: 数组大小
        max_val: 数组中的最大值
        interval: 动画帧之间的间隔(毫秒)
        """
        self.array_size = array_size
        self.max_val = max_val
        self.interval = interval
        self.array = np.random.randint(1, max_val, array_size)
        
        # 用于存储排序过程中的数组状态
        self.array_history = []
        
    def reset_array(self):
        """重置数组为新的随机数组"""
        self.array = np.random.randint(1, self.max_val, self.array_size)
        self.array_history = []
        
    def _update_history(self, array):
        """更新数组历史记录"""
        self.array_history.append(array.copy())
        
    def bubble_sort_visualize(self):
        """执行冒泡排序并记录排序过程"""
        self.reset_array()
        arr = self.array.copy()
        n = len(arr)
        
        self._update_history(arr)
        
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    self._update_history(arr)
        
        return self.array_history
    
    def selection_sort_visualize(self):
        """执行选择排序并记录排序过程"""
        self.reset_array()
        arr = self.array.copy()
        n = len(arr)
        
        self._update_history(arr)
        
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            self._update_history(arr)
        
        return self.array_history
    
    def insertion_sort_visualize(self):
        """执行插入排序并记录排序过程"""
        self.reset_array()
        arr = self.array.copy()
        n = len(arr)
        
        self._update_history(arr)
        
        for i in range(1, n):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
                self._update_history(arr)
            arr[j + 1] = key
            self._update_history(arr)
        
        return self.array_history
    
    def visualize_sort(self, sort_name):
        """
        可视化指定的排序算法
        
        参数:
        sort_name: 排序算法名称，可选值: 'bubble', 'selection', 'insertion'
        """
        if sort_name == 'bubble':
            history = self.bubble_sort_visualize()
            title = '冒泡排序可视化'
        elif sort_name == 'selection':
            history = self.selection_sort_visualize()
            title = '选择排序可视化'
        elif sort_name == 'insertion':
            history = self.insertion_sort_visualize()
            title = '插入排序可视化'
        else:
            raise ValueError("不支持的排序算法名称")
        
        # 创建图形和轴
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 设置图表标题和轴标签
        ax.set_title(title, fontsize=15)
        ax.set_xlabel('数组索引', fontsize=12)
        ax.set_ylabel('值', fontsize=12)
        
        # 创建条形图
        bar_rects = ax.bar(range(len(self.array)), history[0], align='edge')
        
        # 设置轴范围
        ax.set_xlim(0, self.array_size)
        ax.set_ylim(0, self.max_val * 1.1)
        
        # 添加迭代文本
        iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
        
        # 更新函数
        def update_plot(frame_number):
            for rect, val in zip(bar_rects, history[frame_number]):
                rect.set_height(val)
            iteration_text.set_text(f'迭代次数: {frame_number}')
            return bar_rects
        
        # 创建动画
        anim = animation.FuncAnimation(
            fig, 
            update_plot, 
            frames=len(history), 
            interval=self.interval,
            blit=False, 
            repeat=False
        )
        
        plt.tight_layout()
        plt.show()
        
        # 保存动画为GIF（可选）
        # anim.save(f'{sort_name}_sort_visualization.gif', writer='pillow')
        
        return anim

# 示例用法
if __name__ == "__main__":
    # 创建可视化工具实例
    visualizer = SortingVisualizer(array_size=30, max_val=100, interval=100)
    
    # 可视化冒泡排序
    # visualizer.visualize_sort('bubble')
    
    # 可视化选择排序
    visualizer.visualize_sort('selection')
    
    # 可视化插入排序
    # visualizer.visualize_sort('insertion') 