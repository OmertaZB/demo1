import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from matplotlib.gridspec import GridSpec
from sort_algorithms import (
    bubble_sort,
    selection_sort,
    insertion_sort,
    quick_sort,
    merge_sort,
    heap_sort
)

"""
排序算法可视化比较工具
同时可视化多种排序算法的执行过程，方便直观比较
"""

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class SortingVisualizerComparison:
    def __init__(self, array_size=30, max_val=100, interval=50):
        """
        初始化排序可视化比较工具
        
        参数:
            array_size: 数组大小
            max_val: 数组中的最大值
            interval: 动画帧之间的间隔（毫秒）
        """
        self.array_size = array_size
        self.max_val = max_val
        self.interval = interval
        
        # 生成随机数组
        self.array = np.random.randint(1, max_val, array_size)
        
        # 排序算法及其历史记录
        self.algorithms = {
            "冒泡排序": {"func": self._bubble_sort, "history": []},
            "选择排序": {"func": self._selection_sort, "history": []},
            "插入排序": {"func": self._insertion_sort, "history": []},
            "快速排序": {"func": self._quick_sort, "history": []},
            "归并排序": {"func": self._merge_sort, "history": []},
            "堆排序": {"func": self._heap_sort, "history": []}
        }
    
    def reset_array(self):
        """重置数组为新的随机数组"""
        self.array = np.random.randint(1, self.max_val, self.array_size)
        
        # 重置所有算法的历史记录
        for algo in self.algorithms.values():
            algo["history"] = []
    
    def _update_history(self, algorithm, array):
        """更新指定算法的历史记录"""
        self.algorithms[algorithm]["history"].append(array.copy())
    
    def _bubble_sort(self):
        """冒泡排序"""
        arr = self.array.copy()
        n = len(arr)
        
        # 记录初始状态
        self._update_history("冒泡排序", arr)
        
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swapped = True
                    self._update_history("冒泡排序", arr)
            
            if not swapped:
                break
    
    def _selection_sort(self):
        """选择排序"""
        arr = self.array.copy()
        n = len(arr)
        
        # 记录初始状态
        self._update_history("选择排序", arr)
        
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            
            if min_idx != i:
                arr[i], arr[min_idx] = arr[min_idx], arr[i]
                self._update_history("选择排序", arr)
    
    def _insertion_sort(self):
        """插入排序"""
        arr = self.array.copy()
        n = len(arr)
        
        # 记录初始状态
        self._update_history("插入排序", arr)
        
        for i in range(1, n):
            key = arr[i]
            j = i - 1
            
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
                self._update_history("插入排序", arr)
            
            arr[j + 1] = key
            if j + 1 != i:  # 只有在实际发生插入时才记录
                self._update_history("插入排序", arr)
    
    def _quick_sort(self):
        """快速排序"""
        arr = self.array.copy()
        
        # 记录初始状态
        self._update_history("快速排序", arr)
        
        def _quick_sort_recursive(arr, low, high):
            if low < high:
                pivot_idx = _partition(arr, low, high)
                _quick_sort_recursive(arr, low, pivot_idx - 1)
                _quick_sort_recursive(arr, pivot_idx + 1, high)
        
        def _partition(arr, low, high):
            pivot = arr[high]
            i = low - 1
            
            for j in range(low, high):
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
                    self._update_history("快速排序", arr)
            
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            self._update_history("快速排序", arr)
            
            return i + 1
        
        _quick_sort_recursive(arr, 0, len(arr) - 1)
    
    def _merge_sort(self):
        """归并排序"""
        arr = self.array.copy()
        
        # 记录初始状态
        self._update_history("归并排序", arr)
        
        def _merge_sort_recursive(arr, left, right):
            if left < right:
                mid = (left + right) // 2
                _merge_sort_recursive(arr, left, mid)
                _merge_sort_recursive(arr, mid + 1, right)
                _merge(arr, left, mid, right)
        
        def _merge(arr, left, mid, right):
            L = arr[left:mid + 1].copy()
            R = arr[mid + 1:right + 1].copy()
            
            i = j = 0
            k = left
            
            while i < len(L) and j < len(R):
                if L[i] <= R[j]:
                    arr[k] = L[i]
                    i += 1
                else:
                    arr[k] = R[j]
                    j += 1
                k += 1
                self._update_history("归并排序", arr)
            
            while i < len(L):
                arr[k] = L[i]
                i += 1
                k += 1
                self._update_history("归并排序", arr)
            
            while j < len(R):
                arr[k] = R[j]
                j += 1
                k += 1
                self._update_history("归并排序", arr)
        
        _merge_sort_recursive(arr, 0, len(arr) - 1)
    
    def _heap_sort(self):
        """堆排序"""
        arr = self.array.copy()
        n = len(arr)
        
        # 记录初始状态
        self._update_history("堆排序", arr)
        
        def _heapify(arr, n, i):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            
            if left < n and arr[left] > arr[largest]:
                largest = left
            
            if right < n and arr[right] > arr[largest]:
                largest = right
            
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                self._update_history("堆排序", arr)
                _heapify(arr, n, largest)
        
        # 构建最大堆
        for i in range(n // 2 - 1, -1, -1):
            _heapify(arr, n, i)
        
        # 一个个从堆中取出元素
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            self._update_history("堆排序", arr)
            _heapify(arr, i, 0)
    
    def run_all_sorts(self):
        """运行所有排序算法"""
        # 重置历史记录
        for algo in self.algorithms.values():
            algo["history"] = []
        
        # 运行每个排序算法
        for name, algo in self.algorithms.items():
            algo["func"]()
    
    def visualize_comparison(self, algorithms=None):
        """
        可视化比较多个排序算法
        
        参数:
            algorithms: 要比较的算法列表，如果为None则比较所有算法
        """
        if algorithms is None:
            algorithms = list(self.algorithms.keys())
        
        # 运行所有排序算法
        self.run_all_sorts()
        
        # 创建图形和子图
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(len(algorithms), 1, figure=fig)
        axes = [fig.add_subplot(gs[i, 0]) for i in range(len(algorithms))]
        
        # 为每个算法创建条形图
        bars = []
        for i, algo_name in enumerate(algorithms):
            ax = axes[i]
            ax.set_title(f"{algo_name}")
            ax.set_xlim(0, self.array_size)
            ax.set_ylim(0, self.max_val * 1.1)
            
            # 创建条形图
            bar = ax.bar(range(self.array_size), self.array, align='center', alpha=0.7)
            bars.append(bar)
            
            # 添加迭代计数器
            ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10)
        
        # 找出最长的历史记录长度
        max_frames = max(len(self.algorithms[algo]["history"]) for algo in algorithms)
        
        # 动画更新函数
        def update(frame):
            for i, algo_name in enumerate(algorithms):
                history = self.algorithms[algo_name]["history"]
                frame_idx = min(frame, len(history) - 1)
                
                # 更新条形图高度
                for j, val in enumerate(history[frame_idx]):
                    bars[i][j].set_height(val)
                
                # 更新迭代计数器
                axes[i].texts[0].set_text(f'迭代: {frame_idx}')
            
            # 当 blit=False 时，不需要返回值
            # return [bar for sublist in bars for bar in sublist]
        
        # 创建动画
        anim = animation.FuncAnimation(
            fig, 
            update, 
            frames=max_frames,
            interval=self.interval,
            blit=False,
            repeat=False
        )
        
        plt.tight_layout()
        plt.show()
        
        return anim

# 示例用法
if __name__ == "__main__":
    # 创建可视化比较工具实例
    visualizer = SortingVisualizerComparison(array_size=30, max_val=100, interval=100)
    
    # 可视化比较所有排序算法
    visualizer.visualize_comparison(["冒泡排序", "选择排序", "插入排序"])
    
    # 重置数组并比较另一组算法
    visualizer.reset_array()
    visualizer.visualize_comparison(["快速排序", "归并排序", "堆排序"]) 