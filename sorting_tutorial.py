import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import matplotlib

"""
排序算法教学工具
提供交互式学习体验，帮助理解各种排序算法的工作原理
"""

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class SortingTutorial:
    def __init__(self, root):
        """初始化排序算法教学工具"""
        self.root = root
        self.root.title("排序算法教学工具")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # 当前选择的算法
        self.current_algorithm = tk.StringVar(value="冒泡排序")
        
        # 数组大小和速度
        self.array_size = tk.IntVar(value=20)
        self.speed = tk.IntVar(value=50)  # 动画速度（毫秒）
        
        # 数组数据
        self.array = np.random.randint(1, 100, self.array_size.get())
        self.array_history = []
        self.animation = None
        self.is_sorting = False
        
        # 算法说明 - 移到这里，在创建UI之前
        self.algorithm_descriptions = {
            "冒泡排序": """
冒泡排序是一种简单的排序算法，它重复地遍历要排序的数列，一次比较两个元素，如果它们的顺序错误就交换它们。

工作原理:
1. 比较相邻的元素。如果第一个比第二个大，就交换它们。
2. 对每一对相邻元素做同样的工作，从开始第一对到结尾的最后一对。这步做完后，最大的元素会"冒泡"到最后的位置。
3. 针对所有的元素重复以上的步骤，除了最后一个已排序的元素。
4. 重复步骤1~3，直到没有任何一对数字需要比较。

时间复杂度: O(n²)
空间复杂度: O(1)
稳定性: 稳定
            """,
            "选择排序": """
选择排序是一种简单直观的排序算法，它每次从未排序部分找出最小元素，放到已排序部分的末尾。

工作原理:
1. 在未排序序列中找到最小（大）元素，存放到排序序列的起始位置。
2. 从剩余未排序元素中继续寻找最小（大）元素，放到已排序序列的末尾。
3. 重复第二步，直到所有元素均排序完毕。

时间复杂度: O(n²)
空间复杂度: O(1)
稳定性: 不稳定
            """,
            "插入排序": """
插入排序是一种简单直观的排序算法，它的工作原理是通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。

工作原理:
1. 从第一个元素开始，该元素可以认为已经被排序。
2. 取出下一个元素，在已经排序的元素序列中从后向前扫描。
3. 如果该元素（已排序）大于新元素，将该元素移到下一位置。
4. 重复步骤3，直到找到已排序的元素小于或等于新元素的位置。
5. 将新元素插入到该位置后。
6. 重复步骤2~5。

时间复杂度: O(n²)
空间复杂度: O(1)
稳定性: 稳定
            """,
            "快速排序": """
快速排序是一种分治的排序算法，它通过一趟排序将要排序的数据分割成独立的两部分，其中一部分的所有数据都比另外一部分的所有数据都要小，然后再按此方法对这两部分数据分别进行快速排序。

工作原理:
1. 从数列中挑出一个元素，称为"基准"（pivot）。
2. 重新排序数列，所有比基准值小的元素摆放在基准前面，所有比基准值大的元素摆在基准后面。在这个分区结束之后，该基准就处于数列的中间位置。这个称为分区（partition）操作。
3. 递归地把小于基准值元素的子数列和大于基准值元素的子数列排序。

时间复杂度: 平均O(n log n)，最坏O(n²)
空间复杂度: O(log n)
稳定性: 不稳定
            """,
            "归并排序": """
归并排序是一种分治算法，它将数组分成两半，递归地排序两半，然后合并两半。

工作原理:
1. 将长度为n的输入序列分成两个长度为n/2的子序列。
2. 对这两个子序列分别采用归并排序。
3. 将两个排序好的子序列合并成一个最终的排序序列。

时间复杂度: O(n log n)
空间复杂度: O(n)
稳定性: 稳定
            """
        }
        
        # 创建UI
        self._create_ui()
    
    def _create_ui(self):
        """创建用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 算法选择
        ttk.Label(control_frame, text="选择算法:").pack(anchor=tk.W, pady=(0, 5))
        algorithms = ["冒泡排序", "选择排序", "插入排序", "快速排序", "归并排序"]
        algorithm_combo = ttk.Combobox(control_frame, textvariable=self.current_algorithm, values=algorithms, state="readonly")
        algorithm_combo.pack(fill=tk.X, pady=(0, 10))
        algorithm_combo.bind("<<ComboboxSelected>>", self._on_algorithm_change)
        
        # 数组大小
        ttk.Label(control_frame, text="数组大小:").pack(anchor=tk.W, pady=(0, 5))
        size_scale = ttk.Scale(control_frame, from_=5, to=50, variable=self.array_size, orient=tk.HORIZONTAL)
        size_scale.pack(fill=tk.X, pady=(0, 10))
        size_scale.bind("<ButtonRelease-1>", self._on_size_change)
        
        # 动画速度
        ttk.Label(control_frame, text="动画速度:").pack(anchor=tk.W, pady=(0, 5))
        speed_scale = ttk.Scale(control_frame, from_=1, to=200, variable=self.speed, orient=tk.HORIZONTAL)
        speed_scale.pack(fill=tk.X, pady=(0, 10))
        
        # 按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="重置数组", command=self._reset_array).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="开始排序", command=self._start_sorting).pack(side=tk.LEFT)
        
        # 创建右侧内容区域
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 创建可视化区域
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=content_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 创建算法说明区域
        description_frame = ttk.LabelFrame(content_frame, text="算法说明", padding="10")
        description_frame.pack(fill=tk.BOTH, expand=True)
        
        self.description_text = scrolledtext.ScrolledText(description_frame, wrap=tk.WORD, height=10)
        self.description_text.pack(fill=tk.BOTH, expand=True)
        self.description_text.config(state=tk.NORMAL)
        self.description_text.delete(1.0, tk.END)
        self.description_text.insert(tk.END, self.algorithm_descriptions[self.current_algorithm.get()])
        self.description_text.config(state=tk.DISABLED)
        
        # 初始化可视化
        self._update_visualization()
    
    def _on_algorithm_change(self, event):
        """算法选择改变时的回调"""
        # 更新算法说明
        self.description_text.config(state=tk.NORMAL)
        self.description_text.delete(1.0, tk.END)
        self.description_text.insert(tk.END, self.algorithm_descriptions[self.current_algorithm.get()])
        self.description_text.config(state=tk.DISABLED)
    
    def _on_size_change(self, event):
        """数组大小改变时的回调"""
        self._reset_array()
    
    def _reset_array(self):
        """重置数组"""
        if self.is_sorting:
            messagebox.showinfo("提示", "排序正在进行中，请等待完成或关闭窗口重新开始")
            return
        
        self.array = np.random.randint(1, 100, self.array_size.get())
        self._update_visualization()
    
    def _update_visualization(self):
        """更新可视化"""
        self.ax.clear()
        self.ax.bar(range(len(self.array)), self.array, align='center', alpha=0.7)
        self.ax.set_title(f"{self.current_algorithm.get()}可视化")
        self.ax.set_xlabel("索引")
        self.ax.set_ylabel("值")
        self.ax.set_xlim(-1, len(self.array))
        self.ax.set_ylim(0, max(self.array) * 1.1)
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _start_sorting(self):
        """开始排序"""
        if self.is_sorting:
            messagebox.showinfo("提示", "排序正在进行中，请等待完成或关闭窗口重新开始")
            return
        
        self.is_sorting = True
        self.array_history = [self.array.copy()]
        
        # 根据选择的算法执行相应的排序
        algorithm = self.current_algorithm.get()
        if algorithm == "冒泡排序":
            self._bubble_sort()
        elif algorithm == "选择排序":
            self._selection_sort()
        elif algorithm == "插入排序":
            self._insertion_sort()
        elif algorithm == "快速排序":
            arr = self.array.copy()
            self._quick_sort(arr, 0, len(arr) - 1)
        elif algorithm == "归并排序":
            arr = self.array.copy()
            self._merge_sort(arr, 0, len(arr) - 1)
        
        # 创建动画
        self._create_animation()
    
    def _bubble_sort(self):
        """冒泡排序"""
        arr = self.array.copy()
        n = len(arr)
        
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    self.array_history.append(arr.copy())
    
    def _selection_sort(self):
        """选择排序"""
        arr = self.array.copy()
        n = len(arr)
        
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            self.array_history.append(arr.copy())
    
    def _insertion_sort(self):
        """插入排序"""
        arr = self.array.copy()
        n = len(arr)
        
        for i in range(1, n):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
                self.array_history.append(arr.copy())
            arr[j + 1] = key
            self.array_history.append(arr.copy())
    
    def _quick_sort(self, arr, low, high):
        """快速排序"""
        if low < high:
            # 分区操作，返回基准元素的索引
            pivot_idx = self._partition(arr, low, high)
            
            # 递归排序基准元素左右两侧的子数组
            self._quick_sort(arr, low, pivot_idx - 1)
            self._quick_sort(arr, pivot_idx + 1, high)
    
    def _partition(self, arr, low, high):
        """快速排序的分区操作"""
        # 选择最右边的元素作为基准
        pivot = arr[high]
        i = low - 1
        
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                self.array_history.append(arr.copy())
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        self.array_history.append(arr.copy())
        
        return i + 1
    
    def _merge_sort(self, arr, left, right):
        """归并排序"""
        if left < right:
            # 找出中间点
            mid = (left + right) // 2
            
            # 递归排序左右两半
            self._merge_sort(arr, left, mid)
            self._merge_sort(arr, mid + 1, right)
            
            # 合并两半
            self._merge(arr, left, mid, right)
    
    def _merge(self, arr, left, mid, right):
        """归并排序的合并操作"""
        # 创建临时数组
        L = arr[left:mid + 1].copy()
        R = arr[mid + 1:right + 1].copy()
        
        # 初始化指针
        i = j = 0
        k = left
        
        # 合并两个子数组
        while i < len(L) and j < len(R):
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
            self.array_history.append(arr.copy())
        
        # 处理剩余元素
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
            self.array_history.append(arr.copy())
        
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
            self.array_history.append(arr.copy())
    
    def _create_animation(self):
        """创建排序动画"""
        self.ax.clear()
        bars = self.ax.bar(range(len(self.array)), self.array_history[0], align='center', alpha=0.7)
        self.ax.set_title(f"{self.current_algorithm.get()}可视化")
        self.ax.set_xlabel("索引")
        self.ax.set_ylabel("值")
        self.ax.set_xlim(-1, len(self.array))
        self.ax.set_ylim(0, max(self.array) * 1.1)
        
        # 添加迭代文本
        iteration_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)
        
        def update(frame):
            for i, val in enumerate(self.array_history[frame]):
                bars[i].set_height(val)
            iteration_text.set_text(f'迭代次数: {frame}')
            return bars
        
        self.animation = animation.FuncAnimation(
            self.fig, 
            update, 
            frames=len(self.array_history),
            interval=self.speed.get(),
            repeat=False,
            blit=False
        )
        
        self.canvas.draw()
        
        # 动画完成后重置状态
        def on_animation_complete(*args):  # 修改为接受任意参数
            self.is_sorting = False
        
        self.animation.event_source.add_callback(on_animation_complete)

# 主函数
if __name__ == "__main__":
    root = tk.Tk()
    app = SortingTutorial(root)
    root.mainloop() 