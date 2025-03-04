import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

"""
排序算法比较工具
用于直观比较不同排序算法在相同数据集上的性能差异
"""

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 排序算法实现
def bubble_sort(arr):
    """冒泡排序"""
    arr = arr.copy()
    n = len(arr)
    comparisons = 0
    swaps = 0
    
    for i in range(n):
        for j in range(0, n - i - 1):
            comparisons += 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swaps += 1
    
    return arr, comparisons, swaps

def selection_sort(arr):
    """选择排序"""
    arr = arr.copy()
    n = len(arr)
    comparisons = 0
    swaps = 0
    
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            comparisons += 1
            if arr[j] < arr[min_idx]:
                min_idx = j
        
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            swaps += 1
    
    return arr, comparisons, swaps

def insertion_sort(arr):
    """插入排序"""
    arr = arr.copy()
    n = len(arr)
    comparisons = 0
    swaps = 0
    
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        
        while j >= 0:
            comparisons += 1
            if arr[j] > key:
                arr[j + 1] = arr[j]
                swaps += 1
                j -= 1
            else:
                break
        
        if j + 1 != i:
            arr[j + 1] = key
            swaps += 1
    
    return arr, comparisons, swaps

def quick_sort(arr):
    """快速排序"""
    arr = arr.copy()
    comparisons = [0]
    swaps = [0]
    
    def _quick_sort(arr, low, high):
        if low < high:
            pivot_idx = _partition(arr, low, high)
            _quick_sort(arr, low, pivot_idx - 1)
            _quick_sort(arr, pivot_idx + 1, high)
    
    def _partition(arr, low, high):
        pivot = arr[high]
        i = low - 1
        
        for j in range(low, high):
            comparisons[0] += 1
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                swaps[0] += 1
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        swaps[0] += 1
        
        return i + 1
    
    _quick_sort(arr, 0, len(arr) - 1)
    return arr, comparisons[0], swaps[0]

def merge_sort(arr):
    """归并排序"""
    arr = arr.copy()
    comparisons = [0]
    swaps = [0]  # 在归并排序中，我们计算的是元素移动次数
    
    def _merge_sort(arr, left, right):
        if left < right:
            mid = (left + right) // 2
            _merge_sort(arr, left, mid)
            _merge_sort(arr, mid + 1, right)
            _merge(arr, left, mid, right)
    
    def _merge(arr, left, mid, right):
        L = arr[left:mid + 1].copy()
        R = arr[mid + 1:right + 1].copy()
        
        i = j = 0
        k = left
        
        while i < len(L) and j < len(R):
            comparisons[0] += 1
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
            swaps[0] += 1
        
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
            swaps[0] += 1
        
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
            swaps[0] += 1
    
    _merge_sort(arr, 0, len(arr) - 1)
    return arr, comparisons[0], swaps[0]

def heap_sort(arr):
    """堆排序"""
    arr = arr.copy()
    n = len(arr)
    comparisons = 0
    swaps = 0
    
    def heapify(arr, n, i):
        nonlocal comparisons, swaps
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        if left < n:
            comparisons += 1
            if arr[left] > arr[largest]:
                largest = left
        
        if right < n:
            comparisons += 1
            if arr[right] > arr[largest]:
                largest = right
        
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            swaps += 1
            heapify(arr, n, largest)
    
    # 构建最大堆
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    # 一个个从堆中取出元素
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        swaps += 1
        heapify(arr, i, 0)
    
    return arr, comparisons, swaps

def python_sort(arr):
    """Python内置排序（Timsort）"""
    arr = arr.copy()
    # 由于内置排序无法直接计算比较和交换次数，我们只计时
    arr.sort()
    return arr, 0, 0  # 返回0表示未计算

class SortingComparison:
    def __init__(self, root):
        """初始化排序算法比较工具"""
        self.root = root
        self.root.title("排序算法比较工具")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # 排序算法
        self.algorithms = {
            "冒泡排序": bubble_sort,
            "选择排序": selection_sort,
            "插入排序": insertion_sort,
            "快速排序": quick_sort,
            "归并排序": merge_sort,
            "堆排序": heap_sort,
            "Python内置排序": python_sort
        }
        
        # 数据集类型
        self.data_types = ["随机数据", "已排序数据", "逆序数据", "部分排序数据"]
        
        # 数据集大小
        self.array_size = tk.IntVar(value=100)
        
        # 选择的算法
        self.selected_algorithms = {}
        for algo in self.algorithms:
            self.selected_algorithms[algo] = tk.BooleanVar(value=True)
        
        # 选择的数据类型
        self.selected_data_type = tk.StringVar(value="随机数据")
        
        # 结果数据
        self.results = {}
        
        # 创建UI
        self._create_ui()
        
        # 初始化数据
        self._generate_data()
    
    def _create_ui(self):
        """创建用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 数据集大小
        ttk.Label(control_frame, text="数据集大小:").pack(anchor=tk.W, pady=(0, 5))
        size_scale = ttk.Scale(control_frame, from_=10, to=1000, variable=self.array_size, orient=tk.HORIZONTAL)
        size_scale.pack(fill=tk.X, pady=(0, 10))
        
        # 数据类型选择
        ttk.Label(control_frame, text="数据类型:").pack(anchor=tk.W, pady=(0, 5))
        for data_type in self.data_types:
            ttk.Radiobutton(control_frame, text=data_type, variable=self.selected_data_type, value=data_type).pack(anchor=tk.W)
        
        # 算法选择
        ttk.Label(control_frame, text="选择算法:").pack(anchor=tk.W, pady=(10, 5))
        for algo in self.algorithms:
            ttk.Checkbutton(control_frame, text=algo, variable=self.selected_algorithms[algo]).pack(anchor=tk.W)
        
        # 按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="生成数据", command=self._generate_data).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="运行比较", command=self._run_comparison).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="导出结果", command=self._export_results).pack(side=tk.LEFT)
        
        # 创建右侧内容区域
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 创建选项卡
        tab_control = ttk.Notebook(content_frame)
        tab_control.pack(fill=tk.BOTH, expand=True)
        
        # 时间比较选项卡
        self.time_tab = ttk.Frame(tab_control)
        tab_control.add(self.time_tab, text="时间比较")
        
        # 比较次数选项卡
        self.comparisons_tab = ttk.Frame(tab_control)
        tab_control.add(self.comparisons_tab, text="比较次数")
        
        # 交换次数选项卡
        self.swaps_tab = ttk.Frame(tab_control)
        tab_control.add(self.swaps_tab, text="交换次数")
        
        # 数据可视化选项卡
        self.data_tab = ttk.Frame(tab_control)
        tab_control.add(self.data_tab, text="数据可视化")
        
        # 创建图表
        self._create_charts()
    
    def _create_charts(self):
        """创建图表"""
        # 时间比较图表
        self.time_fig, self.time_ax = plt.subplots(figsize=(8, 5))
        self.time_canvas = FigureCanvasTkAgg(self.time_fig, master=self.time_tab)
        self.time_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 比较次数图表
        self.comp_fig, self.comp_ax = plt.subplots(figsize=(8, 5))
        self.comp_canvas = FigureCanvasTkAgg(self.comp_fig, master=self.comparisons_tab)
        self.comp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 交换次数图表
        self.swap_fig, self.swap_ax = plt.subplots(figsize=(8, 5))
        self.swap_canvas = FigureCanvasTkAgg(self.swap_fig, master=self.swaps_tab)
        self.swap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 数据可视化图表
        self.data_fig, self.data_ax = plt.subplots(figsize=(8, 5))
        self.data_canvas = FigureCanvasTkAgg(self.data_fig, master=self.data_tab)
        self.data_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 更新图表
        self._update_charts()
    
    def _generate_data(self):
        """生成数据集"""
        size = self.array_size.get()
        data_type = self.selected_data_type.get()
        
        if data_type == "随机数据":
            self.data = np.random.randint(1, 1000, size)
        elif data_type == "已排序数据":
            self.data = np.sort(np.random.randint(1, 1000, size))
        elif data_type == "逆序数据":
            self.data = np.sort(np.random.randint(1, 1000, size))[::-1]
        elif data_type == "部分排序数据":
            # 生成部分排序的数据（前一半排序，后一半随机）
            half = size // 2
            self.data = np.zeros(size, dtype=int)
            self.data[:half] = np.sort(np.random.randint(1, 500, half))
            self.data[half:] = np.random.randint(500, 1000, size - half)
        
        # 更新数据可视化
        self._update_data_visualization()
    
    def _update_data_visualization(self):
        """更新数据可视化"""
        self.data_ax.clear()
        self.data_ax.plot(self.data, '-o', markersize=2)
        self.data_ax.set_title(f"{self.selected_data_type.get()} (大小: {len(self.data)})")
        self.data_ax.set_xlabel("索引")
        self.data_ax.set_ylabel("值")
        self.data_fig.tight_layout()
        self.data_canvas.draw()
    
    def _run_comparison(self):
        """运行排序算法比较"""
        # 检查是否选择了算法
        selected = [algo for algo, var in self.selected_algorithms.items() if var.get()]
        if not selected:
            messagebox.showwarning("警告", "请至少选择一种排序算法")
            return
        
        # 重置结果
        self.results = {
            "算法": [],
            "时间 (秒)": [],
            "比较次数": [],
            "交换次数": []
        }
        
        # 运行选中的算法
        for algo_name in selected:
            algo_func = self.algorithms[algo_name]
            
            # 计时
            start_time = time.time()
            sorted_arr, comparisons, swaps = algo_func(self.data)
            end_time = time.time()
            
            # 记录结果
            self.results["算法"].append(algo_name)
            self.results["时间 (秒)"].append(end_time - start_time)
            self.results["比较次数"].append(comparisons)
            self.results["交换次数"].append(swaps)
        
        # 更新图表
        self._update_charts()
    
    def _update_charts(self):
        """更新所有图表"""
        if not self.results or not self.results["算法"]:
            return
        
        # 更新时间比较图表
        self.time_ax.clear()
        self.time_ax.bar(self.results["算法"], self.results["时间 (秒)"])
        self.time_ax.set_title("排序算法时间比较")
        self.time_ax.set_xlabel("算法")
        self.time_ax.set_ylabel("时间 (秒)")
        plt.setp(self.time_ax.get_xticklabels(), rotation=45, ha="right")
        self.time_fig.tight_layout()
        self.time_canvas.draw()
        
        # 更新比较次数图表
        self.comp_ax.clear()
        self.comp_ax.bar(self.results["算法"], self.results["比较次数"])
        self.comp_ax.set_title("排序算法比较次数")
        self.comp_ax.set_xlabel("算法")
        self.comp_ax.set_ylabel("比较次数")
        plt.setp(self.comp_ax.get_xticklabels(), rotation=45, ha="right")
        self.comp_fig.tight_layout()
        self.comp_canvas.draw()
        
        # 更新交换次数图表
        self.swap_ax.clear()
        self.swap_ax.bar(self.results["算法"], self.results["交换次数"])
        self.swap_ax.set_title("排序算法交换次数")
        self.swap_ax.set_xlabel("算法")
        self.swap_ax.set_ylabel("交换次数")
        plt.setp(self.swap_ax.get_xticklabels(), rotation=45, ha="right")
        self.swap_fig.tight_layout()
        self.swap_canvas.draw()
    
    def _export_results(self):
        """导出结果到CSV文件"""
        if not self.results or not self.results["算法"]:
            messagebox.showwarning("警告", "没有可导出的结果")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(self.results)
        
        # 获取保存路径
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")],
            title="保存结果"
        )
        
        if file_path:
            # 保存到CSV
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            messagebox.showinfo("成功", f"结果已保存到 {file_path}")

# 主函数
if __name__ == "__main__":
    root = tk.Tk()
    app = SortingComparison(root)
    root.mainloop() 