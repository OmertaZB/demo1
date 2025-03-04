import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sort_algorithms import (
    bubble_sort, 
    selection_sort, 
    insertion_sort, 
    shell_sort, 
    merge_sort, 
    quick_sort, 
    heap_sort, 
    counting_sort, 
    python_sort
)
import matplotlib

"""
排序算法基准测试工具
用于测试不同排序算法在各种数据集上的性能表现
"""

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class SortingBenchmark:
    def __init__(self):
        """初始化基准测试工具"""
        # 定义要测试的排序算法
        self.algorithms = {
            "冒泡排序": bubble_sort,
            "选择排序": selection_sort,
            "插入排序": insertion_sort,
            "希尔排序": shell_sort,
            "归并排序": merge_sort,
            "快速排序": quick_sort,
            "堆排序": heap_sort,
            "计数排序": counting_sort,
            "Python内置排序": python_sort
        }
        
        # 存储测试结果
        self.results = {}
    
    def generate_dataset(self, size, data_type='random', range_min=0, range_max=1000):
        """
        生成测试数据集
        
        参数:
        size: 数据集大小
        data_type: 数据类型，可选值: 'random'(随机), 'sorted'(已排序), 'reversed'(逆序), 'nearly_sorted'(接近排序)
        range_min: 最小值
        range_max: 最大值
        
        返回:
        生成的数据集
        """
        if data_type == 'random':
            return [random.randint(range_min, range_max) for _ in range(size)]
        elif data_type == 'sorted':
            return list(range(range_min, range_min + size))
        elif data_type == 'reversed':
            return list(range(range_min + size - 1, range_min - 1, -1))
        elif data_type == 'nearly_sorted':
            arr = list(range(range_min, range_min + size))
            # 随机交换5%的元素
            swaps = int(size * 0.05)
            for _ in range(swaps):
                i, j = random.sample(range(size), 2)
                arr[i], arr[j] = arr[j], arr[i]
            return arr
        else:
            raise ValueError("不支持的数据类型")
    
    def run_benchmark(self, sizes, data_types=['random'], trials=3):
        """
        运行基准测试
        
        参数:
        sizes: 数据集大小列表
        data_types: 数据类型列表
        trials: 每个测试重复次数，取平均值
        """
        self.results = {}
        
        for data_type in data_types:
            self.results[data_type] = {}
            for size in sizes:
                print(f"测试数据类型: {data_type}, 大小: {size}")
                self.results[data_type][size] = {}
                
                for algo_name, algo_func in self.algorithms.items():
                    total_time = 0
                    
                    for _ in range(trials):
                        # 生成数据集
                        data = self.generate_dataset(size, data_type)
                        
                        # 测量排序时间
                        start_time = time.time()
                        algo_func(data)
                        end_time = time.time()
                        
                        total_time += (end_time - start_time)
                    
                    # 计算平均时间
                    avg_time = total_time / trials
                    self.results[data_type][size][algo_name] = avg_time
                    print(f"  {algo_name}: {avg_time:.6f} 秒")
    
    def plot_results(self, plot_type='bar', save_path=None):
        """
        绘制测试结果
        
        参数:
        plot_type: 图表类型，可选值: 'bar'(柱状图), 'line'(折线图)
        save_path: 保存路径，如果为None则不保存
        """
        if not self.results:
            print("没有测试结果可绘制")
            return
        
        data_types = list(self.results.keys())
        sizes = list(self.results[data_types[0]].keys())
        algorithms = list(self.algorithms.keys())
        
        if plot_type == 'bar':
            # 为每种数据类型创建一个图表
            for data_type in data_types:
                plt.figure(figsize=(14, 8))
                
                # 为每个数据集大小创建一个子图
                for i, size in enumerate(sizes):
                    plt.subplot(1, len(sizes), i+1)
                    
                    # 提取该大小下所有算法的执行时间
                    times = [self.results[data_type][size][algo] for algo in algorithms]
                    
                    # 绘制柱状图
                    bars = plt.bar(algorithms, times, color='skyblue')
                    plt.title(f'数据大小: {size}')
                    plt.xticks(rotation=45, ha='right')
                    plt.ylabel('执行时间 (秒)')
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # 在柱状图上添加具体数值
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.6f}',
                                ha='center', va='bottom', rotation=0, fontsize=8)
                
                plt.suptitle(f'排序算法在{data_type}数据上的性能比较', fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                
                if save_path:
                    plt.savefig(f"{save_path}_{data_type}_bar.png")
                plt.show()
        
        elif plot_type == 'line':
            # 为每种数据类型创建一个图表
            for data_type in data_types:
                plt.figure(figsize=(12, 8))
                
                # 为每个算法绘制一条线
                for algo in algorithms:
                    times = [self.results[data_type][size][algo] for size in sizes]
                    plt.plot(sizes, times, marker='o', label=algo)
                
                plt.title(f'排序算法在{data_type}数据上的性能比较', fontsize=16)
                plt.xlabel('数据集大小')
                plt.ylabel('执行时间 (秒)')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                plt.xscale('log')  # 使用对数刻度更好地显示不同大小
                plt.yscale('log')  # 使用对数刻度更好地显示不同时间范围
                
                if save_path:
                    plt.savefig(f"{save_path}_{data_type}_line.png")
                plt.show()
    
    def export_results(self, file_path):
        """
        将测试结果导出到CSV文件
        
        参数:
        file_path: 文件保存路径
        """
        if not self.results:
            print("没有测试结果可导出")
            return
        
        # 准备数据
        rows = []
        for data_type in self.results:
            for size in self.results[data_type]:
                row = {'数据类型': data_type, '数据大小': size}
                row.update(self.results[data_type][size])
                rows.append(row)
        
        # 创建DataFrame并导出
        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"结果已导出到 {file_path}")

# 示例用法
if __name__ == "__main__":
    # 创建基准测试实例
    benchmark = SortingBenchmark()
    
    # 运行基准测试
    # 测试不同大小的随机数据
    benchmark.run_benchmark(
        sizes=[1000, 5000, 10000],
        data_types=['random', 'sorted', 'reversed', 'nearly_sorted'],
        trials=3
    )
    
    # 绘制结果
    benchmark.plot_results(plot_type='bar', save_path='sorting_benchmark_results')
    benchmark.plot_results(plot_type='line', save_path='sorting_benchmark_results')
    
    # 导出结果
    benchmark.export_results('sorting_benchmark_results.csv') 