import time
import random
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Callable

"""
排序算法比较程序
实现了多种常见排序算法，并比较它们在随机数据上的性能表现
"""

# 冒泡排序
def bubble_sort(arr: List[int]) -> List[int]:
    """
    冒泡排序原理：
    1. 比较相邻的元素，如果第一个比第二个大，就交换它们
    2. 对每一对相邻元素做同样的工作，从开始第一对到结尾的最后一对
    3. 每轮结束后，最大的元素会"冒泡"到最后的位置
    4. 重复以上步骤，每次比较到最后一个未排序的元素
    
    时间复杂度：
    - 最好情况：O(n) - 当数组已经排序好时
    - 最坏情况：O(n²) - 当数组逆序排列时
    - 平均情况：O(n²)
    
    空间复杂度：O(1) - 只需要一个临时变量用于交换
    
    稳定性：稳定 - 相等元素的相对位置不会改变
    """
    n = len(arr)
    arr_copy = arr.copy()
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr_copy[j] > arr_copy[j + 1]:
                arr_copy[j], arr_copy[j + 1] = arr_copy[j + 1], arr_copy[j]
    return arr_copy

# 选择排序
def selection_sort(arr: List[int]) -> List[int]:
    """
    选择排序原理：
    1. 在未排序序列中找到最小（大）元素，存放到排序序列的起始位置
    2. 从剩余未排序元素中继续寻找最小（大）元素，放到已排序序列的末尾
    3. 重复第二步，直到所有元素均排序完毕
    
    时间复杂度：
    - 最好情况：O(n²) - 即使数组已经排序好，仍需要进行n轮比较
    - 最坏情况：O(n²) - 当数组逆序排列时
    - 平均情况：O(n²)
    
    空间复杂度：O(1) - 只需要一个临时变量用于交换
    
    稳定性：不稳定 - 相等元素的相对位置可能会改变
    """
    n = len(arr)
    arr_copy = arr.copy()
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr_copy[j] < arr_copy[min_idx]:
                min_idx = j
        arr_copy[i], arr_copy[min_idx] = arr_copy[min_idx], arr_copy[i]
    return arr_copy

# 插入排序
def insertion_sort(arr: List[int]) -> List[int]:
    """
    插入排序原理：
    1. 将数组分为已排序和未排序两部分，初始时已排序部分只有第一个元素
    2. 从未排序部分取出第一个元素，在已排序部分从后向前扫描
    3. 如果已排序部分的元素大于新元素，则将该元素移到下一位置
    4. 重复步骤3，直到找到已排序部分的元素小于或等于新元素的位置
    5. 将新元素插入到该位置
    6. 重复步骤2~5，直到未排序部分为空
    
    时间复杂度：
    - 最好情况：O(n) - 当数组已经排序好时
    - 最坏情况：O(n²) - 当数组逆序排列时
    - 平均情况：O(n²)
    
    空间复杂度：O(1) - 只需要一个临时变量用于插入
    
    稳定性：稳定 - 相等元素的相对位置不会改变
    """
    n = len(arr)
    arr_copy = arr.copy()
    for i in range(1, n):
        key = arr_copy[i]
        j = i - 1
        while j >= 0 and arr_copy[j] > key:
            arr_copy[j + 1] = arr_copy[j]
            j -= 1
        arr_copy[j + 1] = key
    return arr_copy

# 希尔排序
def shell_sort(arr: List[int]) -> List[int]:
    """
    希尔排序原理：
    1. 希尔排序是插入排序的一种改进版本，也称为"缩小增量排序"
    2. 先将整个待排序列分割成若干个子序列分别进行直接插入排序
    3. 排序时按下标的一定增量进行分组，对每组使用直接插入排序
    4. 随着增量逐渐减少，每组包含的元素越来越多，当增量减至1时，整个数组恰被分成一组，排序完成
    
    时间复杂度：
    - 最好情况：O(n log n) - 取决于间隔序列
    - 最坏情况：O(n²) - 但通常优于简单插入排序
    - 平均情况：O(n log² n) - 取决于间隔序列的选择
    
    空间复杂度：O(1) - 只需要一个临时变量用于插入
    
    稳定性：不稳定 - 相等元素的相对位置可能会改变
    """
    n = len(arr)
    arr_copy = arr.copy()
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr_copy[i]
            j = i
            while j >= gap and arr_copy[j - gap] > temp:
                arr_copy[j] = arr_copy[j - gap]
                j -= gap
            arr_copy[j] = temp
        gap //= 2
    return arr_copy

# 归并排序
def merge_sort(arr: List[int]) -> List[int]:
    """
    归并排序原理：
    1. 采用分治法（Divide and Conquer）的一个典型应用
    2. 将已有序的子序列合并，得到完全有序的序列
    3. 先使每个子序列有序，再使子序列段间有序
    4. 若将两个有序表合并成一个有序表，称为二路归并
    
    具体步骤：
    1. 把长度为n的输入序列分成两个长度为n/2的子序列
    2. 对这两个子序列分别采用归并排序
    3. 将两个排序好的子序列合并成一个最终的排序序列
    
    时间复杂度：
    - 最好情况：O(n log n) - 始终是这个复杂度
    - 最坏情况：O(n log n) - 始终是这个复杂度
    - 平均情况：O(n log n)
    
    空间复杂度：O(n) - 需要额外的空间来存储合并过程中的临时数组
    
    稳定性：稳定 - 相等元素的相对位置不会改变
    """
    arr_copy = arr.copy()
    if len(arr_copy) > 1:
        mid = len(arr_copy) // 2
        left_half = arr_copy[:mid]
        right_half = arr_copy[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr_copy[k] = left_half[i]
                i += 1
            else:
                arr_copy[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr_copy[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr_copy[k] = right_half[j]
            j += 1
            k += 1
    return arr_copy

# 快速排序
def quick_sort(arr: List[int]) -> List[int]:
    """
    快速排序原理：
    1. 也是采用分治法的一个典型应用
    2. 选择一个基准元素（pivot），通常选择第一个或最后一个元素
    3. 将所有比基准值小的元素放在基准前面，比基准值大的元素放在基准后面
    4. 对划分好的两个子序列分别进行快速排序
    
    时间复杂度：
    - 最好情况：O(n log n) - 每次划分恰好在中间
    - 最坏情况：O(n²) - 当数组已经排序好或逆序排列时（可通过随机选择基准值来避免）
    - 平均情况：O(n log n)
    
    空间复杂度：O(log n) - 递归调用的栈空间
    
    稳定性：不稳定 - 相等元素的相对位置可能会改变
    """
    arr_copy = arr.copy()
    
    def _quick_sort(arr, low, high):
        if low < high:
            pivot_index = partition(arr, low, high)
            _quick_sort(arr, low, pivot_index - 1)
            _quick_sort(arr, pivot_index + 1, high)
    
    def partition(arr, low, high):
        # 选择最右边的元素作为基准值
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    if len(arr_copy) > 1:
        _quick_sort(arr_copy, 0, len(arr_copy) - 1)
    return arr_copy

# 堆排序
def heap_sort(arr: List[int]) -> List[int]:
    """
    堆排序原理：
    1. 利用堆这种数据结构所设计的一种排序算法
    2. 堆是一个近似完全二叉树的结构，并同时满足堆的性质：子节点的键值或索引总是小于（或者大于）它的父节点
    
    具体步骤：
    1. 将初始待排序序列构建成大顶堆（所有父节点的值大于子节点）
    2. 将堆顶元素与末尾元素交换，将最大元素"沉"到数组末端
    3. 重新调整结构，使其满足堆定义，然后继续交换堆顶元素与当前末尾元素
    4. 反复执行步骤2和3，直到整个序列有序
    
    时间复杂度：
    - 最好情况：O(n log n) - 始终是这个复杂度
    - 最坏情况：O(n log n) - 始终是这个复杂度
    - 平均情况：O(n log n)
    
    空间复杂度：O(1) - 只需要一个临时变量用于交换
    
    稳定性：不稳定 - 相等元素的相对位置可能会改变
    """
    arr_copy = arr.copy()
    n = len(arr_copy)
    
    def heapify(arr, n, i):
        """
        调整堆的函数，用于维护堆的性质
        i: 当前需要调整的节点索引
        n: 堆的大小
        """
        largest = i  # 初始化最大值为根节点
        left = 2 * i + 1  # 左子节点
        right = 2 * i + 2  # 右子节点
        
        # 如果左子节点存在且大于根节点
        if left < n and arr[largest] < arr[left]:
            largest = left
        
        # 如果右子节点存在且大于当前最大值
        if right < n and arr[largest] < arr[right]:
            largest = right
        
        # 如果最大值不是根节点，则交换并继续调整
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)
    
    # 构建初始大顶堆
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr_copy, n, i)
    
    # 一个个交换元素并调整堆
    for i in range(n - 1, 0, -1):
        arr_copy[i], arr_copy[0] = arr_copy[0], arr_copy[i]  # 交换
        heapify(arr_copy, i, 0)  # 调整剩余堆
    
    return arr_copy

# 计数排序
def counting_sort(arr: List[int]) -> List[int]:
    """
    计数排序原理：
    1. 计数排序是一种非比较排序算法，适用于一定范围内的整数排序
    2. 通过统计数组中每个值的出现次数，然后按顺序把数据填充回去
    
    具体步骤：
    1. 找出待排序数组中最大和最小的元素
    2. 统计数组中每个值为i的元素出现的次数，存入计数数组count的第i项
    3. 对所有的计数累加，从count[i-1]到count[i]
    4. 反向填充目标数组：将每个元素i放在新数组的第count[i]项，每放一个元素就将count[i]减去1
    
    时间复杂度：
    - O(n + k)，其中k是整数的范围
    
    空间复杂度：
    - O(n + k)，需要额外的计数数组和输出数组
    
    稳定性：稳定 - 相等元素的相对位置不会改变
    
    适用场景：
    - 适合于数据范围不大的情况
    - 当数据范围很大时，所需的内存空间会很大，可能导致内存溢出
    """
    arr_copy = arr.copy()
    max_val = max(arr_copy)
    min_val = min(arr_copy)
    range_val = max_val - min_val + 1
    
    count = [0] * range_val
    output = [0] * len(arr_copy)
    
    # 统计每个元素出现的次数
    for i in arr_copy:
        count[i - min_val] += 1
    
    # 累加计数数组
    for i in range(1, len(count)):
        count[i] += count[i - 1]
    
    # 构建输出数组
    for i in range(len(arr_copy) - 1, -1, -1):
        output[count[arr_copy[i] - min_val] - 1] = arr_copy[i]
        count[arr_copy[i] - min_val] -= 1
    
    return output

# Python内置排序
def python_sort(arr: List[int]) -> List[int]:
    """
    Python内置排序原理：
    1. Python的内置排序函数sort()使用的是Timsort算法
    2. Timsort是一种混合排序算法，结合了归并排序和插入排序的优点
    3. 对于小数组，它使用插入排序；对于大数组，它使用归并排序
    
    时间复杂度：
    - 最好情况：O(n) - 当数组已经排序好时
    - 最坏情况：O(n log n) - 始终是这个复杂度
    - 平均情况：O(n log n)
    
    空间复杂度：O(n) - 需要额外的空间来存储临时数组
    
    稳定性：稳定 - 相等元素的相对位置不会改变
    """
    arr_copy = arr.copy()
    arr_copy.sort()
    return arr_copy

# 测量排序算法的执行时间
def measure_time(sort_func: Callable, arr: List[int]) -> float:
    """
    测量排序算法的执行时间
    
    参数:
    sort_func: 排序函数
    arr: 待排序数组
    
    返回:
    执行时间(秒)
    """
    start_time = time.time()
    sort_func(arr)
    end_time = time.time()
    return end_time - start_time

# 生成随机数组
def generate_random_array(size: int, min_val: int = 0, max_val: int = 1000) -> List[int]:
    """
    生成指定大小的随机整数数组
    
    参数:
    size: 数组大小
    min_val: 最小值
    max_val: 最大值
    
    返回:
    随机整数数组
    """
    return [random.randint(min_val, max_val) for _ in range(size)]

# 主函数
def main():
    """
    主函数：测试各种排序算法的性能并绘制比较图
    """
    # 设置随机种子以确保结果可重现
    random.seed(42)
    
    # 定义数组大小
    array_size = 5000
    
    # 生成随机数组
    random_array = generate_random_array(array_size)
    
    # 定义要测试的排序算法
    sort_algorithms = {
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
    
    # 测量每个算法的执行时间
    execution_times = {}
    for name, algorithm in sort_algorithms.items():
        print(f"测试 {name}...")
        execution_time = measure_time(algorithm, random_array)
        execution_times[name] = execution_time
        print(f"{name} 耗时: {execution_time:.6f} 秒")
    
    # 绘制柱状图
    plt.figure(figsize=(12, 8))
    algorithms = list(execution_times.keys())
    times = list(execution_times.values())
    
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    bars = plt.bar(algorithms, times, color='skyblue')
    plt.xlabel('排序算法')
    plt.ylabel('执行时间 (秒)')
    plt.title(f'各种排序算法在{array_size}个随机元素上的执行时间比较')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 在柱状图上添加具体数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.6f}',
                 ha='center', va='bottom', rotation=0)
    
    # 保存图表
    plt.savefig('sort_algorithms_comparison.png')
    plt.show()

if __name__ == "__main__":
    main()