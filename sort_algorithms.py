import time
import random
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Callable

"""
排序算法模块
包含各种排序算法的实现，供其他工具调用
"""

def bubble_sort(arr):
    """
    冒泡排序
    
    时间复杂度: O(n²)
    空间复杂度: O(1)
    稳定性: 稳定
    
    参数:
        arr: 待排序的数组
    
    返回:
        排序后的数组
    """
    arr = arr.copy()
    n = len(arr)
    
    for i in range(n):
        # 优化：如果一轮中没有交换，说明已经排序完成
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        if not swapped:
            break
    
    return arr

def selection_sort(arr):
    """
    选择排序
    
    时间复杂度: O(n²)
    空间复杂度: O(1)
    稳定性: 不稳定
    
    参数:
        arr: 待排序的数组
    
    返回:
        排序后的数组
    """
    arr = arr.copy()
    n = len(arr)
    
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
    
    return arr

def insertion_sort(arr):
    """
    插入排序
    
    时间复杂度: O(n²)
    空间复杂度: O(1)
    稳定性: 稳定
    
    参数:
        arr: 待排序的数组
    
    返回:
        排序后的数组
    """
    arr = arr.copy()
    n = len(arr)
    
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        
        arr[j + 1] = key
    
    return arr

def shell_sort(arr):
    """
    希尔排序
    
    时间复杂度: 取决于间隔序列，平均为O(n^1.3)
    空间复杂度: O(1)
    稳定性: 不稳定
    
    参数:
        arr: 待排序的数组
    
    返回:
        排序后的数组
    """
    arr = arr.copy()
    n = len(arr)
    gap = n // 2
    
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            
            arr[j] = temp
        
        gap //= 2
    
    return arr

def merge_sort(arr):
    """
    归并排序
    
    时间复杂度: O(n log n)
    空间复杂度: O(n)
    稳定性: 稳定
    
    参数:
        arr: 待排序的数组
    
    返回:
        排序后的数组
    """
    arr = arr.copy()
    
    if len(arr) <= 1:
        return arr
    
    def merge(left, right):
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    # 分割数组
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # 合并结果
    return merge(left, right)

def quick_sort(arr):
    """
    快速排序
    
    时间复杂度: 平均O(n log n)，最坏O(n²)
    空间复杂度: O(log n)
    稳定性: 不稳定
    
    参数:
        arr: 待排序的数组
    
    返回:
        排序后的数组
    """
    arr = arr.copy()
    
    def _quick_sort(arr, low, high):
        if low < high:
            # 分区操作，返回基准元素的索引
            pivot_idx = _partition(arr, low, high)
            
            # 递归排序基准元素左右两侧的子数组
            _quick_sort(arr, low, pivot_idx - 1)
            _quick_sort(arr, pivot_idx + 1, high)
    
    def _partition(arr, low, high):
        # 选择最右边的元素作为基准
        pivot = arr[high]
        i = low - 1
        
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    if len(arr) > 1:
        _quick_sort(arr, 0, len(arr) - 1)
    
    return arr

def heap_sort(arr):
    """
    堆排序
    
    时间复杂度: O(n log n)
    空间复杂度: O(1)
    稳定性: 不稳定
    
    参数:
        arr: 待排序的数组
    
    返回:
        排序后的数组
    """
    arr = arr.copy()
    n = len(arr)
    
    # 构建最大堆
    for i in range(n // 2 - 1, -1, -1):
        _heapify(arr, n, i)
    
    # 一个个从堆中取出元素
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # 交换
        _heapify(arr, i, 0)
    
    return arr

def _heapify(arr, n, i):
    """堆排序的辅助函数，用于维护堆的性质"""
    largest = i  # 初始化最大值为根节点
    left = 2 * i + 1
    right = 2 * i + 2
    
    # 如果左子节点存在且大于根节点
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    # 如果右子节点存在且大于当前最大值
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    # 如果最大值不是根节点
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # 交换
        _heapify(arr, n, largest)  # 递归调整堆

def counting_sort(arr, max_val=None):
    """
    计数排序
    
    时间复杂度: O(n + k)，其中k是数值范围
    空间复杂度: O(n + k)
    稳定性: 稳定
    
    参数:
        arr: 待排序的数组
        max_val: 数组中的最大值，如果不提供则自动计算
    
    返回:
        排序后的数组
    """
    arr = arr.copy()
    
    if not arr:
        return arr
    
    # 如果没有提供最大值，则计算
    if max_val is None:
        max_val = max(arr)
    
    # 创建计数数组
    count = [0] * (max_val + 1)
    
    # 统计每个元素出现的次数
    for num in arr:
        count[num] += 1
    
    # 重建排序后的数组
    sorted_arr = []
    for i in range(len(count)):
        sorted_arr.extend([i] * count[i])
    
    return sorted_arr

def radix_sort(arr):
    """
    基数排序
    
    时间复杂度: O(d * (n + k))，其中d是最大数字的位数，k是基数
    空间复杂度: O(n + k)
    稳定性: 稳定
    
    参数:
        arr: 待排序的非负整数数组
    
    返回:
        排序后的数组
    """
    arr = arr.copy()
    
    if not arr:
        return arr
    
    # 找出最大值，确定位数
    max_val = max(arr)
    exp = 1
    
    # 按位进行计数排序
    while max_val // exp > 0:
        _counting_sort_by_digit(arr, exp)
        exp *= 10
    
    return arr

def _counting_sort_by_digit(arr, exp):
    """基数排序的辅助函数，按指定位数进行计数排序"""
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    
    # 统计当前位上每个数字出现的次数
    for i in range(n):
        index = (arr[i] // exp) % 10
        count[index] += 1
    
    # 更新count，使count[i]表示小于等于i的元素个数
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    # 构建输出数组
    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1
    
    # 将输出数组复制回原数组
    for i in range(n):
        arr[i] = output[i]

def bucket_sort(arr, bucket_size=5):
    """
    桶排序
    
    时间复杂度: 平均O(n + k)，最坏O(n²)，其中k是桶的数量
    空间复杂度: O(n + k)
    稳定性: 取决于桶内排序算法，通常使用插入排序，因此是稳定的
    
    参数:
        arr: 待排序的数组
        bucket_size: 每个桶的大小
    
    返回:
        排序后的数组
    """
    arr = arr.copy()
    
    if not arr:
        return arr
    
    # 确定最大值和最小值
    min_val, max_val = min(arr), max(arr)
    
    # 计算桶的数量
    bucket_count = (max_val - min_val) // bucket_size + 1
    buckets = [[] for _ in range(bucket_count)]
    
    # 将元素分配到桶中
    for num in arr:
        index = (num - min_val) // bucket_size
        buckets[index].append(num)
    
    # 对每个桶进行排序，这里使用插入排序
    for i in range(bucket_count):
        buckets[i] = insertion_sort(buckets[i])
    
    # 合并桶
    result = []
    for bucket in buckets:
        result.extend(bucket)
    
    return result

def python_sort(arr):
    """
    Python内置排序（Timsort）
    
    时间复杂度: O(n log n)
    空间复杂度: O(n)
    稳定性: 稳定
    
    参数:
        arr: 待排序的数组
    
    返回:
        排序后的数组
    """
    arr = arr.copy()
    arr.sort()
    return arr

# 测试代码
if __name__ == "__main__":
    import random
    import time
    
    # 生成随机数组
    test_arr = [random.randint(0, 1000) for _ in range(1000)]
    
    # 测试所有排序算法
    algorithms = [
        ("冒泡排序", bubble_sort),
        ("选择排序", selection_sort),
        ("插入排序", insertion_sort),
        ("希尔排序", shell_sort),
        ("归并排序", merge_sort),
        ("快速排序", quick_sort),
        ("堆排序", heap_sort),
        ("计数排序", counting_sort),
        ("基数排序", radix_sort),
        ("桶排序", bucket_sort),
        ("Python内置排序", python_sort)
    ]
    
    for name, func in algorithms:
        # 复制数组以避免修改原数组
        arr_copy = test_arr.copy()
        
        # 计时
        start_time = time.time()
        sorted_arr = func(arr_copy)
        end_time = time.time()
        
        # 验证排序结果
        is_sorted = all(sorted_arr[i] <= sorted_arr[i+1] for i in range(len(sorted_arr)-1))
        
        print(f"{name}: {'成功' if is_sorted else '失败'}, 耗时: {end_time - start_time:.6f}秒")