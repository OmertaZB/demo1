# 排序算法性能比较

这个项目实现了多种排序算法，并比较它们在随机数据上的性能表现。每种算法都有详细的注释说明其原理、时间复杂度和空间复杂度。

## 实现的排序算法

- **冒泡排序 (Bubble Sort)**: 通过重复遍历要排序的数列，一次比较两个元素，如果它们的顺序错误就交换它们，直到没有再需要交换的元素为止。
  - 时间复杂度：O(n²)，最好情况O(n)
  - 空间复杂度：O(1)
  - 稳定性：稳定

- **选择排序 (Selection Sort)**: 每次从未排序部分找出最小元素，放到已排序部分的末尾。
  - 时间复杂度：O(n²)
  - 空间复杂度：O(1)
  - 稳定性：不稳定

- **插入排序 (Insertion Sort)**: 构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。
  - 时间复杂度：O(n²)，最好情况O(n)
  - 空间复杂度：O(1)
  - 稳定性：稳定

- **希尔排序 (Shell Sort)**: 插入排序的改进版，先将整个待排序列分割成若干子序列分别进行直接插入排序。
  - 时间复杂度：O(n log² n)
  - 空间复杂度：O(1)
  - 稳定性：不稳定

- **归并排序 (Merge Sort)**: 分治法的典型应用，将已有序的子序列合并，得到完全有序的序列。
  - 时间复杂度：O(n log n)
  - 空间复杂度：O(n)
  - 稳定性：稳定

- **快速排序 (Quick Sort)**: 选择一个基准元素，将所有小于基准的元素放在前面，大于基准的放在后面，然后递归地对子序列进行排序。
  - 时间复杂度：O(n log n)，最坏情况O(n²)
  - 空间复杂度：O(log n)
  - 稳定性：不稳定

- **堆排序 (Heap Sort)**: 利用堆这种数据结构所设计的一种排序算法，将待排序序列构造成大顶堆，然后交换堆顶元素和末尾元素。
  - 时间复杂度：O(n log n)
  - 空间复杂度：O(1)
  - 稳定性：不稳定

- **计数排序 (Counting Sort)**: 非比较排序，通过统计数组中每个值的出现次数，然后按顺序把数据填充回去。
  - 时间复杂度：O(n + k)，其中k是整数的范围
  - 空间复杂度：O(n + k)
  - 稳定性：稳定

- **Python内置排序 (Python's Built-in Sort)**: 使用Timsort算法，结合了归并排序和插入排序的优点。
  - 时间复杂度：O(n log n)
  - 空间复杂度：O(n)
  - 稳定性：稳定

## 环境要求

- Python 3.6+
- matplotlib
- numpy

## 安装依赖

```bash
pip install matplotlib numpy
```

## 运行程序

```bash
python sort_algorithms.py
```

## 输出结果

程序会输出每种排序算法的执行时间，并生成一个名为`sort_algorithms_comparison.png`的柱状图，直观展示各算法的性能差异。

## 自定义设置

如果你想修改测试数据的大小或范围，可以在`sort_algorithms.py`文件中的`main()`函数中调整以下参数：

- `array_size`：测试数组的大小（默认为5000）
- `generate_random_array`函数的`min_val`和`max_val`参数：随机数的范围（默认为0-1000）