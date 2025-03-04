# 排序算法可视化与分析工具集

这是一个全面的排序算法可视化与分析工具集，旨在帮助用户理解、学习和比较各种排序算法的工作原理和性能特点。

## 功能特点

本工具集包含以下主要组件：

### 1. 排序算法可视化工具 (sorting_visualizer.py)

- 动态可视化冒泡排序、选择排序和插入排序的执行过程
- 支持调整数组大小、最大值和动画速度
- 提供中文界面和注释
- 可选择保存动画为GIF文件

### 2. 排序算法基准测试工具 (sorting_benchmark.py)

- 测试多种排序算法在不同数据集上的性能表现
- 支持随机、已排序、逆序和部分排序的数据集
- 生成性能比较图表（条形图和折线图）
- 可导出测试结果为CSV文件

### 3. 排序算法教学工具 (sorting_tutorial.py)

- 提供交互式学习体验，帮助理解各种排序算法的工作原理
- 包含详细的算法说明和步骤解析
- 可视化展示算法执行过程
- 支持调整数组大小和动画速度

### 4. 排序算法比较工具 (sort_algorithms_comparison.py)

- 直观比较不同排序算法在相同数据集上的性能差异
- 测量执行时间、比较次数和交换次数
- 生成多种图表展示比较结果
- 支持导出比较结果

### 5. 排序算法可视化比较工具 (sorting_visualizer_comparison.py)

- 同时可视化多种排序算法的执行过程
- 方便直观比较不同算法的工作方式和效率
- 支持自定义比较的算法组合

### 6. 排序算法模块 (sort_algorithms.py)

- 包含多种排序算法的实现：
  - 冒泡排序 (Bubble Sort)
  - 选择排序 (Selection Sort)
  - 插入排序 (Insertion Sort)
  - 希尔排序 (Shell Sort)
  - 归并排序 (Merge Sort)
  - 快速排序 (Quick Sort)
  - 堆排序 (Heap Sort)
  - 计数排序 (Counting Sort)
  - 基数排序 (Radix Sort)
  - 桶排序 (Bucket Sort)
  - Python内置排序 (Timsort)

## 安装与使用

### 环境要求

- Python 3.6+
- 依赖库：matplotlib, numpy, pandas, tkinter

### 安装步骤

1. 克隆或下载本仓库
2. 安装依赖库：

```bash
pip install -r requirements.txt
```

### 使用方法

#### 排序算法可视化工具

```bash
python sorting_visualizer.py
```

#### 排序算法基准测试工具

```bash
python sorting_benchmark.py
```

#### 排序算法教学工具

```bash
python sorting_tutorial.py
```

#### 排序算法比较工具

```bash
python sort_algorithms_comparison.py
```

#### 排序算法可视化比较工具

```bash
python sorting_visualizer_comparison.py
```

## 示例

### 排序可视化示例

![排序可视化示例](https://example.com/sorting_visualization.gif)

### 性能比较示例

![性能比较示例](https://example.com/performance_comparison.png)

## 项目结构

```
.
├── README.md                         # 项目说明文档
├── requirements.txt                  # 项目依赖库
├── sorting_visualizer.py             # 排序算法可视化工具
├── sorting_benchmark.py              # 排序算法基准测试工具
├── sorting_tutorial.py               # 排序算法教学工具
├── sort_algorithms_comparison.py     # 排序算法比较工具
├── sorting_visualizer_comparison.py  # 排序算法可视化比较工具
└── sort_algorithms.py                # 排序算法模块
```

## 贡献指南

欢迎对本项目进行贡献！您可以通过以下方式参与：

1. 提交Bug报告或功能请求
2. 提交Pull Request改进代码
3. 完善文档或添加示例

## 许可证

本项目采用MIT许可证。详情请参阅LICENSE文件。

## 致谢

感谢所有为本项目做出贡献的开发者和用户。