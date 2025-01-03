# ML-Grokking

## 文件使用说明

作业中主要依赖的函数文件均在 utilities 文件夹中。各文件主要的用途陈述如下：

### _complement.py

该文件主要实现了计算某一集合 $\mathcal{T}$ 的加性补集 $\mathcal{C_T}$ 的方法。关于 $\mathcal{C_T}$ 的定义参见报告附录部分。

### _helper.py

该文件主要实现一些计算数值结果的一些辅助方法，包括 DCT 以及滑窗平均。

### _network.py

该文件包含所有的网络架构实现。

### _training.py

该文件包含用于训练模型的方法。所有参数在文件中给出注释。

### _update.py

该文件包含单步的训练与测试，以及计算每步的网络各项指标的正确率的方法。

## 结果复现

$\texttt{example.ipynb}$ 给出了测试样例，作为复现论文结果的参考。关于各参数如何具体使用可以参考 \_ $\texttt{training.py}$ 文件中的注释。

## 库

所有代码都是基于 $\texttt{Python}$ 3.12.4 实现。使用的库包括：$\texttt{torch}$ 2.5.1, $\texttt{numpy}$ 1.26.4
