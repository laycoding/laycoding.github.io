---
title: BackPropagation学习笔记
layout: post
---
# BackPropagation学习笔记
#### 这是学习[Udacity深度学习课程](https://cn.udacity.com/course/deep-learning-nanodegree-foundation--nd101/)时记录的笔记

### 反向传播的历史：

反向传播算法最早被提出是在上世纪70年代，但是重要性被普遍认可是在1986年David Rumelhart, Geoffrey Hinton和 Ronald Williams发表在nature上的一篇论文之后： [Learning representations by back-propagating errors](https://www.nature.com/nature/journal/v323/n6088/pdf/323533a0.pdf)<br />
论文提到尽管反向传播算法的工作方法似是而非我们大脑的学习方法，但是提供了复杂网络使用梯度下降的计算的可能性，因为这比以前的方法(ig:perceptron-convergence procedure)快了许多。

### 这篇笔记中的符号标记：
>* $W_{j}$表示$j-1^{th}$到$j^{th}$层的传播权重；
>* $z^{j}$表示到第$j^{th}$层的输入(weighted input of $j^{th}$layer)；
>* $a^{j}$表示第$j^{th}$层的激活函数激活后的输出(activated output of $j^{th}$layer)；
>* $C$表示总误差，但它和一般意义下的SSE又有所不同，为了计算方便，它定义为：$$C = \frac{1}{2} \sum_{j} (y_{j} - a^{L}_{j}) ^{2}$$
>* $\delta^{j}$是一个虚拟变量，定义为：$$\delta^{j} = \partial C / \partial z^{j}$$后边会具体解释,感兴趣的同学可以看MichaelNielsen的[解释](http://neuralnetworksanddeeplearning.com/chap2.html)；
>* $\odot$ 表示[hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices), 即两个相同维度的矩阵hadamard积等于俩矩阵相同位置的乘积，熟悉matlab的同学可以直接理解为点乘。
>* 假使所有输入输出都是$x$ by 1的<font face="黑体" size= 2 color=red>列向量</font>（惯例）

###### 小Tips:
搞清楚每个变量的维数是保证代码运行正确的关键<br \>
$z^{j}$，$a^{j}$永远都是一维的

### 我所理解的反向传播：
>* 通过正向传播，可以轻松的算出最后一层的误差(当然前提是知道最后一层的label)
>* 链式法则让我们把$C$对前后网络层的权重$W$的导数连接起来，这样就可以形成递推，反向推导出前一层的$\partial C / \partial W^{j}$


下面我们具体看一下反向传播是如何实现的吧！

#### 我们先构建一个简单的神经网络，并使用正向传播导出总误差：

![simple_neural_network](http://wx1.sinaimg.cn/mw690/0066a0DZly1fgewdmo5kvj30fp06pq3c.jpg)

利用：<br \>
    $z^{j}=X * W_{j}$<br \>
    $a^{j}=ActivateFunction(z^{j})=\sigma (z^{j})$<br \>
导出：<br \>
    $C = \frac{1}{2} \sum_{j} (y_{j} - a^{3}_{j}) ^{2}$

### 让我们简单复习一下[链式法则](https://www.baidu.com/link?url=6PIBLB3vJso34EIAu4WR3-kbnJhHSLOOUW1kwxsj59yT7Qkw73uJdASvObQj2XkxWXrl7O7EKuA2-_frbABj1a&wd=&eqid=fe630e650002776a00000003593aaaf3):
这里我们考虑单变量复合函数求导法则：若$y=f(u)$在点$u$可导，$u=g(x)$在点$x$可导，则复合函数$y=f(g(x))$在点$x$可导，且有关系式：$$\frac{dy}{dx} = \frac{dy}{du} * \frac{du}{dx}$$<br \>
OK，我们来讨论多元复合函数的求导公式，对于二元函数，设$u = f(x, y)$,而$x,y$又是自变量s,t的函数$$x=\varphi(s, t) y =  \psi (s,t) $$可导出如下公式：$$\frac{\partial u}{\partial t} = \frac{\partial u}{\partial x} * \frac{\partial x}{\partial t} +  \frac{\partial u}{\partial y} * \frac{\partial y}{\partial t}$$
$$\frac{\partial u}{\partial s} = \frac{\partial u}{\partial x} * \frac{\partial x}{\partial s} +  \frac{\partial u}{\partial y} * \frac{\partial y}{\partial s}$$<be \>
通过导数的定义式容易证明上述成立。

### Recap：
求C对任意权重W的导数，难点在于我们很难（几乎不可能）把显式表达式写出来对其求导。反向传播，就利用链式法则把C对$j^{th}$层权重的导数通过$W_{j}$与C对$j-1^{th}$层权重的导数连接起来，这样我们就可以做到，已知C对最后一层权重的导数，求出C对前一层权重的导数，以此类推，可以求的任意$\frac{\partial C}{\partial W_{j}}$,所以最重要的两步在于：

##### 1.定义$\delta$(课程中称为error_term),并找出递推关系式：$$\delta^{j} = \frac{\partial C}{\partial z^{j}}$$
由$a^{j} = \sigma (z^{j})$，导出：$$\delta^{j} = \frac{\partial C }{ \partial a^{j} }* \frac{\partial a^{j}}{\partial z^{j}}=\frac{\partial C }{\partial a^{j}} * {\sigma (z^{j})}'$$
然后我们用链式法则，推导$\delta^{j}$与$\delta^{j+1}$的关系
$$\delta^{j} = \frac{\partial C}{\partial z^{j}} = \frac{\partial C}{\partial z^{j+1}} * \frac{\partial z^{j+1}}{\partial z^{j}}$$
由$z^{j+1}={W_{j+1}}^{T}*\delta (z^{j})$ (这里一定要注意维数，并且输入输出都是列向量),导出：
$$\delta^{j} = ({W_{j+1}}^{T} * \delta^{j+1})\odot {\sigma(z^{j})}'$$
维数是：$j^{th}$层节点数 乘 1

#### 2、我们找出$\delta$的递推关系式之后，把$\delta$和$\frac{\partial C}{\partial W_{j}}$联系起来：
$$\frac{\partial C}{\partial W_{j}} = \frac{\partial C}{\partial z^{j}} * \frac{\partial z^{j}}{\partial W_{j}}=\delta^{j} * \frac{\partial z^{j}}{\partial W_{j}}$$
由$z^{j}={W_{j}}^{T} * a^{j-1}$，导出：
$$\frac{\partial C}{\partial W_{j}} = \delta ^{j} * (a^{j-1})^{T}$$

#### 至此，我们就完成了back-propagation公式的推导（对于有bias项的同理可以推出，课程中暂时没有出现）。

### summary：
>* 1.计算最后第J层error_term:$$\delta^{J} = (a^{J}-y)\odot {\sigma(z^{J})}'$$
>* 2.计算error_term的递推公式：$$\delta^{j} = ({W_{j+1}}^{T} * \delta^{j+1})\odot {\sigma(z^{j})}'$$
>* 3.计算$\frac{\partial C}{\partial W_{j}}$与error_term的关系：$$\frac{\partial C}{\partial W_{j}} = \delta ^{j} * (a^{j-1})^{T}$$

最后附上，课程项目一的[github](https://github.com/laycoding/Udacity_Projects/blob/master/Your_first_neural_network.ipynb)地址


```python

```
