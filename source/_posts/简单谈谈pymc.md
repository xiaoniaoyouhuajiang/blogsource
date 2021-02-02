---
title: 受到广泛应用的贝叶斯方法(python实验)
---
# 受到广泛应用的贝叶斯方法(python实验)

---
贝叶斯理论是概率论中的一个受到广泛提及的理论方法，简单来说就是将现实中的事件都看作是相互联系的，某一事件的可能性是受到它的先验影响的，关于“贝爷”，他的理论有多牛逼我就不说了，各位自行百度。
**主要说说mcmc是怎么样估计模型参数的吧~**


```python
from IPython.display import Image

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import scipy
import scipy.stats as stats
import scipy.optimize as opt
import statsmodels.api as sm

%matplotlib inline
plt.style.use('bmh')
colors = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', 
          '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']

messages = pd.read_csv('data/hangout_chat_data.csv')
```

    D:\Anaconda3\lib\site-packages\statsmodels\compat\pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools



```python
messages.head()
```







>一个好奇的男孩每天都在观察他家经过的汽车数量。他孜孜不倦地记下每天通过的汽车总数。在过去一周，他的笔记本包含以下数量：12,33,20,29,20,30,18

从贝叶斯的角度来看，这些数据是由随机过程生成的。但是，某一次观察到的数据，它是固定的并且不会改变。该随机过程具有一些固定的模型参数。贝叶斯使用概率分布来表示这些参数中的不确定性。

因为男孩正在计数（对象是非负整数），所以通常使用泊松分布来模拟数据（例如，随机过程）。泊松分布采用单个参数$$\mu$$，其描述数据的均值和方差。你可以在下面看到3个泊松分布，其中包含不同的$$\mu$$值。

$$p(x \ | \ \mu) = \frac{e^{-\mu}\mu^{x}} {x!} \mbox{    for     } 
x = 0, 1, 2, \cdots$$



$$\lambda = E(x) = Var(\mu)$$

*将上面的内容可视化*:


```python
fig = plt.figure(figsize=(11,3))
ax = fig.add_subplot(111)
x_lim = 60
mu = [5, 20, 40]
for i in np.arange(x_lim):
    plt.bar(i, stats.poisson.pmf(mu[0], i), color=colors[3])
    plt.bar(i, stats.poisson.pmf(mu[1], i), color=colors[4])
    plt.bar(i, stats.poisson.pmf(mu[2], i), color=colors[5])
    
_ = ax.set_xlim(0, x_lim)
_ = ax.set_ylim(0, 0.2)
_ = ax.set_ylabel('Probability mass')
_ = ax.set_title('Poisson distribution')
_ = plt.legend(['$\mu$ = %s' % mu[0], '$\mu$ = %s' % mu[1], '$\mu$ = %s' % mu[2]])
```


![png](https://thumbnail0.baidupcs.com/thumbnail/ee891ef82c9a1a4d226af822d47e973c?fid=1076969831-250528-216886472348547&time=1538546400&rt=sh&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-vJDe8Ex1f0KyhcON%2Blwwg3vDQV0%3D&expires=8h&chkv=0&chkbd=0&chkpc=&dp-logid=6391904499165231363&dp-callid=0&size=c710_u400&quality=100&vuk=-&ft=video)


下面用真实的数据去进行**概率建模**，刚才加载的是通话的时间数据集。对于样本数据中的**回应时长（response_time）**，我们来探索这一行为时间分布的参数，当然，采用的是贝叶斯方法，我们先将真实数据可视化。


```python
fig = plt.figure(figsize=(11,3))
_ = plt.title('Frequency of messages by response time')
_ = plt.xlabel('Response time (seconds)')
_ = plt.ylabel('Number of messages')
_ = plt.hist(messages['time_delay_seconds'].values, 
             range=[0, 60], bins=60, histtype='stepfilled')
```


![png](https://thumbnail0.baidupcs.com/thumbnail/a055c6d27ec5a1bb4c30a9d4d17b0ce5?fid=1076969831-250528-989296221144536&time=1538546400&rt=sh&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-6fobyldL4xyk4a%2FpmnuFzb840Xg%3D&expires=8h&chkv=0&chkbd=0&chkpc=&dp-logid=6391922076248484241&dp-callid=0&size=c710_u400&quality=100&vuk=-&ft=video)


### 频率学派估算$\mu$的方法
我们考虑使用scipy的优化方法来优化我们构建的一个函数，通过智能迭代"$$\mu$$"，进行最大似然估计。


```python
y_obs = messages['time_delay_seconds'].values

##构建目标函数
def poisson_logprob(mu,sign=-1):
    return np.sum(sign*stats.poisson.logpmf(y_obs,mu))

freq_results = opt.minimize_scalar(poisson_logprob)
%time print("$mu$的最佳估计时间是：%s" % freq_results['x'])
```

    $mu$的最佳估计时间是：18.2307692323807
    Wall time: 0 ns


    D:\Anaconda3\lib\site-packages\scipy\optimize\optimize.py:2189: RuntimeWarning: invalid value encountered in double_scalars
      w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom


所以，$\mu$的估计值是18.0413533867。优化技术不提供任何不确定性的度量 - 它只返回一个点值。而且效率非常高......

下图说明了我们正在优化的功能。在$$\mu$$的每个值处，该图显示了给定数据和模型的$$\mu$$处的对数概率。优化器以爬山的方式工作 - 从曲线上的随机点开始并逐渐爬升直到它无法达到更高的点。


```python
x = np.linspace(1, 60)
y_min = np.min([poisson_logprob(i, sign=1) for i in x])
y_max = np.max([poisson_logprob(i, sign=1) for i in x])
fig = plt.figure(figsize=(6,4))
_ = plt.plot(x, [poisson_logprob(i, sign=1) for i in x])
_ = plt.fill_between(x, [poisson_logprob(i, sign=1) for i in x], 
                     y_min, color=colors[0], alpha=0.3)
_ = plt.title('Optimization of $\mu$')
_ = plt.xlabel('$\mu$')
_ = plt.ylabel('Log probability of $\mu$ given data')
_ = plt.vlines(freq_results['x'], y_max, y_min, colors='red', linestyles='dashed')
_ = plt.scatter(freq_results['x'], y_max, s=110, c='red', zorder=3)
_ = plt.ylim(ymin=y_min, ymax=0)
_ = plt.xlim(xmin=1, xmax=60)
```


![png](https://thumbnail0.baidupcs.com/thumbnail/17d84526162738736c537c831ce950ad?fid=1076969831-250528-358410629706121&time=1538546400&rt=sh&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-UPS6IeJZEB%2BLiL9nbJBja%2FgNjzc%3D&expires=8h&chkv=0&chkbd=0&chkpc=&dp-logid=6391931174921436427&dp-callid=0&size=c710_u400&quality=100&vuk=-&ft=video)


上述优化估计泊松模型的参数（$$\mu$$）为18.我们知道对于任何泊松分布，参数$$\mu$$表示其均值和方差。下图说明了这种分布。


```python
fig = plt.figure(figsize=(11,3))
ax = fig.add_subplot(111)
x_lim = 60
mu = np.int(freq_results['x'])
for i in np.arange(x_lim):
    plt.bar(i, stats.poisson.pmf(mu, i), color=colors[3])
    
_ = ax.set_xlim(0, x_lim)
_ = ax.set_ylim(0, 0.1)
_ = ax.set_xlabel('Response time in seconds')
_ = ax.set_ylabel('Probability mass')
_ = ax.set_title('Estimated Poisson distribution for Hangout chat response time')
_ = plt.legend(['$\lambda$ = %s' % mu])

```


![png](https://thumbnail0.baidupcs.com/thumbnail/308d22c22ef18d9680a1bf08793edcfc?fid=1076969831-250528-407159994590110&time=1538546400&rt=sh&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-J5SzTn%2BGrjpD0lOJtS8VVyfs5P8%3D&expires=8h&chkv=0&chkbd=0&chkpc=&dp-logid=6391928274122964797&dp-callid=0&size=c710_u400&quality=100&vuk=-&ft=video)


上述泊松模型和$$\mu$$的估计值表明观测的概率小于10或大于30可能性是很小的。绝大多数概率分布在10到30之间。但是，我们知道，实际数据中，观测值在0到60之间。

## 理论：贝叶斯公式以及贝叶斯推断
---
如果你以前遇到贝叶斯定理，那么下面的公式你看着就会很熟悉。

$$\overbrace{p(\mu \ |\ Data)}^{\text{posterior}} = \dfrac{\overbrace{p(Data \ | \ \mu)}^{\text{likelihood}} \cdot \overbrace{p(\mu)}^{\text{prior}}}{\underbrace{p(Data)}_{\text{marginal likelihood}}}$$


```python
Image('graphics/Poisson-dag.png', width=320)
```




![png](https://thumbnail0.baidupcs.com/thumbnail/e61780bee60992bf09a44a2c93b90016?fid=1076969831-250528-383221845757493&time=1538546400&rt=sh&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-QyjkU0eG3X8yLGZD3N3y3fUbLCI%3D&expires=8h&chkv=0&chkbd=0&chkpc=&dp-logid=6391940674246702074&dp-callid=0&size=c710_u400&quality=100&vuk=-&ft=video)



上面的模式可以解释如下（从上倒下）：
 - 我们观察每个数据集中的的数据（y）i（观察到的数据）
 - 这个数据是由一个随机过程产生的，我们认为这个过程可以表示为泊松分布（近似地）
 - 这个Poisson分布有一个参数$$\mu$$，我们知道它在0到60之间（由Prior分布）
   - 我们将$$\mu​$$建模为均匀分布，因为我们对这个范围内的预期位置没有先验

### MCMC的神奇机制
马尔可夫链蒙特卡罗（MCMC）的过程在下面的动画中很好地说明。 MCMC采样器从先前分布中提取参数值，并计算数据来具有这些参数值的分布的可能性。

$$\overbrace{p(\mu \ |\ Data)}^{posterior} \varpropto \overbrace{p(Data \ | \ \mu)}^{likelihood} \cdot \overbrace{p(\mu)}^{prior}$$

该计算充当MCMC采样器的指导灯。当它从参数先验中抽取值时，它会计算这些参数给出数据的可能性 - 并将尝试引导采样器走向更高概率的区域。

在与上面讨论的频率优化技术的概念上类似的方式中，MCMC采样器向最高可能性区域移动。然而，贝叶斯方法并不关心绝对最大值的发现 - 而是在最高概率区域内遍历和收集样本。收集的所有样本都被认为是可靠的参数。


```python
Image(url='graphics/mcmc-animate.gif')
```




<img src="https://thumbnail0.baidupcs.com/thumbnail/0ddd06e7c8f4932ae6aff0513d2b88a8?fid=1076969831-250528-64833448167890&time=1538546400&rt=sh&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-KdOxChRLmz8ufyLOpeVxznA6%2FlI%3D&expires=8h&chkv=0&chkbd=0&chkpc=&dp-logid=6391981313235955834&dp-callid=0&size=c710_u400&quality=100&vuk=-&ft=video"/>



### 进一步
在初步了解了pymc以及mcmc方法是干什么的之后，我们有了以下认识：
* pymc准确性ok，pymc内置的mcmc性能也很不错
* 贝叶斯方法的解释性强
* 由于返回的预估参数是一组分布，对于模型的准确性能有更好的把握；


```python

```
