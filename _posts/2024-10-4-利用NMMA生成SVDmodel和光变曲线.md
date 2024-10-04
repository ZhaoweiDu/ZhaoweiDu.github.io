---
layout:     post
title:      利用NMMA生成SVD光变曲线和生成SVDmodel
subtitle:   NMMA教程（1）
date:       2024-10-4
author:     Zhao-Wei Du
catalog: false
tags:
    - Python
---
NMMA（The Nuclear Multimessenger Astronomy Framework）是一个集成了引力波、千新星、GRB余辉和超新星的`Python`模型，目前本人只理解了部分功能，特做此笔记以方便后人借鉴。

[NMMA官方文档](https://nuclear-multimessenger-astronomy.github.io/nmma/#)

[NMMAgithub页面](https://github.com/nuclear-multimessenger-astronomy/nmma)

### 1、SVD

SVD是Singular Value Decomposition的简写，是一种线性代数工具，广泛用于矩阵分解、降维、数据压缩和特征提取等领域。SVD的基本思想是将一个矩阵分解为三个矩阵的乘积，使得矩阵的结构能够被很好地表达。

#### SVD计算思路

对于一个矩阵$A$（大小为$m\times n$），SVD将其分解为以下形式：

$$
A=U\Sigma V^{\rm T}
$$
其中：

- $U$是一个$m\times n$的正交矩阵，其列向量为$A$的左奇异向量。
- $\Sigma$是一个$m\times n$的对角矩阵，其对角线上的元素为$A$的奇异值，按降序排列。奇异值可以理解为矩阵的“尺度”。
- $V$是一个$m\times n$的正交矩阵，其列向量为$A$的右奇异向量。

#### 计算步骤

1. **计算协方差矩阵**：首先计算矩阵$A$的协方差矩阵。对于方阵$A$，协方差矩阵为$A^{\rm T}A$或$AA^{\rm T}$。
2. **特征值分解**：
   - 计算协方差矩阵的特征值和特征向量。对于$A^{\rm T}A$和$AA^{\rm T}$，它们的特征值分别是$V$和$U$的特征向量，而特征值的平方根就是奇异值$\Sigma$。
3. **构建奇异值矩阵$\Sigma$**：将奇异值排列在对角线上，形成矩阵$\Sigma$。非对角线元素为零。
4. **构建矩阵$U$和$V$**：
   - 使用计算出的特征向量来构建正交矩阵$U$和$V$。
   - 确保矩阵$U$和$V$的列向量是单位正交向量。
5. **最终组合**：通过公式$A=U\Sigma V^{\rm T}$，你可以将矩阵$A$恢复，或者在降维的情况下近似恢复。

#### SVD的应用

1. **降维**：通过只保留最大的一部分奇异值，可以在保留矩阵主要特征的同时，极大地减少数据维度。这在图像压缩、数据压缩和降噪等方面非常有用。
2. **低秩近似**：通过丢弃小奇异值，SVD可以用于构建矩阵的低秩近似，从而减少计算负担。
3. **数据分析**：在推荐系统、主题建模等问题中，SVD可以用于提取重要的潜在结构。

### 2、在NMMA中利用SVDmodel生成千新星的光变曲线

[NMMA官方提供的SVDmodel](https://gitlab.com/Theodlz/nmma-models)

```python
import numpy as np
import nmma.em.io as io
from nmma.em.model import SVDLightCurveModel

#这个地方的模型名字一定要和你模型的名字匹配
model_name = "Bu2019lm" #一般这里的模型名字是‘Bu2019lm.joblib’的后缀前的部分
dt = 0.2
sample_times = np.arrange(0.5, 20 + dt, 0.1)
filts = ["sdssu"] #这里的作用是确定画出来的波段

ModelPath = "svdmodels" #这里不能照抄，是你模型在计算机中的路径
#推荐使用绝对路径以避免程序移动中出错
#eg: /home/zhaoweidu/nmma_code/svdmodels

#你使用的模型需要输入哪些参数，在这里需要列出来
#如果你使用的模型所需要的参数正好和NMMA某个模型一致，那么你可以直接调用NMMA的dictionary来节约时间
#model_parameters = nmma.em.model.model_parameters_dict["Bu2019lm"]
model_parameters = ['log10_mej_dyn', 'log10_mej_wind', 'KNphi', 'KNtheta']

light_curve_model = SVDLightCurveModel(
        model_name,
        sample_times,
        svd_path=ModelPath,
        interpolation_type="sklearn_gp", #目前只确认了这种方法的插值没有bug
        model_parameters=training_model.model_parameters,
        filters=filts,
    )

#输入你想计算的参数的值
data = {'log10_mej_dyn': -2, 'log10_mej_wind': -2, 'KNphi': 45.0, 'KNtheta': 72.54}
#定义红移，如果不想定义红移也可以定义"luminosity_distance"
data["redshift"] = 0
lbol, mag = light_curve_model.generate_lightcurve(sample_times, data)

#此时的mag是一个dictionary格式的数据，通过调用filts的keywords可以输入具体的对应时间的绝对星等
```

### 3、在NMMA中生成自己的SVDmodel

首先，生成专属于自己的SVDmodel需要一系列特殊命名的数据，如果自己没有能力生成数，可以从公开的数据集中选取，[一些使用POSSIS算法生成的数据](https://drive.google.com/drive/folders/1QCajfQtxOc74NgFWkZTqG_dD1hGdLogB?usp=drive_link)。

如果你的数据集所使用的参数早就在NMMA中存在了，那么你就不需要自己再去提取数据了，可以直接使用内置的函数来进行数据提取。

```python
import numpy as np
import os, sys, time, glob
import copy
import nmma.em.io as io
from nmma.em import training, utils, model_parameters

tini, tmax, dt = 0.1, 5.0, 0.2
tt = np.arange(tini, tmax + dt, dt) 

model_name = "Bu2019lm"
filts = ["sdssu"]


dataDir = "lcs" #这个是你数据集的文件夹
ModelPath = "svdmodels" #这个是一会模型生成的位置
filenames = glob.glob("%s/*.dat" % dataDir)

data = io.read_photometry_files(filenames, filters=filts)
training_data, parameters = model_parameters.Bu2019lm(data)

training_model = training.SVDTrainingModel(
    model_name,
    copy.deepcopy(training_data),
    parameters,
    tt,
    filts,
    svd_path=ModelPath,
    n_coeff=5, #这个似乎是控制模型精细度的，越高模型越精细，相应的生成模型的速度也就越慢
    interpolation_type="sklearn_gp",
    n_epochs=100
)
```

运行上述代码时，会出现如下字样：

```
The grid will be interpolated to sample_time with interp1d
Normalizing mag filter sdssu...
Training model...
Computing GP for filter sdssu...
Calculating the coefficents
Coefficient 1/5...
Coefficient 2/5...
Coefficient 3/5...
Coefficient 4/5...
Coefficient 5/5...
```

运行结束后，会在你输入ModelPath的地方生成一个文件夹和一个文件，明明为“Bu2019lm.joblib”的文件就是本次training生成的SVDmodel，文件夹里面是特殊频段相关的文件，后缀也为“.joblib”。



