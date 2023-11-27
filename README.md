[toc]

# 基于多粒度兴趣建模的学习资源推荐算法代码

## 代码结构

.
├── data # 数据
│   ├── alibaba-fashion
│   ├── amazon-book
│   ├── last-fm
│   ├── mooccube # MOOCCube 数据
│   └── mooper # MOOPer 数据
├── main.py # 入口文件
├── app.py # 算法后端接口
├── modules # 模型
│   ├── model.py # 模型代码
│   
├── output # 画图
│   └── img # 图片保存地址
├── README.md
├── run.sh # 代码启动脚本
|
├── utils # 工具包
│   ├── dataset.py # 数据集
│   ├── evaluate.py # 算法评测代码
│   ├── helper.py # 工具类
│   ├── load_data.py # 数据加载
│   ├── metrics.py # 评测指标
│   ├── parser.py # 命令行参数解析
│   
└── weights # 模型保存


## 代码环境

` 244 服务器 conda 环境 torch1.9 包含代码运行所有环境, 执行如下命令可切换至目标环境 `

```bash
    conda activate torch1.9
```

The code has been tested running under Python 3.8.11. The required packages are as follows:

- pytorch == 1.9.0
- numpy == 1.15.4
- scipy == 1.1.0
- sklearn == 0.20.0
- torch_scatter == 2.0.5
- dgl == 0.6.1
- networkx == 2.5
- pandas == 1.3.1
- flask == 2.2.2 # 后端代码依赖，算法代码不需要


## 代码启动

### 算法代码

```bash
    ./run.sh
```

### 后端接口

```bash
    python app.py
```

` 默认启动 docker 9000 端口， 该端口映射的物理端口为 24202`