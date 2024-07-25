## 代码环境
The code has been tested running under Python 3.8.11. The required packages are as follows:

- numpy == 1.24.4
- pandas == 2.0.3
- transformers == 4.36.2
- torch == 1.9.0

### 个人主页资源推荐（202407更新）
首次使用生成匹配表（这一步是对数据集的预处理，不需要针对每个用户都重新运行）：
```
bash ./run.sh
```
生成匹配表后，可直接运行：
```
python query.py --user_ID {ID}
```

### 个人主页同伴推荐（202407更新）
对数据集的预处理，不需要针对每个用户都重新运行:
```
python user_partner_preprocess.py
```
预处理后，可直接运行：
```
python user_partner.py
```

### 个人主页社区推荐
```
python community_recommend.py
```

### 备课中心资源推荐算法代码

```
python get_source_similarity.py
```
修改要求之后的备课中心资源推荐代码
```
python beike.py
```

### 备课中心协同备课教师推荐代码

```
python teacher_partner.py
```

### 检课中心资源推荐代码
```
python jianke.py
```

### 资源中心资源推荐代码
```
resource_recommend.py
```

