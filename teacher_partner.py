import pandas as pd
import numpy as np
import torch

teacher = pd.read_excel('teacher.xlsx')

name = teacher['姓名']
segment = teacher['学段']
subject = teacher['学科']
province = teacher['省']
city = teacher['市']
district = teacher['区']
school = teacher['学校']

# 输入用户的学段 学科 省 市 区 学校，以“魏静园”为例
x0 = "魏静园"
x1 = "小学"
x2 = "语文"
x3 = "北京"
x4 = "北京市"
x5 = "海淀区"
x6 = "北京师范大学跨越式课题组"

# 硬性条件：筛选学段和学科与输入相同的老师
teacher_filtered = teacher[(teacher['学段'] == x1) & (teacher['学科'] == x2)]

result = teacher_filtered.copy()
result['weight'] = 0  # 初始化权重列
result['probability'] = 0 # 初始化推荐概率列

# 找省相同的老师，赋予权重值为1
province_match_index = result[result['省'] == x3].index
result.loc[province_match_index, 'weight'] = 1

# 找省市相同的老师，赋予权重值为2
province_city_match_index = result[(result['省'] == x3) & (result['市'] == x4)].index
result.loc[province_city_match_index, 'weight'] = 2

# 找省市区均相同的老师，赋予权重值为3
province_city_district_match_index = result[(result['省'] == x3) & (result['市'] == x4) & (result['区'] == x5)].index
result.loc[province_city_district_match_index, 'weight'] = 3

# 找学校相同的老师，赋予最高权重值4
school_match_index = result[result['学校'] == x6].index
result.loc[school_match_index, 'weight'] = 4

# 排除姓名与x0相同的结果
result = result[result['姓名'] != x0]
# 推荐列表按照权重值由高到低排序
result = result.sort_values(by='weight', ascending=False)

# 将权重列转化为概率
result['probability'] = np.exp(result['weight']) / np.sum(np.exp(result['weight']))
# 确保前三个老师是概率最高的
top_three = result.head(3)

for i in range(10):
    # 从剩余的结果中按照概率随机选择三个老师
    remaining = result.iloc[3:]
    remaining['probability_normalized'] = remaining['probability'] / remaining['probability'].sum()  # Normalize probabilities
    random_three_indices = np.random.choice(remaining.index, size=3, p=remaining['probability_normalized'], replace=False)
    random_three = remaining.loc[random_three_indices]
    final_result = pd.concat([top_three, random_three])

    # 按照推荐概率排序并选择前六个老师
    final_result = final_result.sort_values(by='probability', ascending=False).head(6)

    print(final_result[['姓名', '学科', '学段', '省', '市', '区', '学校', 'probability']])