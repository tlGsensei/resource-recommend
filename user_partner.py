import pandas as pd
import numpy as np
import torch

teacher = pd.read_excel('merged_user_data.xlsx')

index = teacher['index']
segment = teacher['学段']
subject = teacher['学科']
district = teacher['地区']
school = teacher['学校']
score = teacher['我的备课库中备课数量'] + teacher['我的检课中的检课数量'] + teacher['我的上课中的上课数量'] + teacher['我的听课中的听课数量'] + teacher['我的反思的反思数量'] + teacher['我加入的社区数量'] + teacher['我创建的社区数量']

# 输入用户的index 学段 学科 地区 学校
x0 = "0186cb3f8351402ebc22f67614876520"
x1 = "小学"
x2 = "数学"
x3 = "甘肃武威市凉州区"
x4 = "武威新城区第一小学"

teacher_filtered = teacher

result = teacher_filtered.copy()
result['weight'] = 0  # 初始化权重列
result['probability'] = 0 # 初始化推荐概率列
result['score'] = score

# 找学科相同的老师，赋予权重值为1
xueke_match_index = result[result['学科'] == x2].index
result.loc[xueke_match_index, 'weight'] = 1 * 0.25

# 找且学段相同的老师，赋予权重值为2
xueduan_match_index = result[(result['学科'] == x2) & (result['学段'] == x1)].index
result.loc[xueduan_match_index, 'weight'] = 2 * 0.25

# 找且省市区均相同的老师，赋予权重值为3
province_city_district_match_index = result[(result['学科'] == x2) & (result['学段'] == x1) & (result['地区'] == x3)].index
result.loc[province_city_district_match_index, 'weight'] = 3 * 0.25

# 找且学校相同的老师，赋予最高权重值4
school_match_index = result[(result['学科'] == x2) & (result['学段'] == x1) & (result['地区'] == x3) & (result['学校'] == x4)].index
result.loc[school_match_index, 'weight'] = 4 * 0.25

# 排除姓名与x0相同的结果
result = result[result['index'] != x0]

# 归一化 score 列
result['normalized_score'] = (result['score'] - result['score'].min()) / (result['score'].max() - result['score'].min())

# 综合权重和归一化 score 计算综合权重
result['combined_weight'] = result['weight'] + result['normalized_score']

# 将权重列转化为概率
result['probability'] = np.exp(result['combined_weight']) / np.sum(np.exp(result['combined_weight']))

# 推荐列表按照权重值由高到低排序
result = result.sort_values(by='probability', ascending=False)
# 确保前7个老师是概率最高的
top = result.head(7)

for i in range(10):
    # 从剩余的结果中按照概率随机选择3个老师
    remaining = result.iloc[7:]
    remaining['probability_normalized'] = remaining['probability'] / remaining['probability'].sum()  # Normalize probabilities
    random_three_indices = np.random.choice(remaining.index, size=3, p=remaining['probability_normalized'], replace=False)
    random_three = remaining.loc[random_three_indices]
    final_result = pd.concat([top, random_three])

    # 按照推荐概率排序并选择前10个老师
    final_result = final_result.sort_values(by='probability', ascending=False).head(10)

    print(final_result[['index', '学科', '学段', '地区', '学校', 'score', 'probability']])

pass