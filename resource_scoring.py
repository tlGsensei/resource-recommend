import pandas as pd
import numpy as np
from datetime import datetime
import openpyxl.utils.exceptions as excel_exceptions


# resource = pd.read_excel('resource.xlsx')
# resource.to_csv('resource.csv', index=False)
# csv文件的读取速度更快
resource = pd.read_csv('resource.csv')
columns = ['资源ID', '资源质量', '适用学段', '适用学科', '浏览量', '创建时间']
# 筛选列
filtered_resource = resource[columns]


# 资源质量“良好”时的分数
score_B = 0.7
def assign_score(row):
    if row['资源质量'] == '优秀':
        return 1
    elif row['资源质量'] == '良好':
        return score_B
    else:
        return 0  # 返回 None 或其他默认值，取决于需要处理的情况
filtered_resource['资源质量分数'] = filtered_resource.apply(assign_score, axis=1)
# 40498个资源“资源质量”为None的有7088个；为“优秀”的有1730个，为“良好”的有31680个，计算平均值作为缺失项的分数0.716
non_zero_scores = filtered_resource[filtered_resource['资源质量分数'] != 0]['资源质量分数']
A_scores = filtered_resource[filtered_resource['资源质量分数'] == 1.0]['资源质量分数']
B_scores = filtered_resource[filtered_resource['资源质量分数'] == score_B]['资源质量分数']
average_non_zero_score = non_zero_scores.mean()
# 筛选出“资源质量分数”为零的行，并将它们替换为非零分数的平均值
filtered_resource.loc[filtered_resource['资源质量分数'] == 0, '资源质量分数'] = average_non_zero_score


# 设置基础评分和对数转换的底数
base_score = 1
log_base = 2  # 以2为底的对数转换
# 对浏览量不为零的资源进行对数转换和归一化
non_zero_views = filtered_resource[filtered_resource['浏览量'] != 0]['浏览量']
filtered_resource.loc[filtered_resource['浏览量'] != 0, '浏览量分数'] = base_score + np.log(non_zero_views) / np.log(log_base)
max_score = filtered_resource['浏览量分数'].max()
min_score = filtered_resource['浏览量分数'].min()
filtered_resource.loc[filtered_resource['浏览量'] != 0, '浏览量分数'] = (filtered_resource['浏览量分数'] - min_score) / (max_score - min_score)
# 对浏览量为零的资源直接将评分置为0
filtered_resource.loc[filtered_resource['浏览量'] == 0, '浏览量分数'] = 0


# 设置基础评分和对数转换的底数
base_score = 12
log_base = 2  # 以2为底的对数转换
# 假设创建时间格式为字符串，需要转换为 datetime 类型
filtered_resource['创建时间'] = pd.to_datetime(filtered_resource['创建时间'])
# 计算当前时间
current_time = datetime.now()
# 计算资源存在的时间（以天为单位）
filtered_resource['存在时间'] = (current_time - filtered_resource['创建时间']).dt.days
# 假设存在时间越短得分越高，通过对存在时间进行归一化处理来得到时间评分
max_time = filtered_resource['存在时间'].max()
min_time = filtered_resource['存在时间'].min()
filtered_resource['时间分数'] = base_score - np.log(filtered_resource['存在时间']) / np.log(log_base)
max_score = filtered_resource['时间分数'].max()
min_score = filtered_resource['时间分数'].min()
filtered_resource['时间分数'] = (filtered_resource['时间分数'] - min_score) / (max_score - min_score)

# CRITIC权重
weight_quality = 0.2278
weight_views = 0.2367
weight_time = 0.5355
# 计算加权平均评分
filtered_resource['综合评分'] = (filtered_resource['资源质量分数'] * weight_quality +
                               filtered_resource['浏览量分数'] * weight_views +
                               filtered_resource['时间分数'] * weight_time) / (weight_quality + weight_views + weight_time)

filtered_resource.to_csv('score.csv', encoding='utf-8-sig', index=False)
filtered_resource.to_excel('score.xlsx')
pass