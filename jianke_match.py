import pandas as pd
import numpy as np
import torch

# 输入：资源指标，学科，年段，资源名称，教材版本

# resource = pd.read_csv('resource.csv')
score = pd.read_csv('score.csv')
resource_new = pd.read_excel('resource_new.xlsx')

# 处理缺失值
resource_new['资源名称'] = resource_new['资源名称'].fillna('')  # 将缺失值填充为空字符串

# 创建资源ID列
resource_new['资源ID'] = ''

# 遍历 resource_new 表的每一行，查找匹配的资源ID
for index, row in resource_new.iterrows():
    resource_name = row['资源名称']
    matching_rows = score[score['资源标题'].str.contains(resource_name, na=False)]

    # 如果有匹配的行，取第一行的资源ID值
    if not matching_rows.empty:
        resource_id = matching_rows.iloc[0]['资源ID']
        resource_new.at[index, '资源ID'] = resource_id

# 将合并后的结果保存到新的 Excel 文件中
resource_new.to_excel('updated_resource_new.xlsx', index=False)