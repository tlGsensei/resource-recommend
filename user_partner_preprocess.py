import pandas as pd

teacher = pd.read_excel('user.xlsx')
# teacher_index = pd.read_excel('user_index/1.xls')
# 读取四个Excel文件并合并
files = ['user_index/1.xls', 'user_index/2.xls', 'user_index/3.xls', 'user_index/4.xls']
# 创建一个空的DataFrame列表
dfs = []
for file in files:
    # 读取每个Excel文件
    df = pd.read_excel(file)
    # 添加到DataFrame列表
    dfs.append(df)
# 合并所有DataFrame
teacher_index = pd.concat(dfs, ignore_index=True)

# 假设第一列都是用户ID，将第一列设为索引
teacher.set_index(teacher.columns[0], inplace=True)
teacher_index.set_index(teacher_index.columns[0], inplace=True)
# 合并两个表格
merged_table = pd.merge(teacher, teacher_index, left_index=True, right_index=True, how='inner')
# 重置索引以用户ID作为第一列
merged_table.reset_index(inplace=True)
# 删除名为 'Unnamed' 的列
merged_table = merged_table.loc[:, ~merged_table.columns.str.contains('^Unnamed')]

merged_table.to_excel('merged_user_data.xlsx', index=False)