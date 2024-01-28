import pandas as pd
import glob
import openpyxl.utils.exceptions as excel_exceptions

# 获取所有要读取的 Excel 文件的文件名
file_pattern = 'beishi/待完善元数据/*.xls'  # 指定 Excel 文件所在的文件夹路径和文件名模式
file_list = glob.glob(file_pattern)
# 创建一个空的 DataFrame 来存储所有 Excel 文件的内容
unlabeled_data = pd.DataFrame()
# 循环遍历文件列表并读取数据
for file in file_list:
    # 读取 Excel 文件
    data = pd.read_excel(file)
    unlabeled_data = pd.concat([unlabeled_data, data], ignore_index=True)
# print(unlabeled_data)

labeled_data = pd.DataFrame()
labeled_file = 'beishi/已标记元数据.xls'
labeled_data = pd.read_excel(labeled_file)
# print(labeled_data)

all_data = pd.DataFrame()
all_data = pd.concat([labeled_data, unlabeled_data], ignore_index=True)
# print(all_data)

# 读取用户评价
comment = pd.read_excel('beishi/资源埋点数据.xlsx', sheet_name='用户的评价内容')
# print(comment)
# 合并 comment 表，将相同资源ID的评价内容合并到第一个资源中
comment = comment.groupby("资源ID")["评价内容"].apply(lambda x: ' '.join(x)).reset_index()

# 读取单个用户观看资源中心内容的次数
times = pd.read_excel('beishi/资源埋点数据.xlsx', sheet_name='单个用户观看资源中心内容的次数')
# print(times)

# 读取资源预览被点击查看的时长
time = pd.read_excel('beishi/资源埋点数据.xlsx', sheet_name='资源预览被点击查看的时长')
# 使用条件筛选，筛选"操作"列为"进入资源详情页"的记录
filtered_time = time[time["操作"] == "进入资源详情页"]
# 将 "时间" 列解析为日期时间对象
filtered_time["时间"] = pd.to_datetime(filtered_time["时间"])
# 对于访问资源id和用户id相同的记录，只保留“时间”最早的记录
filtered_time = filtered_time.sort_values(by="时间").groupby(["访问资源id", "用户id"]).first().reset_index()
# 根据用户id进行分组，并对每组按时间排序
sorted_filtered_time = filtered_time.sort_values(by=["用户id", "时间"])
# 用户id和按照时间访问资源id的新表，资源序列用空格分割
history_track = sorted_filtered_time.groupby("用户id")["访问资源id"].apply(lambda x: ' '.join(x)).reset_index()
history_track.to_excel('history_track.xlsx', index=False)

# 处理非法字符并替换为空字符串
def clean_string(value):
    if pd.notna(value):
        return ''.join(filter(lambda x: x.isprintable(), str(value)))
    return value

# 统计资源信息，实际上没有资源ID重复的情况
resource_all = all_data.iloc[:, 0:26]
# print(resource_all)
resource = resource_all.drop_duplicates(subset=["资源ID"], keep="first")
# 合并两个 DataFrame，根据资源ID列匹配
resource = resource.merge(comment, on="资源ID", how="left")
# print(resource)
# 清洗 DataFrame 中的每个单元格
for col in resource_all.columns:
    resource[col] = resource_all[col].apply(clean_string)
# 将 DataFrame 写入 Excel 文件，如果还有非法字符，会被替换为空字符串
try:
    resource.to_excel('resource.xlsx', index=False)
except excel_exceptions.IllegalCharacterError as e:
    print(f"An IllegalCharacterError occurred: {e}")

# 统计教师信息
teacher_all = all_data.iloc[:, 26:40]
# print(teacher_all)
teacher = teacher_all.drop_duplicates(subset=["昵称", "姓名"], keep="first")
# print(teacher)
teacher.to_excel('teacher.xlsx', index=False)