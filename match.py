import pandas as pd
import numpy as np

score = pd.read_csv('score.csv')
user = pd.read_excel('user.xlsx')

columns = ['适用学段', '适用学科']
filtered_score = score[columns]
segments1 = filtered_score['适用学段'].unique()
# ['初中' '高中' '小学' '小学,高中' '小学,初中,高中' '小学,初中' '初中,高中' nan]
subjects1 = filtered_score['适用学科'].unique()
# ['数学' '生物' '语文' '数学,信息科技' '英语' '化学' '科学' '道德与法治' '全部学科' '思想品德' '物理' '历史', '语文,数学,英语' '政治,道德与法治' '语文,化学' '地理' nan '劳动', '语文,数学,英语,音乐,美术,科学,物理,化学,生物,历史,地理,历史与社会,道德与法治,体育,劳动技术,劳动,综合实践,书法,班会' '音乐', '数学,英语' '班会' '信息科技' '思想品德,政治,道德与法治' '体育与健康' '劳动,综合实践' '体育' '美术' '科学,综合实践', '综合实践' '美术,书法' '科学,地理,综合实践' '综合实践,班会' '美术,综合实践' '语文,美术' '书法', '美术,道德与法治,综合实践' '语文,数学,美术,劳动,综合实践' '劳动技术,劳动,综合实践' '语文,综合实践,班会' '英语,科学', '语文,数学,综合实践' '数学,科学,物理,化学,生物' '历史,历史与社会' '道德与法治,综合实践,班会', '语文,道德与法治,综合实践,班会' '语文,班会' '体育,综合实践' '数学,物理,化学,生物' '科学,物理', '语文,数学,道德与法治,班会' '劳动技术,劳动', '语文,数学,英语,音乐,美术,科学,物理,化学,生物,历史,地理,历史与社会,道德与法治,体育,综合实践,书法,班会' '科学,生物,综合实践', '语文,书法' '英语,音乐,美术,道德与法治' '科学,生物' '音乐,美术,书法' '语文,美术,综合实践', '科学,历史与社会,道德与法治,综合实践' '科学,化学,生物,地理,历史与社会,综合实践' '美术,道德与法治', '物理,化学,生物,地理,综合实践' '历史与社会' '道德与法治,综合实践' '科学,物理,生物' '数学,科学,道德与法治,综合实践,班会', '语文,数学,英语,道德与法治,综合实践,班会' '地理,综合实践' '数学,科学,物理,化学,生物,道德与法治,综合实践' '语文,英语', '美术,历史与社会,综合实践' '语文,数学,英语,音乐,美术,科学,道德与法治,综合实践,书法' '音乐,体育,综合实践', '语文,数学,英语,音乐,美术,科学,生物,历史与社会,道德与法治,体育,班会' '科学,物理,化...
columns = ['学段', '学科']
filtered_user = user[columns]
segments2 = filtered_user['学段'].unique()
# ['小学' '初中' '\\N' '高中']
subjects2 = filtered_user['学科'].unique()
# ['信息科技' '数学' '英语' '语文' '\\N' '科学' '物理' '化学' '道德与法治' '历史' '政治' '劳动' '地理', '通用技术' '生物' '美术' '体育与健康（教师）' '体育与健康' '体育' '品德与生活' '音乐' '心理' '综合实践' '劳动技术', '书法']

# 创建一个空字典用于存储每个老师对应学段和学科的资源序号列表
teacher_resources = {}

# 遍历老师表中的每个老师
for teacher_id, teacher_row in user.iterrows():
    teacher_user_id = teacher_row['用户id']
    teacher_segment = teacher_row['学段']
    teacher_subject = teacher_row['学科']
    if teacher_segment != r'\N' and teacher_subject != r'\N':
        # 在资源表中筛选出符合老师学段和学科的资源序号列表，并排序
        matching_resources = score[
            score['适用学段'].fillna('').str.contains(teacher_segment, regex=False) &
            score['适用学科'].fillna('').str.contains(teacher_subject, regex=False)
        ][['资源ID', '综合评分']].sort_values(by='综合评分', ascending=False)['资源ID'].tolist()
    elif teacher_segment == r'\N' and teacher_subject != r'\N':
        # 如果老师的学段是 \N，则推荐所有学科匹配的资源
        matching_resources = score[
            score['适用学科'].fillna('').str.contains(teacher_subject, regex=False)
        ][['资源ID', '综合评分']].sort_values(by='综合评分', ascending=False)['资源ID'].tolist()
    elif teacher_segment != r'\N' and teacher_subject == r'\N':
        # 如果老师的学科是 \N，则推荐所有学段匹配的资源
        matching_resources = score[
            score['适用学段'].fillna('').str.contains(teacher_segment, regex=False)
        ][['资源ID', '综合评分']].sort_values(by='综合评分', ascending=False)['资源ID'].tolist()
    else:
        # 如果学科和学段均为 \N，则推荐空列表
        matching_resources = []
    # 将资源序号列表存储到字典中对应的老师ID下
    teacher_resources[teacher_id] = {'用户id': teacher_user_id, '资源序号列表': matching_resources}

teacher_resources_df = pd.DataFrame.from_dict(teacher_resources, orient='index')
teacher_resources_df.to_csv('teacher_resources.csv', encoding='utf-8-sig', index=False)