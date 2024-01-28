import pandas as pd
import numpy as np
from datetime import datetime
from transformers import BertTokenizer, BertModel
import torch

def get_bert_embeddings(sentence, model, tokenizer):
    # 使用 BERT tokenizer 对句子进行标记
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sentence)))

    # 将标记转换为模型输入的格式
    inputs = tokenizer.encode_plus(sentence, return_tensors="pt", add_special_tokens=True)

    # 获取 BERT 模型的输出
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state

    # 获取句子的平均嵌入表示
    sentence_embedding = torch.mean(last_hidden_states, dim=1).squeeze()

    return sentence_embedding

def get_bert_similarity(sentence1, sentence2, model, tokenizer):
    # 获取两个句子的嵌入表示
    embedding1 = get_bert_embeddings(sentence1, model, tokenizer)
    embedding2 = get_bert_embeddings(sentence2, model, tokenizer)

    # 计算余弦相似度
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)

    return similarity.item()

resource = pd.read_csv('resource.csv')
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

x1 = '高中'
x2 = '生物'
x3 = '北师大版'
x4 = '一年级上'
topic = 'DNA的复制'

if x1 != r'\N' and x2 != r'\N':
    # 在资源表中筛选出符合老师学段和学科的资源序号列表，并排序
    matching_resources = score[
        score['适用学段'].fillna('').str.contains(x1, regex=False) &
        score['适用学科'].fillna('').str.contains(x2, regex=False)
        ][['资源ID', '资源标题', '综合评分']].sort_values(by='综合评分', ascending=False)[['资源ID', '资源标题', '综合评分']].values.tolist()
elif x1 == r'\N' and x2 != r'\N':
    # 如果老师的学段是 \N，则推荐所有学科匹配的资源
    matching_resources = score[
        score['适用学科'].fillna('').str.contains(x2, regex=False)
    ][['资源ID', '资源标题', '综合评分']].sort_values(by='综合评分', ascending=False)[['资源ID', '资源标题', '综合评分']].values.tolist()
elif x1 != r'\N' and x2 == r'\N':
    # 如果老师的学科是 \N，则推荐所有学段匹配的资源
    matching_resources = score[
        score['适用学段'].fillna('').str.contains(x1, regex=False)
    ][['资源ID', '资源标题', '综合评分']].sort_values(by='综合评分', ascending=False)[['资源ID', '资源标题', '综合评分']].values.tolist()
else:
    # 如果学科和学段均为 \N，则推荐空列表
    matching_resources = []

# print(matching_resources)

# 加载预训练的中文 BERT 模型和 tokenizer
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

for resource in matching_resources:
    resource_title = resource[1]  # 假设第二列是资源标题
    similarity = get_bert_similarity(resource_title, topic, model, tokenizer)
    resource.append(similarity)

# 首先，根据相似度（假设为每个子列表的第四个元素）选取前30个最高相似度的行
top_30_similar = sorted(matching_resources, key=lambda x: x[3], reverse=True)[:30]

# 然后，在这30个行中根据第三列（假设为综合评分）进行降序排序
sorted_top_30 = sorted(top_30_similar, key=lambda x: x[2], reverse=True)

print(sorted_top_30)
pass

