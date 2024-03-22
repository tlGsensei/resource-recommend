import pandas as pd
import numpy as np
from datetime import datetime
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm

# 加载预训练的中文 BERT 模型和 tokenizer
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

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

def calculate_similarity_and_update_score(row, resource_new):
    id0_title = row['资源标题'].values[0]
    id0_embedding = get_bert_embeddings(id0_title, model, tokenizer)

    # 如果resource_new的长度超过100，则随机选择100个条目
    if len(resource_new) > 200:
        resource_new = resource_new.sample(n=200)

    similarities = []
    for index, resource_row in tqdm(resource_new.iterrows(), total=len(resource_new)):
        title = resource_row['资源标题']
        similarity = get_bert_similarity(id0_title, title, model, tokenizer)
        similarities.append(similarity)

    resource_new['相似度'] = similarities
    # 加权融合相似度和综合评分，按照3:7的比例
    resource_new['新综合评分'] = 0.3 * resource_new['综合评分'] + 0.7 * resource_new['相似度']
    resource_new = resource_new.sort_values(by='新综合评分', ascending=False)

    return resource_new

resource = pd.read_csv('score.csv')

# 当前资源ID
id0 = "51442125b3d64252bd5920c6f42b3361"

# 查找与当前资源ID匹配的学段和学科
row = resource.loc[resource['资源ID'] == id0]
if not row.empty:
    suitable_stage = row['适用学段'].values[0]
    suitable_subject = row['适用学科'].values[0]

    # 排除与输入ID相同的情况，同时筛选其他条件
    resource_new = resource[(resource['资源ID'] != id0) &
                            (resource['适用学段'].str.contains(suitable_stage)) &
                            ((resource['适用学科'].str.contains(suitable_subject)))]

    # 计算相似度并融合到"综合评分"中
    resource_new = calculate_similarity_and_update_score(row, resource_new)

    # 按照新的"综合评分"列由高到低重新排序resource_new
    resource_new = resource_new.sort_values(by='新综合评分', ascending=False)

    top_10 = resource_new.head(10)

    # 从resource_new中随机选择5个资源，要求不与top_10中的资源重复
    random_resources = resource_new[~resource_new['资源ID'].isin(top_10['资源ID'])].sample(n = 5)

    result = pd.concat([top_10, random_resources])

    print(top_10['资源ID'])

else:
    print("Error")

pass