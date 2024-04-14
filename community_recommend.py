import pandas as pd
from datetime import datetime
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm

teacher = pd.read_excel('teacher.xlsx')
community = pd.read_excel('community.xlsx')

# 输入用户的学段 学科 省 市 区 学校，以“魏静园”为例
x0 = "魏静园"
x1 = "小学"
x2 = "语文"
x3 = "北京"
x4 = "北京市"
x5 = "海淀区"
x6 = "北京师范大学跨越式课题组"

community['probability'] = 0 # 初始化推荐概率列

# 计算省市区学校一致性
community['省市区学校一致性'] = 0
community['地区一致性分数'] = 0.0
# 找省相同的老师，赋予权重值为1
province_match_index = community[community['省'] == x3].index
community.loc[province_match_index, '省市区学校一致性'] = 1
# 找省市相同的老师，赋予权重值为2
province_city_match_index = community[(community['省'] == x3) & (community['市'] == x4)].index
community.loc[province_city_match_index, '省市区学校一致性'] = 2
# 找省市区均相同的老师，赋予权重值为3
province_city_district_match_index = community[(community['省'] == x3) & (community['市'] == x4) & (community['区'] == x5)].index
community.loc[province_city_district_match_index, '省市区学校一致性'] = 3
# 找学校相同的老师，赋予最高权重值4
school_match_index = community[community['学校'] == x6].index
community.loc[school_match_index, '省市区学校一致性'] = 4
# 归一化
max_score = community['省市区学校一致性'].max()
min_score = community['省市区学校一致性'].min()
community.loc[community['省市区学校一致性'] != 0, '地区一致性分数'] = (community['省市区学校一致性'] - min_score) / (max_score - min_score)

# 计算人数分数
# 设置基础评分和对数转换的底数
base_score = 1
log_base = 2  # 以2为底的对数转换
# 对人数不为零的社区进行对数转换和归一化
non_zero_views = community[community['加入人数'] != 0]['加入人数']
community.loc[community['加入人数'] != 0, '人数分数'] = base_score + np.log(non_zero_views) / np.log(log_base)
max_score = community['人数分数'].max()
min_score = community['人数分数'].min()
community.loc[community['加入人数'] != 0, '人数分数'] = (community['人数分数'] - min_score) / (max_score - min_score)
# 对加入人数为零的资源直接将评分置为0
community.loc[community['加入人数'] == 0, '人数分数'] = 0

# 计算时间分数
# 设置基础评分和对数转换的底数
base_score = 12
log_base = 2  # 以2为底的对数转换
# 假设创建时间格式为字符串，需要转换为 datetime 类型
community['创建时间'] = pd.to_datetime(community['创建时间'])
# 计算当前时间
current_time = datetime.now()
# 计算资源存在的时间（以天为单位）
community['存在时间'] = (current_time - community['创建时间']).dt.days
max_time = community['存在时间'].max()
min_time = community['存在时间'].min()
community['时间分数'] = base_score - np.log(community['存在时间']) / np.log(log_base)
max_score = community['时间分数'].max()
min_score = community['时间分数'].min()
community['时间分数'] = (community['时间分数'] - min_score) / (max_score - min_score)

community['总分数'] = community['地区一致性分数'] + community['时间分数'] + community['人数分数']
community.sort_values(by='总分数', ascending=False, inplace=True)

top_50 = community.head(50)

# 计算学科匹配分数
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


top_50['主题相关度'] = 0.0
for k in range(50):
    top_50['主题相关度'].values[k] = get_bert_similarity(top_50['社区的名称'].values[k], x2, model, tokenizer)

top_50.sort_values(by='主题相关度', ascending=False, inplace=True)

top_10 = top_50.head(10)
random_2 = top_50[~top_50['主题相关度'].isin(top_10['主题相关度'])].sample(n=2)
result = pd.concat([top_10, random_2])
