import pandas as pd
import numpy as np
import torch
from datetime import datetime
from transformers import BertTokenizer, BertModel
import re
# 输入：资源指标，学科，年段，资源名称，教材版本

resource_new = pd.read_excel('updated_resource_new.xlsx')

# x0 = "课堂导入"
# x1 = "语文"
# x2 = "小学"   # 输入1-12或小学/初中/高中/通用/其他
# x3 = "通用"
# x4 = "学科教学导入四法"

x0 = "课堂导入"
x1 = "数学"
x2 = "初中"   # 输入1-12或小学/初中/高中/通用/其他
x3 = "通用"
x4 = "巧借现代信息技术构建数学高效课堂"

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

def match_year_group(input_year):
    # 将输入的年段映射为对应的匹配规则
    mapping = {
        "1": ["1", "小学", "小学初中", "通用"],
        "2": ["2", "小学", "小学初中", "通用"],
        "3": ["3", "小学", "小学初中", "通用"],
        "4": ["4", "小学", "小学初中", "通用"],
        "5": ["5", "小学", "小学初中", "通用"],
        "6": ["6", "小学", "小学初中", "通用"],
        "7": ["7", "初中", "小学初中", "通用"],
        "8": ["8", "初中", "小学初中", "通用"],
        "9": ["9", "初中", "小学初中", "通用"],
        "10": ["10", "高中", "通用"],
        "11": ["11", "高中", "通用"],
        "12": ["12", "高中", "通用"],
        "小学": ["小学", "小学初中", "通用", "1", "2", "3", "4", "5", "6"],
        "初中": ["初中", "小学初中", "通用", "7", "8", "9"],
        "高中": ["高中", "通用", "10", "11", "12"],
        "通用": ["通用"],
    }

    def apply_matching(row):
        resource_year_group = row["所属年段\n（如有具体年级，请写年级）"]
        matched_groups = []
        if input_year in mapping:
            matched_groups = mapping[input_year]

        return resource_year_group in matched_groups

    # 在DataFrame上应用匹配规则函数
    resource_filtered["匹配结果"] = resource_filtered.apply(apply_matching, axis=1)

unique_segment_values = resource_new['所属年段\n（如有具体年级，请写年级）'].unique().tolist()

resource_filtered = resource_new[resource_new['资源指标'] == x0]
# resource_filtered = resource_filtered[(resource_new['所属学科'] == x1) & (resource_new['所属年段\n（如有具体年级，请写年级）'] == x2)]
resource_filtered = resource_filtered[(resource_new['所属学科'] == x1)]
# 输入的所属年段为(1,2,3,4,5,6,7,8,9,10,11,12,小学,初中,高中,通用),我希望将输入与resource_new的所属年段列进行匹配，匹配规则是1-6数字可以匹
# 配对应数字或小学或小学初中或通用，7-9数字可以匹配对应数字或初中或小学初中或通用，10-12数字可以匹配对应数字或高中或通用，小学只能匹配小学或“小学初中”，初中只能匹配初中或“小学初中”，高中只能匹配高中，通用匹配通用

match_year_group(x2)
# resource_filtered = resource_filtered[(resource_filtered['匹配结果'] == True)]
resource_filtered = resource_filtered[(resource_filtered['匹配结果'] == True) & (resource_filtered['资源名称'] != x4)]


# 加载预训练的中文 BERT 模型和 tokenizer
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
resource_filtered_list = resource_filtered.values.tolist()
resource_filtered_list = [row[:7] for row in resource_filtered_list]
for resource in resource_filtered_list:
    resource_title = resource[2]  # 假设第二列是资源标题
    similarity = get_bert_similarity(resource_title, x4, model, tokenizer)
    resource.append(similarity)

# 首先，根据相似度（假设为每个子列表的第四个元素）选取前30个最高相似度的行
top_30_similar = sorted(resource_filtered_list, key=lambda x: x[7], reverse=True)[:30]

# 使用 sorted 函数进行排序
sorted_list = sorted(top_30_similar, key=lambda row: 0 if row[6] == x3 else 1)
final_res = []
indicators = [row for row in sorted_list if row[0] == "指标描述"][:4]
if len(indicators) > 0:
    final_res.append(indicators)
indicators = [row for row in sorted_list if row[0] == "常见问题"][:2]
if len(indicators) > 0:
    final_res.append(indicators)
indicators = [row for row in sorted_list if row[0] == "主题讲座"][:2]
if len(indicators) > 0:
    final_res.append(indicators)
indicators = [row for row in sorted_list if row[0] == "教学案例"][:2]
if len(indicators) > 0:
    final_res.append(indicators)
indicators = [row for row in sorted_list if row[0] == "拓展延伸"][:1]
if len(indicators) > 0:
    final_res.append(indicators)

print(final_res)
for row in final_res:
    print(row)
pass
