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

# 加载预训练的中文 BERT 模型和 tokenizer
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 例子
sentence1 = "这是一个示例句子。"
sentence2 = "这是另一个示例句子。"

similarity = get_bert_similarity(sentence1, sentence2, model, tokenizer)
print(f"语义相似度: {similarity}")