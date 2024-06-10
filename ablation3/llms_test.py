from torch.utils.data import Dataset
import json
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
"""
评估大模型采用生成式方法时在词义消歧任务上的性能
"""
def get_sim_score(model,q,d):
    """
    用于计算文本对相似度
    输入：句嵌入模型、文本对
    输出：文本对之间的相似度
    """
    q_embedding = model.encode([q], normalize_embeddings=True)
    d_embedding = model.encode([d], normalize_embeddings=True)
    score = q_embedding @ d_embedding.T
    return score[0][0]


def load_data(path):
    """
    用于读取json文件
    输入：文件路径
    输出：文件内容
    """
    with open(path, 'r', encoding='utf-8') as load_f:
        data = json.load(load_f)
    return data

if __name__ == '__main__':

    device = 'cuda'
    sim_model = SentenceTransformer('C:\研究生工作\模型备份/bge-zh-large-v1.5')#记载句嵌入模型
    sim_model.to(device)

    data = load_data('./results/Yi-1.5-6B-Chat_test.json')#加载大模型训练结果

    pred_data_dict = {}#获取每个词的义项集合
    for d in data:
        pred_data_dict[d['id']] = d
    print(pred_data_dict[0])

    true_data = load_data('./data/test_data.json')
    with open('./data/word_inventory.pkl', 'rb') as f:
        word_sense_data = pickle.load(f)
    pos_sum_dict = {'名词': 0, '形容词': 0, '动词': 0, '副词': 0, '全集': 0}#用于记录各词类数据总数。
    pos_crt_dict = {'名词': 0, '形容词': 0, '动词': 0, '副词': 0, '全集': 0}#用于记录各词类正确预测数，

    for d in tqdm(true_data):
        word = d['gloss'].split('：')[0]
        target_gloss = d['gloss']
        id = d['idx']
        target_pos = ''
        context = d['context']
        cdt_gloss_list = [target_gloss]
        for pos, g in word_sense_data[word]:
            if word + '：' + g != target_gloss:
                cdt_gloss_list.append(word + '：' + g)
            else:
                target_pos = pos
        if int(id) in pred_data_dict:
            pred_gloss = pred_data_dict[int(id)]['llm_output']
            value_list = []
            for cdt_gloss in cdt_gloss_list:
                tmp_sum_sim_score = (1 - get_sim_score(sim_model, cdt_gloss, pred_gloss))#计算释义相似度评分
                value_list.append(tmp_sum_sim_score)

            min_value = min(value_list)  # 求列表最小值
            min_idx = value_list.index(min_value)  # 求最小值对应索引
            pos_sum_dict['全集'] += 1
            if pos in pos_sum_dict:
                pos_sum_dict[pos] += 1
            if min_idx == 0:#第0个位置为标准答案，预测为0即为预测正确
                pos_crt_dict['全集'] += 1
                if pos in pos_sum_dict:
                    pos_crt_dict[pos] += 1
                else:
                    pass
    result = {}
    for key in pos_sum_dict:
        if pos_sum_dict[key] == 0:
            continue
        result[key] = pos_crt_dict[key] / pos_sum_dict[key]#计算准确率

    with open('./output/result_yi_6b.txt', 'w') as f:
        f.write(str(result))


