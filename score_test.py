from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
"""
用于义项评分方法GSS、GSSWP、MBRR的预测结果统计
"""
def get_sim_score(model,q,d):
    """
        用于计算文本对相似度
        输入：句嵌入模型、文本对
        输出：文本对之间的相似度
        """
    q_embedding = model.encode(q, normalize_embeddings=True)
    d_embedding = model.encode(d, normalize_embeddings=True)
    score = q_embedding @ d_embedding.T
    return score
class LCSTS(Dataset):#自定义数据集类
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, path):
        with open(path, 'r', encoding='utf-8') as load_f:
            data = json.load(load_f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    test_data = LCSTS("./data/zero_shot.json")
    print(test_data[0])

    model_path = "./output/bart_gloss_generation/best"
    # 加载模型和分词器
    device = 'cuda:0'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    sim_model = SentenceTransformer('C:\研究生工作\模型备份/bge-zh-large-v1.5')
    sim_model.to(device)

    with open('./data/word_inventory.pkl', 'rb') as f:
        word_sense_data = pickle.load(f)

    max_input_length = 128
    max_target_length = 64

    crt = 0
    pos_sum_dict = {'名词': 0, '形容词': 0, '动词': 0, '副词': 0, '全集': 0}#用于记录各词类数据总数。
    pos_crt_dict = {'名词': 0, '形容词': 0, '动词': 0, '副词': 0, '全集': 0}#用于记录各词类正确预测数，
    score_method='MBRR'#['GSS','GSSWP','MBRR']#选择义项评分方式

    for d in tqdm(test_data):

        word = d['gloss'].split('：')[0]
        target_gloss = d['gloss']
        target_pos = ''
        context = d['context']
        cdt_gloss_list = [target_gloss]
        for pos, g in word_sense_data[word]:
            if word + '：' + g != target_gloss:
                cdt_gloss_list.append(word + '：' + g)
            else:
                target_pos = pos


        num_beams=1
        model.config.num_return_sequences =3
        outputs = model.generate(tokenizer.encode(context, return_tensors="pt", max_length=max_input_length).to(device),
                                 max_new_tokens=max_target_length,do_sample=False,num_beams=3,output_scores=True, return_dict_in_generate=True)
        pred_gloss_list=[]
        for s,seq_score in zip(outputs.sequences,outputs.sequences_scores):
            output_text = tokenizer.decode(s, skip_special_tokens=True).replace(" ", "")
            pred_gloss_list.append([output_text,torch.exp(seq_score).data])

        value_list=[]
        if score_method=='MBRR':
            pred_glosses=[pg[0] for pg in pred_gloss_list]
            pred_scores=torch.tensor([pg[1] for pg in pred_gloss_list])
            gloss_MBRR=torch.sum((1-torch.tensor(get_sim_score(sim_model, pred_glosses, pred_glosses)))*pred_scores,dim=1)
            index=torch.argmin(gloss_MBRR).data
            value_list=list(get_sim_score(sim_model, [pred_glosses[index]], cdt_gloss_list)[0])

        else:
            for cdt_gloss in cdt_gloss_list:
                tmp_sum_sim_score=0
                for pred_gloss,seq_score in pred_gloss_list:
                    if score_method=='GSS':
                        tmp_sum_sim_score+=(1-get_sim_score(sim_model,[cdt_gloss],[pred_gloss])[0][0])*seq_score
                    elif score_method=='GSSWP':
                        tmp_sum_sim_score+=(1-get_sim_score(sim_model,[cdt_gloss],[pred_gloss])[0][0])
                    else:
                        raise Exception("Please choose correct score method.")

                value_list.append(tmp_sum_sim_score)

        min_value = min(value_list)  # 求列表最小值
        min_idx = value_list.index(min_value)  # 求最小值对应索引

        pos_sum_dict['全集'] += 1
        if pos in pos_sum_dict:
            pos_sum_dict[pos] += 1
        if min_idx==0:
            pos_crt_dict['全集'] += 1
            if pos in pos_sum_dict:
                pos_crt_dict[pos] += 1
            else:
                pass
    result={}
    for key in pos_sum_dict:
        if pos_sum_dict[key]==0:
            continue
        result[key]=pos_crt_dict[key]/pos_sum_dict[key]

    with open('./output/result_ceshi.txt','w') as f:
        f.write(str(result))






