from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq,Seq2SeqTrainingArguments,\
                        Seq2SeqTrainer,EarlyStoppingCallback
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM
from tqdm import tqdm
import pickle
import torch
"""
用于第一种义项评分方法GSP的预测结果统计
"""

#自定义数据集类
class LCSTS(Dataset):
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

#批处理数据
def collote_fn(batch_samples):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample['context'])
        batch_targets.append(sample['gloss'])

    batch_data = tokenizer(
        batch_inputs,
        padding=True,
        max_length=128,
        truncation=True,
        return_tensors="pt"
    )
    del batch_data['token_type_ids']

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_targets,
            padding=True,
            max_length=64,
            truncation=True,
            return_tensors="pt"
        )["input_ids"]
        batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
        end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
        for idx, end_idx in enumerate(end_token_index):
            labels[idx][end_idx+1:] = -100
        batch_data['labels'] = labels
    return batch_data


if __name__ == '__main__':
    test_data = LCSTS("./data/few_shot.json")
    print(test_data[0])

    model_path = "./data/bart-large-chinese"

    device = 'cuda:0'

    #加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    with open('./data/word_inventory.pkl', 'rb') as f:
        word_sense_data = pickle.load(f)

    max_input_length = 128
    max_target_length = 64

    pos_sum_dict = {'名词': 0, '形容词': 0, '动词': 0, '副词': 0, '全集': 0}#用于记录各词类数据总数。
    pos_crt_dict = {'名词': 0, '形容词': 0, '动词': 0, '副词': 0, '全集': 0}#用于记录各词类正确预测数，

    for d in tqdm(test_data):
        tmp_data_list = [d]
        word = d['gloss'].split('：')[0]
        target = d['gloss']
        target_pos = ''
        for pos, g in word_sense_data[word]:
            if word + '：' + g != target:
                tmp_data_list.append({'context': d['context'], 'gloss': word + '：' + g, 'idx': d['idx']})
            else:
                target_pos = pos

        test_dataloader = DataLoader(tmp_data_list, batch_size=len(tmp_data_list), shuffle=False, collate_fn=collote_fn)
        for batch, batch_data in enumerate(test_dataloader, start=1):
            batch_data = batch_data.to(device)
            outputs = model(**batch_data)
            index = batch_data['labels'].unsqueeze(-1)

            # 计算每个释义序列的概率
            inputs = torch.softmax(outputs["logits"], dim=2)
            prob = torch.gather(inputs, 2, index, out=None)
            index = index.squeeze(-1)
            prob = prob.squeeze(-1)

            output_sum = torch.sum(prob.masked_fill(index == 0, 0) > 0, dim=-1)

            output_prob = torch.sum(torch.log(prob).masked_fill(index == 0, 0), dim=-1)

            output_prob /= output_sum

            index = torch.argmax(output_prob).data#义项评分最高的为预测答案

            pos_sum_dict['全集'] += 1
            if pos in pos_sum_dict:
                pos_sum_dict[pos] += 1
            if index == 0:#第0个位置为标准答案，预测为0即为预测正确
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

    with open('./output/result_few_shot_greedy.txt', 'w') as f:
        f.write(str(result))




