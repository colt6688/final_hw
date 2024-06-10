import json
import random,copy,os,re
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq,Seq2SeqTrainingArguments,\
                        Seq2SeqTrainer,EarlyStoppingCallback
from datasets import Dataset,DatasetDict

from utils import *

if __name__ == '__main__':
    train_dataset = Dataset.from_list(load_data('./data/train_data.json'))
    dev_dataset = Dataset.from_list(load_data('./data/dev_data.json'))
    print(len(train_dataset), train_dataset[0])
    print(len(dev_dataset), dev_dataset[0])

    model_path = "./data/bart-large-chinese"
    output_path = "output/bart_gloss_generation"
    source_max_length = 128
    target_max_length = 64  # 这2个都是模型限长
    device = 'cuda:0'
    #加载模型、分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    #数据批量处理函数
    def preprocess_function(examples):
        model_inputs = tokenizer(examples['context'], max_length=source_max_length, truncation=True)
        labels = tokenizer(examples["gloss"], max_length=target_max_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    trian_tokenized_dataset = train_dataset.map(preprocess_function, batched=True)
    dev_tokenized_dataset = dev_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model)
    #训练参数设置
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_path,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=6,
        weight_decay=0.01,
        save_steps=1,
        save_total_limit=3,
        num_train_epochs=100,
        predict_with_generate=True,
        fp16=True,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True
    )
    #trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=trian_tokenized_dataset,
        eval_dataset=dev_tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(10)],
    )
    #模型训练，保存valid的loss最低的模型
    trainer.train()
    trainer.save_model()

