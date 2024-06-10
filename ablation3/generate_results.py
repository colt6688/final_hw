import os

os.environ["OPENAI_API_KEY"] = "sb-ed07016f987c6bb701b74fcf399c56d067b5a5c8bc3ad177"

import json
import logging
import pathlib
import argparse
from typing import List, Dict, Any
# from colorama import Fore
from copy import deepcopy
from glob import glob
import agents as agents
from tqdm import tqdm

import time

icl_examples = [
    {
        "role": "user",
        "content": "你是中文词义理解的专家，现在有如下词义理解任务，对于给定上下文，请回答上下文中目标词的含义上下文:你看嘛，既#轻巧#又美观，几十年都用不坏。请回答上下文中\"轻巧\"一词的含义，仅限20个token:",
    },
    {
        "role": "assistant",
        "content": "轻巧指物品重量轻，体积小，便于携带。"
    },
    {
        "role": "user",
        "content": "你是中文词义理解的专家，现在有如下词义理解任务，对于给定上下文，请回答上下文中目标词的含义上下文:1406整治1407年朝鲜太宗开始进行主流河床拓寛、两岸筑坝等工程；1411年设立临时整治机构「开渠都监」；1412年1月15日整治2月15日投入5万2800人进行大规模#整治#。请回答上下文中\"整治\"一词的含义，仅限20个token:",
    },
    {
        "role": "assistant",
        "content": "修理。"
    },
    {
        "role": "user",
        "content": "你是中文词义理解的专家，现在有如下词义理解任务，对于给定上下文，请回答上下文中目标词的含义上下文:缝纫机#针#。请回答上下文中\"针\"一词的含义，仅限20个token:",
    },
    {
        "role": "assistant",
        "content": "缝衣物用的工具，细长而小，一头尖锐，一头有孔或钩，可以引线，多用金属制成。"
    }
]

def template_change(conversation):
    messages = []
    for item in conversation:
        message = {}
        if item['from'] == "gpt":
            message['role'] = "assistant"
            message['content'] = item['value']
        else:
            message['role'] = "user"
            message['content'] = item['value']
        messages.append(message)
    return messages


def construct_llm_data(args: argparse.Namespace):
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)
    
    if args.model_name is not None:
        agent_config['config']['model_name'] = args.model_name

    # initialize the agent
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
        agent_config["config"]
    )

    data_path = f"{args.split}.json"
    all_data = json.load(open(data_path))
    
    part_len = len(all_data) // args.part_num + 1
    to_be_sample_data = all_data[args.part_idx * part_len: min((args.part_idx + 1) * part_len, len(all_data))]
    n_tasks = len(to_be_sample_data)
    pbar = tqdm(total=n_tasks)
    
    final_data = []
    
    for data in to_be_sample_data:
        id = data['id']
        words = data['word']
        context = data['context']

        input_context = f"你是中文词义理解的专家，现在有如下词义理解任务，对于给定上下文，请回答上下文中目标词的含义上下文:{context}。请回答上下文中\"{words}\"一词的含义，仅限20个token:"
        messages = deepcopy(icl_examples)
        messages.append({"role": "user", "content": input_context})
        llm_output: str = agent(messages)
            
        new_item = {
            "id": id,
            "context": context,
            "word": words,
            "llm_output": llm_output,
            "model_name": args.model_name
        }
            
        final_data.append(new_item)
        pbar.update(1)
    pbar.close()
            
    json.dump(final_data, open(f"{args.save_path}/output_{args.part_idx}.json", "w"), indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--agent_path",
        type=str,
        default="configs",
        help="Config path of model.",
    )
    parser.add_argument(
        "--agent_config",
        type=str,
        default="fastchat",
        help="Config of model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="chatglm2-6b-0",
        help="Model name. It will override the 'model_name' in agent_config"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validate",
    )
    parser.add_argument(
        "--part_num",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--part_idx",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--save_path",
        type=str,
    )
    args = parser.parse_args()
    construct_llm_data(args)