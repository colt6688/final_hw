import argparse
from glob import glob
import json

def gather_results(args: argparse.Namespace):
    final_results = []
    for file in glob(f"{args.data_path}/*.json"):
        with open(file) as f:
            data = json.load(f)
            final_results.extend(data)
    json.dump(final_results, open(f"results/{args.model_name}_{args.split}.json", "w"), indent=4, ensure_ascii=False)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="validate")
    parser.add_argument("--model_name", type=str, default="Qwen-14B-Chat")
    parser.add_argument("--data_path", type=str, default="/home/azureuser/weimin/homework/outputs/Qwen-14B-Chat/output/validate")
    args = parser.parse_args()
    gather_results(args)