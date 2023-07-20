import os
import json
import argparse
from tqdm import tqdm
from ..evaluate.evaluate import (
    math_evaluate,
    lama_evaluate,
    qa_evaluate,
    mlqa_evaluate
)

task_tool = {
    "Math": "calculator",
    "LAMA": "QA",
    "QA": "WikiSearch",
    "MLQA": "translator"
}

score_list = [20, 15, 10, 5, 0]


def get_score(task, gold, gold_tool_use, pred_list, tool_use_list):
    scores = [0, 0, 0]
    if task == "Math":
        gold = round(float(gold.replace(",", "")), 2)
        wrong_dict = {}
        score_start = 1
        for i, (pred, tool_use) in enumerate(zip(pred_list, tool_use_list)):
            pred = round(float(pred.replace(",", "")), 2)
            if pred == gold:
                if gold_tool_use ^ tool_use:
                    scores[i] = score_list[1]
                    score_start = 2
                else:
                    scores[i] = score_list[0]
            else:
                wrong_dict[i] = pred
        if len(wrong_dict) == 0:
            return scores
        if len(wrong_dict) == 1:
            key = list(wrong_dict.keys())[0]
            scores[key] = score_list[score_start]
            return scores
        if len(wrong_dict) >= 2:
            wrong_dict = sorted(wrong_dict.items(), key=lambda x: abs(x[1] - gold))
            for key, _ in wrong_dict:
                scores[key] = score_list[score_start]
                score_start += 1
            return scores
    else:
        wrong_dict = {}
        score_start = 1
        for i, (pred, tool_use) in enumerate(zip(pred_list, tool_use_list)):
            if pred is None:
                scores[i] = 0
                continue
            if isinstance(pred, str):
                pred = [pred]
            flag = True
            for p in pred:
                if gold.lower() in p.lower():
                    if gold_tool_use ^ tool_use:
                        scores[i] = score_list[1]
                        score_start = 2
                    else:
                        scores[i] = score_list[0]
                    flag = False
                    break
            if flag:
                wrong_dict[i] = pred[0]
        if len(wrong_dict) == 0:
            return scores
        else:
            for key in wrong_dict:
                scores[key] = score_list[score_start]
            return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Math")
    parser.add_argument("--model", type=str, default="vicuna")
    parser.add_argument("--lora_weights", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--target_path", type=str, default="")
    args = parser.parse_args()

    eval_dict = {
        "Math": math_evaluate,
        "LAMA": lama_evaluate,
        "QA": qa_evaluate,
        "MLQA": mlqa_evaluate
    }
    models = ["chatgpt", "davinci", "vicuna", "alpaca"]

    print("Task:", args.task)

    model_data_dict = {}
    # for model in models:
    #     data_path = os.path.join(args.data_path, args.task, f"{args.task.lower()}_{model}_train.json")
    #     f = open(data_path)
    #     lines = f.readlines()
    #     data_response = [json.loads(line) for line in lines]
    #     _, _, outputs = eval_dict[args.task](args, data_response)
    #     model_data_dict[model] = outputs
    #     f.close()

    for model in models:
        data_path = os.path.join(args.data_path, f"{args.task}_{model}_response.json")
        f = open(data_path)
        lines = f.readlines()[1:]
        data_response = [json.loads(line) for line in lines]
        model_data_dict[model] = data_response
        f.close()

    id_data_dict = {}
    with open(os.path.join(args.data_path, f"{args.task}_gold_response.json"), "r") as f:     
        gold_datas = f.readlines()
        for gold_data in gold_datas:
            gold_data = json.loads(gold_data)
            id = gold_data["id"]
            id_data_dict[id] = {
                "query": gold_data["instruction"] + "\n" + gold_data["input"],
                "responses": [gold_data["output"]],
                "origin": ["gold"],
                "scores": [],
                "answer": [gold_data["answer"]]
            }

    for model, datas in model_data_dict.items():
        for data in datas:
            id = data["id"]
            if id in id_data_dict:
                id_data_dict[id]["responses"].append(data["response"])
                id_data_dict[id]["origin"].append(model)
                pred = data["pred"]
                if isinstance(pred, list) and len(pred) > 0:
                    pred = pred[0]
                if pred is None:
                    pred = ""
                try:
                    pred = str(pred)
                except:
                    pred = ""
                id_data_dict[id]["answer"].append(pred)

    data_list = []
    for key, value in id_data_dict.items():
        if len(value["responses"]) < 5:
            continue
        data_list.append(value)

    # score
    for data in tqdm(data_list):
        data["scores"].append(score_list[0])  # gold赋最高分
        if task_tool[args.task] not in data["responses"][0]:
            data["scores"].append(score_list[1])
        else:
            data["scores"].append(score_list[0])
        pred_list = [data["answer"][2], data["answer"][3], data["answer"][4]]
        tool_use_list = [
            task_tool[args.task] in data["responses"][2],
            task_tool[args.task] in data["responses"][3],
            task_tool[args.task] in data["responses"][4]
        ]
        gold_answer = data["answer"][0]
        gold_tool_use = task_tool[args.task] in data["responses"][0]
        scores = get_score(
            args.task,
            gold_answer,
            gold_tool_use,
            pred_list,
            tool_use_list
        )
        for s in scores:
            data["scores"].append(s)

    # write
    if not os.path.exists(args.target_path):
        os.mkdir(args.target_path)
    target_path = os.path.join(args.target_path)
    with open(target_path, "w") as f:
        for data in data_list:
            try:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            except:
                continue
