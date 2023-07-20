import os
import re
import json
import torch
import argparse
import requests
import os.path as osp
from typing import Union
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from peft import PeftModel
from src.index import DistributedIndex
from src.fid import FiD
from src.retrievers import (
    Contriever,
    DualEncoderRetriever,
    UntiedDualEncoderRetriever
)
from src.atlas import Atlas
from src.model_io import _load_atlas_model_state
from src.util import get_unwrapped_model_if_wrapped


device = "cuda"


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("./templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


def calculator(input_query):
    return eval(input_query)


class ColBERTv2:
    def __init__(self, url: str):
        self.url = url

    def __call__(self, query, k=10):
        topk = colbertv2_get_request(self.url, query, k)
        topk = [doc['text'] for doc in topk]
        return topk

def colbertv2_get_request(url: str, query: str, k: int):
    payload = {'query': query, 'k': k}
    res = requests.get(url, params=payload)
    topk = res.json()['topk'][:k]
    return topk

def WikiSearch(input_query, k=1):
    url = ""
    retrieval_model = ColBERTv2(url)
    output = retrieval_model(input_query, k)
    return output


def translator(model, tokenizer, input_query):
    input_ids = tokenizer(input_query, return_tensors='pt')
    outputs = model.generate(
        **input_ids,
        forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"]
    )
    output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return output


def QA(model, index, opt, input_query):
    query_enc = model.retriever_tokenizer(input_query)
    query_ids_retriever = query_enc["input_ids"].cuda()
    query_mask_retriever = query_enc["attention_mask"].cuda()
    retrieved_passages, _ = model.retrieve(
        index,
        opt.n_context,
        input_query,
        query_ids_retriever,
        query_mask_retriever
    )

    reader_tokens, _ = model.tokenize_passages(input_query, retrieved_passages)

    generation = model.generate(reader_tokens, input_query)
    result = model.reader_tokenizer.decode(generation[0], skip_special_tokens=True)
    return result


def load_llama_model(model, lora_weights):
    if model == "alpaca":
        base_model = "PLMs/alpaca-7b"
    elif model == "vicuna":
        base_model = "PLMs/vicuna-7b"
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    print("pre-trained model's BOS EOS and PAD token id:",bos,eos,pad," => It should be 1,2,none")

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16
    )

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = tokenizer.bos_token_id = 1
    model.config.eos_token_id = tokenizer.eos_token_id = 2

    model.half()
    model.eval()

    return model, tokenizer


def math_evaluate(args, data_responses):
    dataset_right = {
        "ASDiv": 0,
        "SVAMP": 0,
        "GSM8K": 0,
        "total": 0
    }

    dataset_total = {
        "ASDiv": 0,
        "SVAMP": 0,
        "GSM8K": 0
    }

    datas = []
    use_tool = 0
    right = 0
    for data in tqdm(data_responses):
        response = data["response"]
        if "calculator" in response:
            use_tool += 1
            pattern = re.compile(r"calculator\(.*\)")
            pred = pattern.findall(response)
            pred = pred[0] if len(pred) > 0 else None
            query = pred[len("calculator")+1:-1].replace(",", "").replace("^", "**") if pred is not None else None
            try:
                pred = eval(query) if (query != "" and query is not None) else None
            except:
                pred = None
        else:
            sentences = response.split(".")
            sentences = [s for s in sentences if s != ""]
            pred_sentence = sentences[-1] if len(sentences) > 0 else ""
            pattern = re.compile(r"-?[1-9]\d*")
            pred = pattern.findall(pred_sentence)
            pred = int(pred[-1].replace(",", "")) if len(pred) > 0 else None

        id = data["id"]
        answer = data["answer"].replace(",", "")
        try:
            pred = round(float(pred), 2)
        except:
            pred = 0.0
        answer = round(float(answer), 2)
        if answer == pred:
            for key in dataset_total:
                if key in id:
                    dataset_right[key] += 1
                    dataset_right["total"] += 1
                    break
        for key in dataset_total:
            if key in id:
                dataset_total[key] += 1
                break

        data["pred"] = pred
        datas.append(data)
    acc = {}
    for key in dataset_total:
        acc[key] = dataset_right[key] / dataset_total[key]
    acc["total"] = dataset_right["total"] / len(datas)
    tool_rate = use_tool / len(datas)
    return acc, tool_rate, datas


@torch.no_grad()
def lama_evaluate(args, data_responses):
    dataset_right = {
        "TREx": 0,
        "total": 0
    }

    dataset_total = {
        "TREx": 0,
    }

    model_path = "PLMs/atlas-nq-large/model.pth.tar"
    reader_model_path = "PLMs/t5-large-lm-adapt"
    retriever_model_path = "PLMs/contriever"
    index_path = "PLMs/index-large"

    # load index
    index = DistributedIndex()
    index.load_index(path=index_path, total_saved_shards=64)

    # load model
    checkpoint = torch.load(model_path, map_location="cpu")
    model_dict = checkpoint["model"]
    opt_checkpoint = checkpoint["opt"]
    reader = FiD.from_pretrained(reader_model_path)
    reader_tokenizer = AutoTokenizer.from_pretrained(reader_model_path)
    
    contriever_encoder = Contriever.from_pretrained(retriever_model_path)
    retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_path)
    if opt_checkpoint.query_side_retriever_training:
        retriever = UntiedDualEncoderRetriever(opt_checkpoint, contriever_encoder)
    else:
        retriever = DualEncoderRetriever(opt_checkpoint, contriever_encoder)

    model = Atlas(opt_checkpoint, reader, retriever, reader_tokenizer, retriever_tokenizer)
    model = _load_atlas_model_state(opt_checkpoint, opt_checkpoint, model, model_dict)

    unwrapped_model = get_unwrapped_model_if_wrapped(model)
    reader_tokenizer = unwrapped_model.reader_tokenizer

    # eval
    datas = []
    use_tool = 0
    for data in tqdm(data_responses):
        response = data["response"]
        if "QA" in response:
            use_tool += 1
            pattern = re.compile(r"QA\(.*\)")
            pred = pattern.findall(response)
            pred = pred[0] if len(pred) > 0 else None
            query = pred[len("QA")+1:-1].strip() if pred is not None else None
            if query:
                query_enc = unwrapped_model.retriever_tokenize(query)
                query_ids_retriever = query_enc["input_ids"].cuda()
                query_mask_retriever = query_enc["attention_mask"].cuda()
                retrieved_passages, _ = unwrapped_model.retrieve(
                    index,
                    opt_checkpoint.n_context,
                    query,
                    query_ids_retriever,
                    query_mask_retriever
                )
                reader_tokens, _ = unwrapped_model.tokenize_passages(query, retrieved_passages)
                generation = unwrapped_model.generate(reader_tokens, query)
                pred = reader_tokenizer.decode(generation[0], skip_special_tokens=True)
            else:
                pred = None
        else:
            pred = response.split()
            pred = pred[:20] if len(pred) >= 20 else pred
            pred = " ".join(pred)
        
        id = data["id"]
        answer = data["answer"]
        try:
            for a in answer:
                if a.lower() in pred.lower():
                    for key in dataset_total:
                        if key in id:
                            dataset_right[key] += 1
                            dataset_right["total"] += 1
                            break
                    break
        except:
            pass
        for key in dataset_total:
            if key in id:
                dataset_total[key] += 1
                break
        data["pred"] = pred
        datas.append(data)
    acc = {}
    for key in dataset_total:
        acc[key] = dataset_right[key] / dataset_total[key]
    acc["total"] = dataset_right["total"] / len(datas)
    tool_rate = use_tool / len(datas)
    return acc, tool_rate, datas


@torch.no_grad()
def qa_evaluate(args, data_responses):
    dataset_right = {
        "NaturalQuestions": 0,
        "TriviaQA": 0,
        "WebQuestions": 0,
        "total": 0
    }

    dataset_total = {
        "NaturalQuestions": 0,
        "TriviaQA": 0,
        "WebQuestions": 0,
    }

    prompter = Prompter(args.model)
    generation_config = GenerationConfig(
        temperature=0.2,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        repetition_penalty=1.3,
    )
    llama_model, llama_tokenizer = load_llama_model(args.model, args.lora_weights)
    
    datas = []
    use_tool = 0
    for data in tqdm(data_responses):
        preds = []
        response = data["response"]
        if "WikiSearch" in response:
            use_tool += 1
            pattern = re.compile(r"WikiSearch\(.*\)")
            pred = pattern.findall(response)
            pred = pred[0] if len(pred) > 0 else None
            query = pred[len("WikiSearch")+1:-1].strip() if pred is not None else None
            if query:
                try:
                    tool_return = WikiSearch(query, 2)
                except:
                    print("wikisearch error")
                    continue
                for tr in tool_return:
                    if len(tr.split()) > 384:
                        tr = " ".join(tr.split()[:384])
                    instruction = tr + ". " + data["instruction"]
                    prompt = prompter.generate_prompt(instruction)
                    inputs = llama_tokenizer(prompt, return_tensors="pt")
                    input_ids = inputs["input_ids"].to(device)
                    generation_output = llama_model.generate(
                        input_ids=input_ids,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_new_tokens=128
                    )
                    s = generation_output.sequences[0]
                    output = llama_tokenizer.decode(s, skip_special_tokens=True)
                    pred = prompter.get_response(output)
                    preds.append(pred)
        else:
            response = response.split()
            if len(response) > 20:
                response = response[:20]
            pred = " ".join(response)
            preds.append(pred)
        
        id = data["id"]
        answer = data["answer"]
        flag = False
        try:
            for a in answer:
                for pred in preds:
                    if a.lower() in pred.lower():
                        for key in dataset_total:
                            if key in id:
                                dataset_right[key] += 1
                                dataset_right["total"] += 1
                                break
                        flag = True
                        break
                if flag:
                    break
        except:
            pass
        for key in dataset_total:
            if key in id:
                dataset_total[key] += 1
                break
        data["pred"] = preds
        datas.append(data)
    acc = {}
    for key in dataset_total:
        acc[key] = dataset_right[key] / dataset_total[key]
    acc["total"] = dataset_right["total"] / len(datas)
    tool_rate = use_tool / len(datas)
    return acc, tool_rate, datas


def mlqa_evaluate(args, data_responses):
    prompter = Prompter(args.model)
    generation_config = GenerationConfig(
        temperature=0.2,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        repetition_penalty=1.3,
    )
    llama_model, llama_tokenizer = load_llama_model(args.model, args.lora_weights)
    
    tool_model_name = "PLMs/nllb-200-distilled-600M"
    tool_tokenizer = AutoTokenizer.from_pretrained(tool_model_name)
    tool_model = AutoModelForSeq2SeqLM.from_pretrained(tool_model_name)

    datas = []
    right = 0
    use_tool = 0
    for data in tqdm(data_responses):
        response = data["response"]
        if "translator" in response:
            use_tool += 1
            pattern = re.compile(r"translator\(.*\)")
            pred = pattern.findall(response)
            pred = pred[0] if len(pred) > 0 else None
            query = pred[len("translator")+1:-1].strip() if pred is not None else None
            if query:
                input_ids = tool_tokenizer(query, return_tensors="pt")
                outputs = tool_model.generate(
                    **input_ids,
                    forced_bos_token_id=tool_tokenizer.lang_code_to_id["eng_Latn"]
                )
                tool_return = tool_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                instruction = "Given a context, please answer the question."
                input = data["input"]
                input = input.split("\nQuestion: ")
                input[1] = tool_return.strip()
                input = "\nQuestion: ".join(input)
                prompt = prompter.generate_prompt(instruction, input)
                inputs = llama_tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].to(device)
                generation_output = llama_model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=128
                )
                s = generation_output.sequences[0]
                output = llama_tokenizer.decode(s, skip_special_tokens=True)
                pred = prompter.get_response(output)
            else:
                pred = None
        else:
            pred = response
        
        answer = data["answer"]
        try:
            pred = pred.split()
            if len(pred) > 20:
                pred = pred[:20]
            pred = " ".join(pred)
            if answer.lower() in pred.lower():
                right += 1
        except:
            pass
        data["pred"] = pred
        datas.append(data)
    acc = {"MLQA": right / len(datas), "total": right / len(datas)}
    tool_rate = use_tool / len(datas)
    return acc, tool_rate, datas
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--lora_weights", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--target_path", type=str, default=None)
    args = parser.parse_args()

    eval_dict = {
        "Math": math_evaluate,
        "LAMA": lama_evaluate,
        "QA": qa_evaluate,
        "MLQA": mlqa_evaluate
    }

    print("Task:", args.task)

    lines = open(args.data_path).readlines()
    data_response = [json.loads(line) for line in lines]

    acc, tool_rate, outputs = eval_dict[args.task](args, data_response)

    if not os.path.exists(args.target_path):
        os.mkdir(args.target_path)
    target_path = os.path.join(args.target_path)
    with open(target_path, "w") as f:
        f.write(f"acc: {acc}, tool_rate: {tool_rate}\n")
        for data in outputs:
            try:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            except:
                continue
