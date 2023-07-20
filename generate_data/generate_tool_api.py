import argparse
import json
import random
import sys
from tenacity import retry, stop_after_attempt, wait_random_exponential
from utils.utils import BasePrompt
from utils.utils import set_openai_key, set_proxy, API_NAME_DICT
from utils.template import examples_dict, sys_dict

set_proxy("")

def return_examples(examples):
    examples_prompt = ""
    random.shuffle(examples)
    for ex in examples:
        examples_prompt += f"Question: {ex['question']}\nGolden answers: {ex['golden_answer']}\nOutput: {ex['output']}\n"
    return examples_prompt

key_seed = 0

@retry(stop=stop_after_attempt(10), wait=wait_random_exponential(min=2, max=60))
def generate_data(sys_prompt, examples, golden_answer, data, openai_api_list, multi_api_keys):
    global key_seed
    promter = BasePrompt()
    prompt = ""
    if options.engine in API_NAME_DICT["chatgpt"]:
        promter.set_system_prompt(sys_prompt)
    else:
        prompt = sys_prompt

    if multi_api_keys:
        set_openai_key(openai_api_list[key_seed % len(openai_api_list)])
        key_seed += 1
        
    prompt += return_examples(examples)
    prompt += f"Question: {data['question']}\nGolden answers: {golden_answer[:10]}\nOutput: "
    prompt = promter.build_prompt(prompt)
    print(prompt)
    result = promter.get_openai_result(engine=options.engine, max_tokens=options.max_tokens)
    return result.strip()

def main(options):
    if options.multi_api_keys:
        openai_api_list = [key.strip() for key in open(options.api_keys_file, "r").readlines()]
    set_openai_key(options.api_key)
    examples = examples_dict[options.dataset_name]
    sys_prompt = sys_dict[options.dataset_name]

    already = set()
    if options.mode == "a":
        with open(options.output, "r") as reader:
            for line in reader:
                data = json.loads(line)
                already.add(data["id"])

    writer = open(options.output, options.mode)
    with open(options.input, "r") as reader:
        for line in reader:
            data = json.loads(line)
            if data["id"] in already:
                print(f'{data["id"]} already exists!')
                continue

            golden_answer = data['answer']

            result = generate_data(
                sys_prompt=sys_prompt,
                examples=examples, 
                golden_answer=golden_answer, 
                data=data, 
                openai_api_list=openai_api_list, 
                multi_api_keys=options.multi_api_keys
            )

            print(f'{data["id"]} - {json.dumps(result, ensure_ascii=False)}')
            data['tool'] = result
            writer.write(json.dumps(data, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--input", type=str, default=None)
    parse.add_argument("--output", type= str, default=None)
    parse.add_argument("--api_key", type=str, default="")
    parse.add_argument("--dataset_name", type=str, default=None)
    parse.add_argument("--mode", type=str, default="a")
    parse.add_argument("--engine", type=str, default="gpt-3.5-turbo-0301")
    parse.add_argument("--max_tokens", type=int, default=128)
    parse.add_argument("--multi_api_keys", action="store_true")
    parse.add_argument("--api_keys_file", type=str)
    options = parse.parse_args()

    main(options)