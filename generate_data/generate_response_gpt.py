import os
import json
import sys
import argparse
from tenacity import retry, stop_after_attempt, wait_random_exponential
from utils.utils import BasePrompt
from utils.utils import set_openai_key, set_proxy, API_NAME_DICT

set_proxy("")

def return_examples(examples):
    examples_prompt = ""
    for ex in examples:
        examples_prompt += f"Input: {ex['input']}\nOutput: {ex['output']}\n"
    return examples_prompt

key_seed = 0
print_num = 5

@retry(stop=stop_after_attempt(10), wait=wait_random_exponential(min=2, max=60))
def generate_data(examples, data, openai_api_list, options):
    global key_seed
    global print_num
    prompt = ""
    prompter = BasePrompt()
    if options.engine in API_NAME_DICT["chatgpt"]:
        prompter.set_system_prompt(examples[0]['instruction'] + "\nHere are some examples:\n")
    else:
        prompt = examples[0]['instruction'] + "\nHere are some examples:\n"
    if options.multi_api_keys:
        set_openai_key(openai_api_list[key_seed % len(openai_api_list)])
        key_seed += 1

    prompt += return_examples(examples)
    prompt += f"Input: {data['input']}\nOutput: "
    if print_num > 0:
        print(prompt)
    print_num -= 1
    
    prompt = prompter.build_prompt(prompt)
    result = prompter.get_openai_result(engine=options.engine, max_tokens=options.max_tokens)
    return result


def main(options):
    if options.multi_api_keys:
        openai_api_list = [key.strip() for key in open(options.api_keys_file, "r").readlines()]
    set_openai_key(options.api_key)
    example_file = os.path.join(options.example_file, f"{options.task}.json")
    examples = [json.loads(line) for line in open(example_file, 'r').readlines()][:options.example_num]

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

            result = generate_data(
                examples=examples,  
                data=data, 
                openai_api_list=openai_api_list, 
                options=options,
            )

            print(f'{data["id"]} - {json.dumps(result, ensure_ascii=False)}')
            data[options.output_text] = result
            writer.write(json.dumps(data, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--input", type=str, default=None)
    parse.add_argument("--output", type=str, default=None)
    parse.add_argument("--api_key", type=str, default="")
    parse.add_argument("--mode", type=str, default="a")
    parse.add_argument("--engine", type=str, default="gpt-3.5-turbo")
    parse.add_argument("--max_tokens", type=int, default=128)
    parse.add_argument("--example_num", type=int, default=4)
    parse.add_argument("--example_file", type=str, default="./example")
    parse.add_argument("--task", type=str, default="")
    parse.add_argument("--output_text", type=str, default="output")
    parse.add_argument("--multi_api_keys", action="store_true")
    parse.add_argument("--api_keys_file", type=str)
    options = parse.parse_args()

    main(options)