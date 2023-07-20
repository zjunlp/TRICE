import sys
import json
import fire
from tqdm import tqdm
import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from utils.prompter import Prompter


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "",
    task: str = "",
    data_path: str = "",
    lora_weights: str = "",
    output_path: str = "",
    prompt_template: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    print("pre-trained model's BOS EOS and PAD token id:",bos,eos,pad," => It should be 1,2,none")
    print(f"load_8bit={load_8bit}")
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # same as unk token id
    model.config.bos_token_id = tokenizer.bos_token_id = 1
    model.config.eos_token_id = tokenizer.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.2,
        top_p=0.75,
        top_k=40,
        num_beams=4,            # 1->4
        max_new_tokens=128,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,        
            repetition_penalty=1.3,     # add
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s, skip_special_tokens=True)
        return prompter.get_response(output)

    datas = []
    with open(data_path) as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = json.loads(line)
            if task == "Math":
                instruction = "Given a math problem, please answer it and you can use a calculator for help."
                response = evaluate(instruction=instruction, input=line["input"])
            elif task == "MLQA":
                instruction = "Given a context, please answer the question in English and you can use a translator for help."
                response = evaluate(instruction=instruction, input=line["input"])
            elif task == "LAMA":
                instruction = "Given a question, please answer it and you can use a QA model for help."
                response = evaluate(instruction=instruction, input=line["instruction"])
            elif task == "QA":
                instruction = "Given a question, please answer it and you can use a WikiSearch for help."
                response = evaluate(instruction=instruction, input=line["instruction"])
            line["response"] = response
            datas.append(line)

    with open(output_path, "w") as f:
        for data in datas:
            try:
                f.write(json.dumps(data, ensure_ascii=False)+'\n')
            except:
                pass


if __name__ == "__main__":
    fire.Fire(main)
