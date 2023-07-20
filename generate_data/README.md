# Data Generation

Here we introduce how to generate the training data for the **[Behavior Cloning stage](#-Behavior-Cloning)** and the **[RLEF stage](#-RLEF)**. You can also download the **mixed** training data for **Vicuna-7B** directly in [Google Drive]().

## Behavior Cloning Data Generation

During this stage, we construct the dataset based on the hypothesis that a LM needs a tool for help when it gives a wrong answer and does not need a tool when its answer is right. **So note that before generating the training data, you are recommended to first let your LM answer the question (remember to save the LM's response for each instance).** Questions with wrong answers can be saved in `data/raw/{dataset_name}/{dataset_name}_wrong.json` and questions with right answers can be saved in `data/raw/{dataset_name}/{dataset_name}_right.json`.

Generate Tool APIs for questions with wrong answers:

```bash
python generate_tool_api.py \
		--input ../data/raw/GSM8K/GSM8K_wrong.json \
		--output ../data/stage1/GSM8K/GSM8K_wrong.json \
		--dataset_name GSM8K \
		--model w \
		--multi_api_keys \
		--engine gpt-turbo-0301 \
		--api_keys_file ./openai_api_keys.txt
```

Then merge `data/raw/{dataset_name}/{dataset_name}_right.json` and `data/stage1/{dataset_name}/{dataset_name}_wrong.json` into `data/stage1/{dataset_name}/{dataset_name}.json`. The merged file's format is as follows:

```json
{
		"instruction": "Given a math problem, please answer it and you can use a calculator for help.",
  	"input": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  	"output": "The answer is 72." // use the response of the LM as the output directly for questions with right answers.
}
{
  	"instruction": "Given a math problem, please answer it and you can use a calculator for help.",
  	"input": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
  	"output": "calculator((12/60)*50)" // use the Tool API as the output for questions with wrong answers.
}
```

The merged file can be directly used to train in the behavior cloning stage.

## RLEF Data Generation

We collect five responses for each question from four different models, e.g. ChatGPT, InstuctGPT, Vicuna-7B, Alpaca-7B, and the output of the training data in Behavior Cloning stage as the pseudo-human-expert response.

For **ChatGPT** and **InstructGPT**, we prompt them with instructions and few-shot examples:

```bash
python generate_response_gpt.py \
		--input ../data/raw/GSM8K/GSM8K.json \
		--output ../data/stage2/GSM8K/GSM8K_davinci_response.json \
		--task math \
		--mode w \
		--multi_api_keys \
		--example_num 4 \
		--engine text-davinci-003 \
		--api_key_file ./openai_api_keys.txt
```

For **Vicuna-7B** and **Alpaca-7B**, fine-tune them with LoRA for a few steps in order to equip them with initial abilities for question answering and tool generation (you can use a LoRA weight of the middle step of the training process in the Behavior Cloning stage), and then generate responses:

```bash
python generate_response_llama.py \
		--base_model PLMs/vicuna-7b \ # path to the weights of Vicuna-7B
		--task math \
		--data_path ../data/raw/GSM8K/GSM8K.json \
		--lora_weights vicuna-stage1-lora-step-39 \
		--prompt_template vicuna \
		--output_path ../data/stage2/GSM8K/GSM8K_vicuna_response.json
```

After all the responses are generated, the RLEF stage data folder structure is as follows (take GSM8K as an example):

```
data/stage2/GSM8K
 |-- GSM8K_gold_response.json
 |-- GSM8K_chatgpt_response.json
 |-- GSM8K_davinci_response.json
 |-- GSM8K_vicuna_response.json
 |-- GSM8K_alpaca_response.json
```

Then score each response and generate the final training dataset:

```bash
python score.py \
		--task math \
		--model vicuna \
		--lora_weights vicuna-stage1-lora \
		--data_path ../data/stage2/GSM8K \
		--target_path ../data/stage2/GSM8K
```

Finally, the format of the training dataset `data/stage2/{dataset_name}/{dataset_name}.json` for the RLEF stage is as follows:

```json
{
  	"query": "Given a math problem, please solve it and you can use a calculator for help.\nJean has three times as much money as Jane. They have a combined total of $76. How much money does Jean have?",
  	"responses": ["calculator(3*19)", "calculator(3*19)", " calculator(76/4*3)", "Jean has $76 / 3 = $<<76/3=25>>25", "Jean has $54."], 		"origin": ["gold", "chatgpt", "davinci", "vicuna", "alpaca"],
  	"scores": [20, 20, 20, 10, 15],
  	"answer": ["57.0", "57.0", "57.0", "25.0", "54.0"]
}
```

