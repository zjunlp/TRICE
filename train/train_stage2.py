import logging
from dataclasses import dataclass, field
from typing import Optional, Dict
import io
import os
import sys
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from transformers import LlamaForCausalLM, LlamaTokenizer
import json
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict
)

# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "</s> Instruction:\n{instruction}\n\n</s> Input:\n{input}\n\n</s> Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "</s> Instruction:\n{instruction}\n\n</s> Response:\n"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")


@dataclass
class DataArguments:
    data_path: str = field(default="", metadata={"help": "Path to the training data."})
    stop_response: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    batch_size: int = field(default=128)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    rrhf_weight: float = field(default=100.0)
    length_penalty: float = field(default=1.0)
    only_use_provide: bool = field(default=False)
    only_use_sample: bool = field(default=False)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


class ScoreDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(ScoreDataset, self).__init__()
        logging.warning("Loading data...")
        with open(data_path, 'r') as f:
            lines = f.readlines()
        self.data = [json.loads(line.strip()) for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return dict(input_ids=self.data[i])

def _single_tokenize(text, tokenizer, max_len=None):
    if max_len is None:
        max_len = tokenizer.model_max_length
    toked = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        max_length=max_len,
        truncation=True,
    )
    return toked['input_ids'][0]

def stop_response(res):
    stops = ['\n\nHuman:', '\n\nAssistant:', '\n\nhuman:', '\n\nassistant:']
    for stop in stops:
        if res.find(stop) >= 0:
            res = res[:res.find(stop)].strip()
    return res

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    stop_response: bool

    def __call__(self, instances):

        idxs = []
        all_scores = []
        input_ids = []
        score_mask = []
        labels = []
        for idx, ins in enumerate(instances):

            ins = ins['input_ids'] # hack
            query = ins['query']
            responses = ins['responses']
            scores = ins['scores']
            all_scores.append(scores)
            idxs.append([idx] * len(scores))

            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

            def get_input(query):
                if query.find('\n') == -1:
                    return ''
                return '\n'.join(query.split('\n')[1:])

            example = {'instruction':query.split('\n')[0], 'input':get_input(query)}
            prompt_input = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            # print(prompt_input)
            query_input_ids = _single_tokenize(prompt_input, self.tokenizer)
            query_target = torch.LongTensor([IGNORE_INDEX] * (query_input_ids.shape[0] - 1))
            dummy_target = torch.LongTensor([IGNORE_INDEX])
            for res in responses:
                if self.stop_response:
                    r = stop_response(res)
                else:
                    r = res
                res_input_ids = _single_tokenize(r + self.tokenizer.eos_token, self.tokenizer, max_len=self.tokenizer.model_max_length-query_input_ids.shape[0]) # eos here
                input_ids.append(torch.cat((query_input_ids, res_input_ids), dim=0))
                labels.append(torch.cat((query_target, res_input_ids, dummy_target), dim=0))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=labels,
            idxs=torch.LongTensor(idxs),
            scores=torch.FloatTensor(all_scores),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = ScoreDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, stop_response=data_args.stop_response)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


class RRHFTrainer(Trainer):
    def gather_logits_labels(self, logits, labels):

        mask = (labels != -100).float()
        new_logits = logits.clone()  # Create a copy to avoid in-place modification
        labels[labels == -100] = 0 
        output = torch.gather(new_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        output = output * mask # B * L
        return output

    def get_score(self, logit_label, labels):
        mask = (labels != -100).float()
        length = mask.sum(-1)
        scores = logit_label.sum(-1) / (length ** self.args.length_penalty)
        return scores

    def rrhf_loss(self, scores, idxs, rw_scores):
        res_num = rw_scores.size(1)
        new_scores = scores.view(-1, res_num)
        diff = new_scores.unsqueeze(1) - new_scores.unsqueeze(-1)
        rw_diff = rw_scores.unsqueeze(1) - rw_scores.unsqueeze(-1)
        aval = torch.bitwise_and(rw_diff > 0, diff < 0)
        return -diff[aval].sum()

    def sft_loss(self, logit_label, idxs, rw_scores):
        max_idx = torch.argmax(rw_scores, dim=1)
        res_num = rw_scores.size(1)
        logit_label_batch = logit_label.view(-1, res_num, logit_label.size(-1))
        index = max_idx.view(-1, 1, 1).repeat(1, 1, logit_label_batch.size(-1))
        logit_label_batch = torch.gather(logit_label_batch, dim=1, index=index).squeeze()
        return -torch.sum(logit_label_batch.mean())

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.args.only_use_provide:
            inputs['input_ids'] = inputs['input_ids'][-2:]
            inputs['attention_mask'] = inputs['attention_mask'][-2:]
            inputs['labels'] = inputs['labels'][-2:]
            inputs["idxs"] = inputs["idxs"][:,-2:]
            inputs["scores"] = inputs["scores"][:,-2:]
        if self.args.only_use_sample:
            inputs['input_ids'] = inputs['input_ids'][:-2]
            inputs['attention_mask'] = inputs['attention_mask'][:-2]
            inputs['labels'] = inputs['labels'][:-2]
            inputs["idxs"] = inputs["idxs"][:,:-2]
            inputs["scores"] = inputs["scores"][:,:-2]
        logits = model(input_ids=inputs.get('input_ids'), attention_mask=inputs.get('attention_mask'))[0] # (batch * cand) * L * V
        logits = F.log_softmax(logits, dim=-1)
        logit_label = self.gather_logits_labels(logits, inputs.get("labels"))
        scores = self.get_score(logit_label, inputs.get("labels"))
        rrhf_loss = self.rrhf_loss(scores, inputs.get("idxs"), inputs.get("scores"))
        sft_loss = self.sft_loss(logit_label, inputs.get("idxs"), inputs.get("scores"))
        loss = self.args.rrhf_weight * rrhf_loss + sft_loss
        return (loss, scores) if return_outputs else loss

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args)
    print(data_args)

    training_args.gradient_accumulation_steps = training_args.batch_size // training_args.per_device_train_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        training_args.gradient_accumulation_steps = training_args.gradient_accumulation_steps // world_size

    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map
    )

    tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path)

    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = tokenizer.bos_token_id = 1
    model.config.eos_token_id = tokenizer.eos_token_id = 2
    tokenizer.model_max_length = training_args.model_max_length

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if training_args.resume_from_checkpoint:
        checkpoint_name = os.path.join(
            training_args.resume_from_checkpoint, "pytorch_model.bin"
        )
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                training_args.resume_from_checkpoint, "adapter_model.bin"
            )
            training_args.resume_from_checkpoint = (
                False
            )
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    
    trainer = RRHFTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    model.save_pretrained(training_args.output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    train()
