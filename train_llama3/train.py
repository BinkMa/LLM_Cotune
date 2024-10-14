import os
import torch
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import HfArgumentParser, TrainingArguments
from utils import (
    create_datasets, 
    create_and_prepare_model, 
    loftq_init,
    ZephyrSpecialTokens,
    ChatmlSpecialTokens
)
from dataclasses import asdict
import inspect
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import deepspeed
import yaml
import json
from deepspeed.ops.adam import DeepSpeedCPUAdam
# from transformers import AutoTokenizer
import torch.cuda.nvtx as nvtx
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler.profiler import ProfilerAction

# 假设您已经创建了 tokenizer


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    use_flash_attn: bool = field(default=False, metadata={"help": "Whether to use flash attention"})
    use_4bit_quantization: bool = field(default=False, metadata={"help": "Whether to use 4-bit quantization"})
    use_8bit_quantization: bool = field(default=False, metadata={"help": "Whether to use 8-bit quantization"})
    use_nested_quant: bool = field(default=False, metadata={"help": "Whether to use nested quantization"})
    bnb_4bit_compute_dtype: str = field(default="float16", metadata={"help": "Compute dtype for 4-bit quantization"})
    bnb_4bit_quant_type: str = field(default="nf4", metadata={"help": "Quantization type for 4-bit quantization"})
    bnb_4bit_quant_storage_dtype: str = field(default="float16", metadata={"help": "Quant storage dtype for 4-bit quantization"})

@dataclass
class DataArguments:
    dataset_name: str = field(metadata={"help": "The name of the dataset to use"})
    dataset_text_field: str = field(default="content", metadata={"help": "The input text field of the dataset"})
    max_seq_length: int = field(default=2048, metadata={"help": "The maximum sequence length"})
    chat_template_format: str = field(default="none", metadata={"help": "The chat template format to use"})
    add_special_tokens: bool = field(default=False, metadata={"help": "Whether to add special tokens"})
    append_concat_token: bool = field(default=False, metadata={"help": "Whether to append concat token"})
    splits: str = field(default="train,test", metadata={"help": "The splits of the dataset to use"})

@dataclass
class LoraArguments:
    use_peft_lora: bool = field(default=False, metadata={"help": "Whether to use PEFT LoRA"})
    lora_r: int = field(default=8, metadata={"help": "LoRA attention dimension"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    lora_target_modules: str = field(default="all-linear", metadata={"help": "Target modules for LoRA"})
    
@dataclass
class CustomArguments(TrainingArguments):
    use_unsloth: bool = field(default=False, metadata={"help": "Whether to use Unsloth optimization"})
    packing: bool = field(default=False, metadata={"help": "Whether to use packing for efficient training"})
    use_reentrant: bool = field(default=False, metadata={"help": "Whether to use reentrant for gradient checkpointing"})
    custom_max_steps: int = field(default=25, metadata={"help": "Custom maximum number of training steps"})
    deepspeed: Optional[str] = field(default=None, metadata={"help": "Path to deepspeed config file (YAML or JSON)"})

def train_loop(model_engine, train_dataloader, optimizer, lr_scheduler, training_args):
    max_steps = training_args.max_steps
    global_step = 0

    # prof = profile(
    #     activities=[
    #         ProfilerActivity.CPU,
    #         ProfilerActivity.CUDA,
    #     ],
    #     schedule=torch.profiler.schedule(
    #         wait=10,  # 等待 4 步（即从第 5 步开始）
    #         warmup=1,
    #         active=4,  # 激活 6 步（覆盖第 5 到第 10 步）
    #     ),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('/grand/hp-ptycho/binkma/llm/llama3'),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True,
    # )

    # prof.start()
    for epoch in range(int(training_args.num_train_epochs)):
        model_engine.train()
        for step, batch in enumerate(train_dataloader):
            with record_function("data_transfer"):
                batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            
            with record_function("forward"):
                outputs = model_engine(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                loss = outputs.loss
            
            with record_function("backward"):
                model_engine.backward(loss)
            
            with record_function("optimizer_step"):
                model_engine.step()
                lr_scheduler.step()
            
            global_step += 1
            if step % training_args.logging_steps == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

            # prof.step()
            if global_step == 17:  # 在第 10 步后停止 profiling
                # prof.stop()
                break

            if global_step >= max_steps:
                print(f"Reached max_steps ({max_steps}). Stopping training.")
                return
    # prof.stop()


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, CustomArguments))
    model_args, data_args, lora_args, custom_args = parser.parse_args_into_dataclasses()
    
    valid_args = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    training_args_dict = {k: v for k, v in asdict(custom_args).items() if k in valid_args}
    custom_args_dict = {k: v for k, v in asdict(custom_args).items() if k not in valid_args}
    
    training_args_dict['max_steps'] = custom_args.custom_max_steps
    
    training_args = TrainingArguments(**training_args_dict)
    print(f"Max steps: {training_args.max_steps}")

    # Create datasets
    train_dataset, eval_dataset = create_datasets(None, data_args, training_args)

    # Create and prepare model
    model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args, custom_args, lora_args)

    # def collate_fn(batch):
    #     # 假设 'content' 是包含文本的字段
    #     texts = [item['content'] for item in batch]
        
    #     # 使用 tokenizer 处理文本
    #     encoded = tokenizer(texts, padding=True, truncation=True, max_length=data_args.max_seq_length, return_tensors="pt")
        
    #     return encoded

    def collate_fn(batch):
        texts = [item['content'] for item in batch]
        
        inputs = tokenizer(texts, padding=True, truncation=True, 
                        max_length=data_args.max_seq_length, return_tensors="pt")
        

        labels = inputs['input_ids'].clone()
        labels[:, :-1] = inputs['input_ids'][:, 1:]
        labels[:, -1] = -100  
        

        labels[inputs['attention_mask'] == 0] = -100
        
        inputs['labels'] = labels
        return inputs

    # Create DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )

    # Create optimizer
    optimizer = DeepSpeedCPUAdam(model.parameters(),
                             lr=1e-5,
                             betas=(0.9, 0.999),
                             eps=1e-8,
                             bias_correction=True)

    # Create learning rate scheduler
    num_training_steps = len(train_dataloader) * training_args.num_train_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=num_training_steps
    )

    if custom_args.deepspeed:
        with open(custom_args.deepspeed, 'r') as f:
            if custom_args.deepspeed.endswith('.yaml') or custom_args.deepspeed.endswith('.yml'):
                ds_config = yaml.safe_load(f)
            else:
                ds_config = json.load(f)
    else:
        ds_config = None

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        # model_parameters=get_model_parameters(model),
        config=ds_config
    )

    # Train
    train_loop(model_engine, train_dataloader, optimizer, lr_scheduler, training_args)

    # Evaluation (if needed)
    # if training_args.do_eval and eval_dataset is not None:
    #     model_engine.eval()
    #     eval_loss = 0
    #     eval_steps = 0
    #     for batch in DataLoader(eval_dataset, batch_size=training_args.per_device_eval_batch_size):
    #         batch = {k: v.to(model_engine.device) for k, v in batch.items()}
    #         with torch.no_grad():
    #             outputs = model_engine(**batch)
    #         eval_loss += outputs.loss.item()
    #         eval_steps += 1
    #     eval_loss /= eval_steps
    #     print(f"Evaluation Loss: {eval_loss}")

if __name__ == "__main__":
    main()