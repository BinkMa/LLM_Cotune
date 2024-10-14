import os
import torch
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import HfArgumentParser, TrainingArguments, AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator
from utils import (
    create_datasets, 
    create_and_prepare_model, 
    loftq_init,
    ZephyrSpecialTokens,
    ChatmlSpecialTokens
)
from dataclasses import asdict
import inspect
from transformers import TrainerCallback,get_scheduler
import deepspeed
import yaml
from torch.profiler import profile, record_function, ProfilerActivity

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="facebook/opt-13b", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    # Add other model-specific arguments
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
    dataset_text_field: str = field(default="text", metadata={"help": "The input text field of the dataset"})
    max_seq_length: int = field(default=2048, metadata={"help": "The maximum sequence length"})
    chat_template_format: str = field(default="none", metadata={"help": "The chat template format to use"})
    add_special_tokens: bool = field(default=False, metadata={"help": "Whether to add special tokens"})
    append_concat_token: bool = field(default=False, metadata={"help": "Whether to append concat token"})
    splits: str = field(default="train,test", metadata={"help": "The splits of the dataset to use"})

@dataclass
class LoraArguments:
    use_peft_lora: bool = field(default=True, metadata={"help": "Whether to use PEFT LoRA"})
    lora_r: int = field(default=8, metadata={"help": "LoRA attention dimension"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    lora_target_modules: str = field(default="q_proj,v_proj", metadata={"help": "Target modules for LoRA"})
    
@dataclass
class CustomArguments(TrainingArguments):
    use_unsloth: bool = field(default=False, metadata={"help": "Whether to use Unsloth optimization"})
    packing: bool = field(default=False, metadata={"help": "Whether to use packing for efficient training"})
    use_reentrant: bool = field(default=False, metadata={"help": "Whether to use reentrant for gradient checkpointing"})
    custom_max_steps: int = field(default=15, metadata={"help": "Custom maximum number of training steps"})

class StepInfoCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        print(f"Completed step {state.global_step}")

def main():
    # deepspeed.ops.op_builder.CPUAdamBuilder().load()
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, CustomArguments))
    model_args, data_args, lora_args, custom_args = parser.parse_args_into_dataclasses()
    
    valid_args = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    training_args_dict = {k: v for k, v in asdict(custom_args).items() if k in valid_args}
    custom_args_dict = {k: v for k, v in asdict(custom_args).items() if k not in valid_args}
    
    training_args_dict['max_steps'] = custom_args.custom_max_steps
    
    training_args = TrainingArguments(**training_args_dict)
    print(f"Max steps: {training_args.max_steps}")
    #load deepspeed config from deepspeed_config.yaml
    # with open('deepspeed_config.yaml', 'r') as file:
    #     ds_config = yaml.safe_load(file)
    
    # Setup accelerator
    accelerator = Accelerator()

    # Create datasets
    train_dataset, eval_dataset = create_datasets(None, data_args, training_args)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=model_args.use_4bit_quantization,
        bnb_4bit_use_double_quant=model_args.use_nested_quant,
        bnb_4bit_compute_dtype=getattr(torch, model_args.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type="nf4",  # 添加这一行
    )
    

    # Prepare model (including LoRA if enabled)
    model, peft_config ,tokenizer= create_and_prepare_model(model_args, data_args, custom_args, lora_args)


    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field=data_args.dataset_text_field,
        max_seq_length=data_args.max_seq_length,
        tokenizer=tokenizer,
        packing=custom_args_dict.get('packing', False),
        callbacks=[StepInfoCallback()],
    )

    # Train
    # with profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA ],
    #     schedule=torch.profiler.schedule(wait=0, warmup=0, active=2, repeat=1),
    #     profile_memory=True) as prof:

    with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
                profile_memory=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler('/grand/hp-ptycho/binkma/llm/opt/'),
            ) as prof: 
        
        train_result = trainer.train()

        # prof.export_chrome_trace("./trace")
    # Save model
    trainer.save_model()

    # Log and save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

if __name__ == "__main__":
    main()