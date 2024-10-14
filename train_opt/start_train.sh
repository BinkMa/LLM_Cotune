#!/bin/bash -l

#PBS -A hp-ptycho
#PBS -l select=1
#PBS -l walltime=00:05:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -o mpi_output.txt
#PBS -e mpi_errors.txt


NRANKS=$(wc -l < "${PBS_NODEFILE}")
NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
NGPUS="$((${NRANKS}*${NGPU_PER_RANK}))"

# $cd /home/binkma/bm_ANL/mpidemo/
# cd $PBS_O_WORKDIR 
cd /home/binkma/bm_ds/train_opt

module use /soft/modulefiles
module load conda
conda activate zero

export MASTER_PORT=0
export CUDA_VISIBLE_DEVICES=2,3
export HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_DISABLED="true"


# --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \

numactl --cpunodebind=2,3 accelerate launch --config_file "deepspeed_config.yaml" --main_process_port=29500 train.py \
--seed 100 \
--model_name_or_path "facebook/opt-13b" \
--dataset_name "binkma/openhermes-2.5-llama3" \
--chat_template_format "none" \
--add_special_tokens False \
--append_concat_token False \
--splits "train,test" \
--max_seq_length 512 \
--num_train_epochs 1 \
--logging_steps 5 \
--log_level "info" \
--logging_strategy "steps" \
--evaluation_strategy "no" \
--save_strategy "steps" \
--push_to_hub \
--hub_private_repo True \
--hub_strategy "every_save" \
--bf16 True \
--packing True \
--learning_rate 1e-4 \
--lr_scheduler_type "cosine" \
--dataset_text_field "content" \
--weight_decay 1e-4 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "opt13b-openhermes-2.5" \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 2 \
--gradient_checkpointing True \
--use_reentrant True \
--use_flash_attn True \
--use_peft_lora True \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--lora_target_modules "all-linear" \
--use_4bit_quantization True \
--use_nested_quant True \
--bnb_4bit_compute_dtype "bfloat16" \
--bnb_4bit_quant_storage_dtype "bfloat16"