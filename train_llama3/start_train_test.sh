#!/bin/bash -l
#PBS -A hp-ptycho
#PBS -l select=1
#PBS -l walltime=00:05:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -o mpi_output.txt
#PBS -e mpi_errors.txt

export MASTER_PORT=29501

NRANKS=$(wc -l < "${PBS_NODEFILE}")
NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
NGPUS="$((${NRANKS}*${NGPU_PER_RANK}))"

cd /home/binkma/bm_ds/train_llama3

module use /soft/modulefiles
module load conda
conda activate zero

export HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_DISABLED="true"

# DeepSpeed配置文件路径
DS_CONFIG="ds_config.json"

# dcgmi dmon -e 1009,1010 -i 0,1 -d 160 > pcie_throughput.log & DCGMI_PID=$!
deepspeed --include localhost:0,1 --master_port=29501 train3.py \
    --deepspeed ${DS_CONFIG} \
    --seed 100 \
    --model_name_or_path "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --dataset_name "binkma/openhermes-2.5-llama3" \
    --chat_template_format "none" \
    --add_special_tokens False \
    --append_concat_token False \
    --splits "train,test" \
    --max_seq_length 256 \
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
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
    --max_grad_norm 1.0 \
    --output_dir "llama3-openhermes-2.5" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing False \
    --use_reentrant True \
    --dataset_text_field "content" \
    --use_flash_attn True \
    --use_peft_lora False \
    --lora_target_modules "all-linear" \
    --use_4bit_quantization False \
    --use_nested_quant False \
    --bnb_4bit_compute_dtype "bfloat16" \
    --bnb_4bit_quant_storage_dtype "bfloat16"