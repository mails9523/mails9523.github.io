# chongqing
# export ROOT_PATH=/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx
# nanjing
export ROOT_PATH=/mnt/nanjing3cephfs/wx-mm-spr-xxxx

cd $ROOT_PATH/xxxx/code/i2vgen-xl-diffusers
# cd $ROOT_PATH/zhangting/code/i2v_0701/i2vgen-dev

export NCCL_IB_DISABLE=1 
export NCCL_DEBUG=INFO 
export NCCL_P2P_DISABLE=1

echo "NODE_RANK: $RANK"
echo "GPU_NUM: $GPU_NUM"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"



# 多机多卡 全精度
# export NUM_PROCESS=16
# export OUTDIR=logs_multitaskpretrain_bf16
# nohup tensorboard --logdir=$ROOT_PATH/xxxx/code/i2vgen-xl-diffusers/sd-model-finetuned/$OUTDIR --port 8080 --bind_all &
# conda run -n  diffusers \
# accelerate launch --multi_gpu  --num_processes=$NUM_PROCESS --num_machines=$WORLD_SIZE --main_process_port=$MASTER_PORT --main_process_ip=$MASTER_ADDR --machine_rank=$RANK --main_training_function='main' train_multitaskpretrain.py \
#     --init_from_pretrained_2d=$ROOT_PATH/xxxx/data/torch_cache/ali-vilab-i2vgen-xl/models--ali-vilab--i2vgen-xl/snapshots/39e1979ea27be737b0278c06755e321f2b4360d5 \
#     --dataloader_num_workers=$NUM_PROCESS \
#     --resolution=256 \
#     --train_batch_size=1 \
#     --max_train_steps=1000000 \
#     --gradient_accumulation_steps=6 \
#     --checkpointing_steps=1000 \
#     --resume_from_checkpoint=latest \
#     --dataset_file $ROOT_PATH/xxxx/code/i2vgen-xl-diffusers/data/alldata_240722.json \
#     --image_dir $ROOT_PATH/xxxx/code/i2vgen-xl-diffusers/data/gif_data \
#     --log_path logs/$OUTDIR.txt \
#     --output_dir sd-model-finetuned/$OUTDIR
    
# 单机多卡 全精度
# export OUTDIR=logs_multitaskpretrain_bf16
# nohup tensorboard --logdir=$ROOT_PATH/xxxx/code/i2vgen-xl-diffusers/sd-model-finetuned/$OUTDIR --port 8080 --bind_all &
# conda run -n  diffusers \
# accelerate launch --multi_gpu  --num_processes=8 --num_machines=1  --main_process_port=20008\
#     train_multitaskpretrain.py \
#     --init_from_pretrained_2d=$ROOT_PATH/xxxx/data/torch_cache/ali-vilab-i2vgen-xl/models--ali-vilab--i2vgen-xl/snapshots/39e1979ea27be737b0278c06755e321f2b4360d5 \
#     --dataloader_num_workers=8 \
#     --resolution=256 \
#     --train_batch_size=1 \
#     --max_train_steps=500000 \
#     --gradient_accumulation_steps=6 \
#     --checkpointing_steps=1000 \
#     --resume_from_checkpoint=latest \
#     --dataset_file $ROOT_PATH/xxxx/code/i2vgen-xl-diffusers/data/alldata_240722.json \
#     --image_dir $ROOT_PATH/xxxx/code/i2vgen-xl-diffusers/data/gif_data \
#     --log_path logs/$OUTDIR.txt \
#     --output_dir sd-model-finetuned/$OUTDIR

# 单机多卡 半精度
# conda run -n  diffusers \
# accelerate launch --multi_gpu  --num_processes=8 --num_machines=1  --main_process_port=20008\
#     train_multitaskpretrain.py \
#     --init_from_pretrained_2d=$ROOT_PATH/xxxx/data/torch_cache/ali-vilab-i2vgen-xl/models--ali-vilab--i2vgen-xl/snapshots/39e1979ea27be737b0278c06755e321f2b4360d5 \
#     --dataloader_num_workers=8 \
#     --resolution=256 \
#     --train_batch_size=1 \
#     --max_train_steps=1000000 \
#     --mixed_precision bf16 \
#     --gradient_accumulation_steps=6 \
#     --checkpointing_steps=1000 \
#     --dataset_file $ROOT_PATH/xxxx/code/dataset_cartoon_v2/new_data/k84_pretrain_all_rm_visualpathprefix_0608.json \
#     --image_dir $ROOT_PATH/xxxx/code/i2vgen-xl-diffusers/data/gif_data \
#     --log_path logs/$OUTDIR.txt \
#     --output_dir sd-model-finetuned/$OUTDIR

    # --dataset_file $ROOT_PATH/xxxx/code/dataset_cartoon_v2/new_data/k8_pretrain_all.json \
    # --image_dir $ROOT_PATH/xxxx/code/i2vgen-xl-diffusers/tools/cache \

# pretrain data 
# k84_pretrain_all_rm_visualpathprefix_0608.json 

# finetune data
# k8_finetune_all.json # 同



# ------------------------------------------------------------------------------------------------------------------------------------   基于 golden 数据
# 单机多卡 全精度
export OUTDIR=logs_multitaskpretrain_golden
nohup tensorboard --logdir=$ROOT_PATH/xxxx/code/i2vgen-xl-diffusers/sd-model-finetuned/$OUTDIR --port 8080 --bind_all &
conda run -n  diffusers \
accelerate launch --multi_gpu  --num_processes=8 --num_machines=1  --main_process_port=20008\
    train_multitaskpretrain.py \
    --init_from_pretrained_2d=$ROOT_PATH/xxxx/data/torch_cache/ali-vilab-i2vgen-xl/models--ali-vilab--i2vgen-xl/snapshots/39e1979ea27be737b0278c06755e321f2b4360d5 \
    --dataloader_num_workers=8 \
    --resolution=256 \
    --train_batch_size=1 \
    --max_train_steps=500000 \
    --gradient_accumulation_steps=6 \
    --resume_from_checkpoint=latest \
    --checkpointing_steps=1000 \
    --dataset_file $ROOT_PATH/xxxx/code/i2vgen-xl-diffusers/data/golden_multitask_data.json \
    --image_dir $ROOT_PATH/xxxx/code/i2vgen-xl-diffusers/data/gif_data \
    --log_path logs/$OUTDIR.txt \
    --output_dir sd-model-finetuned/$OUTDIR