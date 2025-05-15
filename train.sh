 HF_HUB_OFFLINE=1 HF_HOME=/home/wuwl/data/huggingface HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16 \
 CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
 python code/main.py \
 --do_train \
 --config_path \
 ./code/config.yaml \
 --model_save_path \
 ./output/ \
 --result_save_path \
 ./saisresult/