 HF_HUB_OFFLINE=1 HF_HOME=/home/wuwl/data/huggingface HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16 \
 CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
 python code/main.py \
 --do_predict \
 --model_save_path \
 ./output/20250514_224449/ \
 --result_save_path \
 ./saisresult/