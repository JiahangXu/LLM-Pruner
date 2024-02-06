prune_ckpt_path='llama2_prune_4.5b_mix'
tune_ckpt_path='llama2_4.5b_mix_gpt4alpaca'

# echo "[START] - Start Pruning Model"
# CUDA_VISIBLE_DEVICES=0 python hf_prune.py --base_model meta-llama/Llama-2-7b-hf --pruning_ratio 0.435 --device cpu  --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_mix --save_model
# echo "[FINISH] - Finish Pruning Model"

echo "[START] - Start Tuning"
CUDA_VISIBLE_DEVICES=0 python post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path vicgalle/alpaca-gpt4 --output_dir tune_log/$tune_ckpt_path --wandb_project llama_tune --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64
echo "[FINISH] - Finish Prune and Post-Training."
echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"
