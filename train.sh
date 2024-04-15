accelerate launch \
--config_file=configs/accelerate_config_fsdp.yaml \
--num_processes=4 train.py \
--num_epochs 10 \
--task_name evaE \
--visual_feat evaE_feat \
--lora \
--report_to_wandb
