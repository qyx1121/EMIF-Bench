# EMIF_Bench

## Download VirtualHome
```
$ cd ./virtualhome/simulation/unity_simulator/
$ wget http://virtual-home.org//release/simulator/v2.0/v2.2.4/linux_exec.zip
$ unzip linux_exec.zip
```
You can find more information about how to launch VirtualHome here: [VirtualHome](https://github.com/xavierpuigf/virtualhome?tab=readme-ov-file).

## Dataset
Download our dataset from [Google drive](https://drive.google.com/drive/folders/1HcZ_82-Tlnj05EjWAX14BWh7qtzPu8f2?usp=sharing), and place them in the following structure in:
```
data
├── mm_bench
│   ├── full_data
│   │   ├── data_graph_v9.json
│   │   ├── train_v9.json
│   │   └── val_v9.json
│   └── visual_feats
│       ├── evaB_feat.pt
│       ├── evaE_feat.pt
│       └── evaL_feat.pt
```

## Train & Evaluation
To train the planner we proposed in our manuscript, you can run the code as follows:
```
accelerate launch \
    --config_file=configs/accelerate_config_fsdp.yaml \
    --num_processes=4 train.py \
    --num_epochs 10 \
    --task_name evaE \
    --visual_feat evaE_feat \
    --pretrained_model_path /mnt/hdd1/llama2_hf/llama-2-7b
    --lora \
```

After you have trained the planner, you can run the code as follows:
```
accelerate launch \
    --config_file=configs/accelerate_config_fsdp.yaml \
    --num_processes=4 train.py \
    --num_epochs 10 \
    --task_name evaE \
    --visual_feat evaE_feat \
    --pretrained_model_path /mnt/hdd1/llama2_hf/llama-2-7b
    --lora \
    --evaluate \ 
    --load_checkpoint /mnt/hdd1/ckpt/weights_epoch_9.pt
```
This process of evaluation will generate a result file, which includes the decision path output by the planner, under the "results" folder.

After you get the output decision path, you can run following code:
```
python utils/evaluator.py \
    --pred_path results/res.json \
    --launch_path utils/launch.sh \
    --test_data ./data/mm_bench/full_data/val_v9.json \
    --data_graph ./data/mm_bench/full_data/data_graph_v9.json \
    --output_path ./results/eval_res.json
```
to calculate the task success rate.
