import os
import re
import sys
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger

from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType

from data.data_loading import MM_Bench, mm_collate_fn
from model.modelling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, LlamaConfig,get_cosine_schedule_with_warmup
import wandb
import time
from collections import OrderedDict
import logging
import torch.nn.functional as F

from utils.misc import extract_decoder_hidden_states


os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__, log_level="INFO")


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model_path", type=str, default="/mnt/hdd1/llama2_hf/llama-2-7b")
    parser.add_argument("--llava_model_path", type = str)
    parser.add_argument("--data_path", type=str, default="data/mm_bench")
    parser.add_argument("--task_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=1121)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_generate_plans", type=int, default=20)
    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--visual_feat", type=str, default="evaL_feat")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--warmup_steps_ratio", default=0.01, type=float)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--eval_step", default=1000)
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="mm_eai_bench"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="mm_eai_bench",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument("--lora", action="store_true")

    return parser

def train_one_epoch(args, model, epoch, train_loader, lr_scheduler, tokenizer, device_id, accelerator, optimizer):
    num_batches_per_epoch = len(train_loader)
    total_training_steps = num_batches_per_epoch * args.num_epochs
    model.train()
    for idx, batch in tqdm(enumerate(train_loader),
                           disable=args.rank!=0,
                           total=total_training_steps,
                           initial=(epoch*num_batches_per_epoch)
                           ):
        global_step = idx + epoch * num_batches_per_epoch

        text_input = batch["text_input"]
        text_gts = batch["text_gts"]
        gt_indexes = batch["gt_index"]
        instr_feats = batch["instr_feats"]
        plan_feats = batch["plan_feats"]
        env_object_feats = batch["env_object_feats"][0]
        instr_length = batch["instr_length"]
        plan_prefab_nums = batch["plan_prefab_nums"]

        text_input = [t_input + t_gt for t_input, t_gt in zip(text_input, text_gts)]

        tokens = tokenizer(
            text_input,
            return_tensors = "pt",
            padding=True
        )

        input_id = tokens.input_ids
        attention_mask = tokens.attention_mask
        input_id = torch.cat([input_id, torch.ones(input_id.shape[0], 1) * tokenizer.eos_token_id], dim = 1).to(torch.int64)
        attention_mask = torch.cat([attention_mask, torch.ones(attention_mask.shape[0], 1)], dim = 1).to(torch.int64)
        
        labels = input_id.clone()
        labels[labels== 0] = -100
        for i in range(labels.shape[0]):
            end_idx = torch.where(labels[i] == 29901)[0][-1]
            labels[i][:end_idx + 1] = -100
        
        pred_idx = torch.where(labels == tokenizer.convert_tokens_to_ids("<cls>"))
        object_idx = torch.where(input_id == tokenizer.convert_tokens_to_ids("<cls>"))
        
        with accelerator.autocast():
            input_embeds = model.get_multimodal_embeddings(
                input_id.to(device_id),
                instr_feats, plan_feats,
                object_idx, instr_length, plan_prefab_nums
            )
            output = model(
                inputs_embeds = input_embeds,
                attention_mask = attention_mask.to(device_id),
                labels = labels,
                env_object_feats = env_object_feats,
                env_gt_idx = gt_indexes, pred_idx = pred_idx
            )
            loss_np = output.loss_np
            loss_cl = output.loss_cl

            loss = loss_np + loss_cl
            if accelerator.mixed_precision == "fp16":
                accelerator.backward(loss.to(device_id))
            else:
                accelerator.backward(loss)

            if args.report_to_wandb:
                wandb.log(
                    {
                        "loss_np": loss_np.item(),
                        "loss_cl": loss_cl.item(),
                        "global_step": global_step // args.gradient_accumulation_steps,
                        "lr": optimizer.param_groups[0]['lr']
                    },
                    commit=True,
                    )
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if (global_step + 1) % 2 == 0:
            logger.info(f"step: {global_step}/{total_training_steps}, loss_np: {loss_np.item()}, loss_cl: {loss_cl.item()}")


def evaluate(args, model, val_loader, tokenizer, device_id, accelerator):
    
    model.eval()
    res = {}
    for idx, batch in tqdm(enumerate(val_loader)):

        item_id = batch["id"][0]
        text_input = batch["text_input"][0]
        #text_gts = batch["text_gts"]
        gt_indexes = batch["gt_index"]
        instr_feats = batch["instr_feats"]
        #plan_feats = batch["plan_feats"]
        env_objects = batch["env_objects"][0]
        env_object_feats = batch["env_object_feats"][0]
        instr_length = batch["instr_length"]
        plan_prefab_nums = batch["plan_prefab_nums"]
        
        output_plans = []
        
        for i in range(args.max_generate_plans):
            with torch.no_grad():

                tokens = tokenizer(
                    text_input,
                    return_tensors = "pt",
                    padding=True
                )

                input_id = tokens.input_ids
                attention_mask = tokens.attention_mask
            
                object_idx = torch.where(input_id == tokenizer.convert_tokens_to_ids("<cls>"))
                input_embeds = model.get_multimodal_embeddings(
                    input_id.to(device_id),
                    instr_feats, None,
                    object_idx, instr_length, plan_prefab_nums
                )

                output = model.generate(
                    inputs_embeds = input_embeds,
                    attention_mask =  attention_mask.to(device_id),
                    output_hidden_states = True,
                    return_dict_in_generate = True,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens = 9,
                    output_scores = True,
                    env_object_feats = env_object_feats,
                    num_beams = 3       
                    )
                
                output_plan = tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)[0]
                if output_plan == "done":
                    break

                last_hidden_states = extract_decoder_hidden_states(output).squeeze()                
                last_hidden_states = torch.stack([output['hidden_states'][i][-1][:, -1, :][0] for i in range(len(output['hidden_states']))])             
                object_idx = torch.where(output['sequences'].to('cpu') == tokenizer.convert_tokens_to_ids("<cls>"))
                last_hidden_states = last_hidden_states[object_idx[1] - 1]
                last_hidden_states = model.mm_head(last_hidden_states)
                last_hidden_states = last_hidden_states.to(env_object_feats.device)
                sim = torch.matmul(last_hidden_states, env_object_feats.t())
                pred_index = torch.argmax(sim, dim = -1)
                pred_object_feat = env_object_feats[pred_index]           

                ### rewrite instruction ###
                pos = re.search('\]', text_input).start()
                if text_input[pos - 1] == '[':
                    text_input = text_input[:pos] + output_plan + text_input[pos:]
                else:
                    text_input = text_input[:pos] + ", " + output_plan + text_input[pos:]

                for idx in pred_index:
                    pred_object = env_objects[idx]
                    output_plan = output_plan.replace("<cls>", f"({pred_object})", 1)
                output_plans.append(output_plan)
                instr_feats[0] = torch.cat((instr_feats[0], pred_object_feat.to(instr_feats[0].device)), dim = 0)
        
        res[item_id] = output_plans
    save_dir = os.path.join("results", args.task_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "res.json")
    json.dump(res, open(save_path, "w"))
    return 


def main(args):
    
    accelerator = Accelerator()

    device_id = accelerator.device
    device_map = "auto"
    
    config = LlamaConfig.from_pretrained(args.pretrained_model_path)
    
    if "evaB" in args.visual_feat:
        config.vision_dim = 512 
    elif "evaL" in args.visual_feat:
        config.vision_dim = 768
    else:
        config.vision_dim = 1024
    model = LlamaForCausalLM.from_pretrained(args.pretrained_model_path, config=config, device_map=device_map)
    tokenizer = LlamaTokenizer.from_pretrained(args.pretrained_model_path)
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({'pad_token': '<unk>'})
    tokenizer.add_tokens(['<cls>'])
    model.resize_token_embeddings(len(tokenizer))
    model.tokenizer = tokenizer

    accelerator.wait_for_everyone()
    args.distributed_type = accelerator.distributed_type

    random_seed(args.seed, args.rank)

    prompt = "Please output the next one plan. Instruction: {} Completed plans: [{}] Next plan: "
    dataset = MM_Bench(args.data_path, prompt, args.visual_feat)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=mm_collate_fn)

    val_dataset = MM_Bench(args.data_path, prompt, args.visual_feat, "val")
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=mm_collate_fn)

    if args.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            r=8, 
            lora_alpha=16, 
            lora_dropout=0.05, 
            target_modules=["q_proj", "v_proj"]
            )
        model = get_peft_model(model, lora_config)

    if args.load_checkpoint != "":
        model.load_state_dict(torch.load(args.load_checkpoint, map_location = "cpu"), strict=False)

    if "llava" in args.task_name:
        model.load_state_dict(torch.load(args.llava_model_path, map_location = "cpu"), strict=False)
    
    for n, p in model.named_parameters():
        if "visual_proj" in n or "mm_head" in n or "lm_head" in n:
            p.requires_grad = True

    trainable_params = [n for n,p in model.named_parameters() if p.requires_grad]
    logger.info(f"Total Trainable param: {(sum(p.numel() for p in model.parameters() if p.requires_grad)) / 1e9:.6f} B")
    logger.info(f"Trainable parameters: {trainable_params}")

    total_training_steps = len(dataloader) * args.num_epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    args.warmup_steps = total_training_steps * args.warmup_steps_ratio if args.warmup_steps_ratio is not None else args.warmup_stepsps

    lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps,
            num_training_steps=total_training_steps // args.gradient_accumulation_steps,
        )

    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )
    
    model, optimizer, lr_scheduler, dataloader = accelerator.prepare(model, optimizer, lr_scheduler, dataloader)
    val_dataloader = accelerator.prepare(val_dataloader)

    if args.evaluate:
        print("Evaluating...\n")
        evaluate(args, model, val_dataloader, tokenizer, device_id, accelerator)
    else:
        for epoch in range(args.num_epochs):
            logger.info(f"Start epoch {epoch}")
            train_one_epoch(args, model, epoch, dataloader, lr_scheduler, tokenizer, device_id, accelerator, optimizer)
            unwrap_model = accelerator.unwrap_model(model)
            ori_params = unwrap_model.state_dict()
            save_params = OrderedDict()
            for k, v in unwrap_model.named_parameters():
                if v.requires_grad:
                    save_params[k] = ori_params[k]
            if args.task_name != "":
                save_dir = os.path.join("data/mm_bench/ckpts", args.task_name)
                os.makedirs(save_dir, exist_ok=True)
                torch.save(save_params, f"data/mm_bench/ckpts/{args.task_name}/weights_epoch_{epoch}.pt")


if __name__  == "__main__":
    args = parse_args().parse_args()
    args.run_name = args.task_name
    main(args)