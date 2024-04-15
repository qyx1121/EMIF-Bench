import json
import os
import os.path as osp
import re
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MM_Bench(Dataset):

    def __init__(self, data_dir, prompt, visual_feat, split = "train"):
        super().__init__()
        self.object_feats = torch.load(osp.join(data_dir, f"visual_feats/{visual_feat}.pt"))
        self.data = json.load(open(osp.join(data_dir, f"full_data/{split}_v9.json")))
        self.data_graph = json.load(open(osp.join(data_dir, "full_data/data_graph_v9.json")))
        self.prompt = prompt
        self.object_token = "<cls>"
        self.split = split
    
    def __getitem__(self, index):
        item = self.data[index]
        data_id = item['id']
        g_id = item['gid']
        init_graph = self.data_graph[str(g_id)]['save_graph']
        final_graph = self.data_graph[str(g_id)]['final_graph']
        nl_instruction = item['instruction']
        nl_plans = item['plans']
        mm_instruction = item['mm_instruction']
        mm_plans = item['mm_plans']
        
        pattern = "\('([^']*)', (\d+)\)"
        all_prefabs = list(set(re.findall(pattern, " ".join(mm_plans))))
        prefab2feats = {}
        for it in all_prefabs:
            prefab = it[0]
            prefab2feats[prefab] = self.object_feats[prefab]
        
        all_instr_prefabs = re.findall(pattern, mm_instruction)
        instruct_prefabs = []
        for i in all_instr_prefabs:
            mm_instruction = mm_instruction.replace(str((i[0], int(i[1]))), self.object_token)
            instruct_prefabs.append(i[0])
        
        ### store all object features in current environment
        env_object_feats = []
        env_objects = []
        for n in final_graph['nodes']:
            if n['prefab_name'] in self.object_feats.keys() and n['prefab_name'] not in env_objects:
                env_object_feats.append(self.object_feats[n['prefab_name']])
                env_objects.append(n['prefab_name'])
        env_object_feats = torch.stack(env_object_feats)
    
        ### object features in instruction
        instr_object_feats = []
        for p_name in instruct_prefabs:
            assert p_name in self.object_feats.keys()
            instr_object_feats.append(self.object_feats[p_name])
        instr_object_feats = torch.stack(instr_object_feats)
        
        ### generate each samples
        output_plans = []
       
        for i in range(len(mm_plans)):
            plan_feats = []
            mm_plan = ", ".join(mm_plans[ :i]) if i!=0 else ""
            
            gt = mm_plans[i]

            if gt.lower() != "done":
                gt_plan_prefab = re.findall(pattern, mm_plans[i])
                gt_prefab_index = []
                for p in gt_plan_prefab:
                    try:
                        gt = gt.replace(str((p[0], int(p[1]))), self.object_token)
                        gt_prefab_index.append(env_objects.index(p[0]))
                    except:
                        print("hold")
            
            else:
                gt_prefab_index = -1
            
            plan_prefabs = re.findall(pattern, mm_plan)
            for p in plan_prefabs:
                mm_plan = mm_plan.replace(str((p[0], int(p[1]))), self.object_token)
                plan_feats.append(self.object_feats[p[0]])
            
            if len(plan_feats) != 0:
                plan_feats = torch.stack(plan_feats)
            else:
                plan_feats = None
            instrution = self.prompt.format(mm_instruction, mm_plan)

            original_plan = ", ".join([p for p in nl_plans[ :i]]) +"." if i!=0 else ""
            original_instruction = self.prompt.format(nl_instruction, original_plan)

            item = {
                "instruction": instrution,
                "gt":gt,
                "gt_prefab_index":gt_prefab_index,
                "plan_feats": plan_feats,
                "original_instruction": original_instruction   
                }
            output_plans.append(item)
        
        return {
            "id": data_id,
            "text_input": output_plans,
            "instr_object_feats": instr_object_feats,
            "env_object_feats": env_object_feats,
            "env_objects": env_objects,
            "init_graph": init_graph
        }
    
    def __len__(self):
        return len(self.data)

def mm_collate_fn(batch):
    text_input = []
    ids = []
    gts = []
    gt_prefab_index = []
    instr_feats = []
    plan_feats = []
    original_instructions = []
    env_objects_feats = []
    env_objects = []
    instr_length = []
    plan_prefab_nums = []

    for it in batch:
        #instr_feats += [it['instr_object_feats']] * len(it['text_input'])
        ids.append(it['id'])
        instr_feats.append(it['instr_object_feats'])
        env_objects.append(it['env_objects'])
        env_objects_feats.append(it['env_object_feats'])
        instr_length.append(len(it['text_input']))
        max_prefabs = 0
        for i in it['text_input']:
            text_input.append(i['instruction'])
            gts.append(i['gt'])
            gt_prefab_index.append(i['gt_prefab_index'])
            original_instructions.append(i['original_instruction'])
            plan_feats.append(i['plan_feats'])
            if i['plan_feats'] is not None:
                max_prefabs = max(max_prefabs, i['plan_feats'].shape[0])
        
        plan_prefab_nums.append(max_prefabs)
    
    return {
            "id":ids,
            "text_input": text_input,
            "text_gts": gts,
            "gt_index": gt_prefab_index,
            "instr_feats":instr_feats,
            "plan_feats":plan_feats,
            "env_object_feats": env_objects_feats,
            "env_objects":env_objects,
            "instr_length": instr_length,
            "plan_prefab_nums":plan_prefab_nums
            }