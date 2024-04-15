import os
import re
import sys
import json
import time
import signal
import subprocess
sys.path.append("./virtualhome")
from simulation.unity_simulator.comm_unity import UnityCommunication
from demo.utils_demo import *

from utils import convert_nl_plans_to_script
from tqdm import tqdm
from mmbench_env import MMBenchEnv

import argparse

class MMBenchEvaluator:
    def __init__(self, graph_path, launch_path):
        self.all_graph = json.load(open(graph_path))
        self.launch_path = launch_path
        self.env = MMBenchEnv()
        self.cur_graph = None
        self.init_graph = None
        self.init_room = None
        self.start_server()

        self.agent = UnityCommunication()

    def init_environment(self):
        time.sleep(5)
        self.env.init_environment(self.init_graph)
        self.env.add_character(self.init_room)
        # time.sleep(5)
    
    def evaluate(self, item, pred_plans, gpt_eval = False):
        gid = item['gid']
        goal_condition = item['goal_condition']
        self.init_room = item['init_room']
        self.init_graph = self.all_graph[str(gid)]
        self.init_environment()
        _, self.cur_graph = self.env.comm.environment_graph()

        scripts = []
        sign = -1
        for p in pred_plans:
            script = self.nl_plans_to_script(p, gpt_eval)
            if script is None:
                continue
            sign, message = self.step(script) 
            if sign == 1:
                scripts.append(script)
                pass
            else:
                if sign == -1:
                    sign, message = self.agent.render_script(script=scripts,
                                            processing_time_limit=120,
                                            find_solution=False,
                                            image_width=320,
                                            image_height=240,  
                                            skip_animation=False,
                                            recording=True,
                                            save_pose_data=True,
                                            file_name_prefix='relax')
                else:
                    self.init_environment()
                    scripts.append(script)
                    try:
                        sign, message = self.agent.render_script(script=scripts,
                                                processing_time_limit=120,
                                                find_solution=False,
                                                image_width=320,
                                                image_height=240,  
                                                skip_animation=False,
                                                recording=True,
                                                save_pose_data=True,
                                                file_name_prefix='relax')
                    except:
                        sign = 0
                        self.restart()

                    if sign == 1:
                        self.update_graph(script)
                    else: 
                        scripts.remove(script)
                        self.init_environment()
                        try:
                            sign, message = self.agent.render_script(script=scripts,
                                                processing_time_limit=120,
                                                find_solution=False,
                                                image_width=320,
                                                image_height=240,  
                                                skip_animation=False,
                                                recording=True,
                                                save_pose_data=True,
                                                file_name_prefix='relax')
                        except:
                            self.restart()
                            break

        return self.compute_success_rate(goal_condition)
    
    def judge_goal_condition(self, goal_condition)->bool:
        goal_state_1 = ["CLOSED", "OPEN", "ON", "OFF", "DRUNK", "WARMED"]
        goal_state_2 = ["CLOSE", "HELD", "SITTING"]
        goal_state_3 = ["ON", "INSIDE"]
        pattern = "\('([^']*)', (\d+)\)"
        objs = re.findall(pattern, goal_condition)
        if len(objs) == 2:
            for s in goal_state_3:
                if s in goal_condition:
                    id_1, id_2 = int(objs[0][1]), int(objs[1][1])
                    if {'from_id': id_1, 'to_id': id_2, 'relation_type': s} in self.cur_graph['edges']:
                        return True

        elif len(objs) == 1:
            id_1 = int(objs[0][1])
            node = find_nodes(self.cur_graph, id = id_1)[0]
            for s in goal_state_1:
                if s in goal_condition:
                    if s in node['states']:
                        return True
                
            for s in goal_state_2:
                if s in goal_condition:
                    if s == "HELD":
                        rel_l = {'from_id': 1, 'to_id': id_1, 'relation_type': 'HOLDS_RH'}
                        rel_r = {'from_id': 1, 'to_id': id_1, 'relation_type': 'HOLDS_LH'}
                        if rel_l in self.cur_graph['edges'] or rel_r in self.cur_graph['edges']:
                            return True
                    else:
                        if {'from_id': 1, 'to_id': id_1, 'relation_type': s} in self.cur_graph['edges']:
                            return True
        return False

    def compute_success_rate(self, goal_conditions):
        total_subgoals = len(goal_conditions)
        success = 0
        for g in goal_conditions:
            if self.judge_goal_condition(g):
                success += 1
        
        return {
            "total": total_subgoals,
            "success": success,
            "rate": success / total_subgoals    
            }
    
    
    def nl_plans_to_script(self, nl_plan, gpt_eval = False):
        script = convert_nl_plans_to_script(self.cur_graph, nl_plan, gpt_eval)
        return script

    def start_server(self):
        self.process = subprocess.Popen([self.launch_path], shell=False)
        time.sleep(5)
    
    def restart(self):
        #self.process.kill()
        os.kill(self.process.pid + 1, signal.SIGKILL)
        self.process = subprocess.Popen([self.launch_path], shell=False)
        time.sleep(8)
        self.init_environment()

    def step(self, script):
        ### step-by-step     
        ### 0: fail; 1: success; -1: exception
        try:
            success, message = self.agent.render_script(script=[script],
                                        processing_time_limit=60,
                                        find_solution=False,
                                        image_width=320,
                                        image_height=240,  
                                        skip_animation=False,
                                        recording=True,
                                        save_pose_data=True,
                                        file_name_prefix='relax')

            if success:
                self.update_graph(script)
        
            else:
                return 0, message
            
            return 1, message

        except:
            self.restart()
        
        return -1, None  

    def update_graph(self, script): 
        s, g = self.env.comm.environment_graph()
        
        ### special states ###
        drunk_ids = [n['id'] for n in self.cur_graph['nodes'] if 'DRUNK' in n['states']]
        warm_ids  = [n['id'] for n in self.cur_graph['nodes'] if 'WARMED' in n['states']]

        self.cur_graph = g

        for idx in drunk_ids:
            node = find_nodes(self.cur_graph, id = idx)[0]
            if "DRUNK" not in node['states']:
                node['states'].append("DRUNK")
        for idx in warm_ids:
            node = find_nodes(self.cur_graph, id = idx)[0]
            if "WARMED" not in node['states']:
                node['states'].append("WARMED")

        ### for drink ###
        pattern = r'\((.*?)\)'
        obj_id = re.findall(pattern, script)
        if len(obj_id) == 0:
            return
        obj_id = int(obj_id[0])

        if "Drink" in script:
            node = find_nodes(self.cur_graph, id = obj_id)[0]
            node['states'].append("DRUNK")
        
        elif "Switchon" in script:
            if "microwave" in script or "toaster" in script:
                rels = find_edges_rel(self.cur_graph, "INSIDE", fr_id = None, to_id = obj_id)
                for r in rels:
                    node = find_nodes(self.cur_graph, id = r['from_id'])[0]       
                    if "WARMED" not in node['states']:
                        node['states'].append("WARMED")
            elif "stove" in script:
                rels = find_edges_rel(self.cur_graph, "INSIDE", fr_id =None, to_id = obj_id) + \
                    find_edges_rel(self.cur_graph, "ON", fr_id =None, to_id = obj_id)
                for r in rels:
                    node = find_nodes(self.cur_graph, id = r['from_id'])[0]
                    if "WARMED" not in node['states']:
                        node['states'].append("WARMED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, help="Path to the output.")
    parser.add_argument("--launch_path", type = str, help="Script file to launch VirtualHome.")
    parser.add_argument("--test_data", type = str, help="Path to the test set.")
    parser.add_argument("--data_graph", type = str, help ="Path to the data graph.")
    parser.add_argument("--output_path", type = str, help= "The output path of the results")
    args = parser.parse_args()

    evaluator = MMBenchEvaluator(args.data_graph, args.launch_path)
    val_data = json.load(open(args.test_data))
    preds = json.load(open(args.pred_path))

    res = {}

    for k, v in tqdm(enumerate(preds.items())):
        gt = val_data[k]
        try:
            success_rate = evaluator.evaluate(gt, v)
        except:
            json.dump(res, open(args.output_path, "w"))

        res[k] = v

    success_tasks = 0
    for k,v in res.items():
        success_tasks += v['rate'] == 1.0

    print("Success rate is {}".format(success_tasks / len(val_data)))
    json.dump(res, open(args.output_path, "w"))







        

        

