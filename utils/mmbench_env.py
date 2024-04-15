import sys
sys.path.append("./virtualhome/simulation")
from unity_simulator.comm_unity import UnityCommunication

class MMBenchEnv:
    def __init__(self):
        self.comm = UnityCommunication()
    
    def init_environment(self, graph):
        scene_id = graph['scene_id']
        init_graph = graph['save_graph']
        
        try:
            reset_success = self.comm.reset(scene_id)
            _, g = self.comm.environment_graph()
            max_node_id = max([n['id'] for n in g['nodes']])
            self._separate_new_ids_graph(init_graph, max_node_id)
            s, m = self.comm.expand_scene(init_graph)
        except Exception as e:
            print("An exception occurred while loading the environment\n")
            print(e)
    
        return

    def add_character(self, init_room, character = 'chars/Male2'):
        try:
            self.comm.add_character(character, initial_room=init_room)
        except Exception as e:
            print("An exception occurred while loading the character.\n")
            print(e)
        return 

    def _separate_new_ids_graph(self, graph, max_id):
        for node in graph['nodes']:
            if node['id'] > max_id:
                node['id'] = node['id'] - max_id + 1000
        for edge in graph['edges']:
            if edge['from_id'] > max_id:
                edge['from_id'] = edge['from_id'] - max_id + 1000
            if edge['to_id'] > max_id:
                edge['to_id'] = edge['to_id'] - max_id + 1000
        return