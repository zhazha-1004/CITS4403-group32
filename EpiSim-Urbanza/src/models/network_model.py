
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import random


class NetworkEpidemicModel:
    
    def __init__(self, N: int = 1000, network_type: str = 'small_world',
                 infection_prob: float = 0.05, recovery_prob: float = 0.10,
                 I0: int = 5, **network_params):
        self.N = N
        self.network_type = network_type
        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob
        self.I0 = I0
        
        self.graph = self._create_network(network_type, N, **network_params)
        
        self.node_states = np.zeros(N, dtype=int)
        initial_infected = random.sample(range(N), min(I0, N))
        for node in initial_infected:
            self.node_states[node] = 1
    
    def _create_network(self, network_type: str, N: int, **params) -> nx.Graph:
        if network_type == 'small_world':
            k = params.get('k', 6)
            p = params.get('p', 0.3)
            return nx.watts_strogatz_graph(N, k, p)
        
        elif network_type == 'scale_free':
            m = params.get('m', 3)
            return nx.barabasi_albert_graph(N, m)
        
        elif network_type == 'random':
            p = params.get('p', 0.01)
            return nx.erdos_renyi_graph(N, p)
        
        else:
            raise ValueError(f"Unsupported network type: {network_type}")
    
    def step(self) -> Tuple[int, int, int]:
        new_states = self.node_states.copy()
        
        infected_nodes = np.where(self.node_states == 1)[0]
        for infected_node in infected_nodes:
            neighbors = list(self.graph.neighbors(infected_node))
            for neighbor in neighbors:
                if self.node_states[neighbor] == 0:
                    if random.random() < self.infection_prob:
                        new_states[neighbor] = 1
        
        for infected_node in infected_nodes:
            if random.random() < self.recovery_prob:
                new_states[infected_node] = 2
        
        self.node_states = new_states
        S_count = np.sum(self.node_states == 0)
        I_count = np.sum(self.node_states == 1)
        R_count = np.sum(self.node_states == 2)
        
        return S_count, I_count, R_count
    
    def simulate(self, days: int = 120) -> pd.DataFrame:
        self.node_states = np.zeros(self.N, dtype=int)
        initial_infected = random.sample(range(self.N), min(self.I0, self.N))
        for node in initial_infected:
            self.node_states[node] = 1
        results = {
            'day': [],
            'S': [],
            'I': [],
            'R': [],
            'S_prop': [],
            'I_prop': [],
            'R_prop': [],
            'new_infections': []
        }
        
        S_count = np.sum(self.node_states == 0)
        I_count = np.sum(self.node_states == 1)
        R_count = np.sum(self.node_states == 2)
        
        results['day'].append(0)
        results['S'].append(S_count)
        results['I'].append(I_count)
        results['R'].append(R_count)
        results['S_prop'].append(S_count / self.N)
        results['I_prop'].append(I_count / self.N)
        results['R_prop'].append(R_count / self.N)
        results['new_infections'].append(I_count)
        for day in range(1, days + 1):
            prev_I = np.sum(self.node_states == 1)
            S_count, I_count, R_count = self.step()
            new_infections = I_count - prev_I + (R_count - results['R'][-1])
            
            results['day'].append(day)
            results['S'].append(S_count)
            results['I'].append(I_count)
            results['R'].append(R_count)
            results['S_prop'].append(S_count / self.N)
            results['I_prop'].append(I_count / self.N)
            results['R_prop'].append(R_count / self.N)
            results['new_infections'].append(max(0, new_infections))
            
            if I_count == 0:
                break
        
        return pd.DataFrame(results)
    
    def get_network_properties(self) -> Dict[str, Any]:
        properties = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'average_degree': np.mean([d for n, d in self.graph.degree()]),
            'network_type': self.network_type
        }
        
        if self.N <= 1000:
            properties['clustering_coefficient'] = nx.average_clustering(self.graph)
        
        if self.N <= 1000 and nx.is_connected(self.graph):
            properties['average_path_length'] = nx.average_shortest_path_length(self.graph)
        
        return properties
    
    def __str__(self) -> str:
        props = self.get_network_properties()
        return (f"Network Epidemic Model: {self.network_type}\n"
                f"N={self.N}, edges={props['edges']}, "
                f"avg_degree={props['average_degree']:.2f}")


def compare_network_types(N: int = 1000, days: int = 60) -> Dict[str, Any]:
    network_types = {
        'small_world': {'k': 6, 'p': 0.3},
        'scale_free': {'m': 3},
        'random': {'p': 0.006}
    }
    
    results = {}
    
    for net_type, params in network_types.items():
        print(f"Simulating {net_type} network...")
        
        model = NetworkEpidemicModel(
            N=N, 
            network_type=net_type,
            infection_prob=0.05,
            recovery_prob=0.10,
            I0=5,
            **params
        )
        
        all_results = []
        for run in range(5):
            sim_results = model.simulate(days)
            all_results.append(sim_results)
        avg_results = pd.concat(all_results).groupby('day').mean().reset_index()
        
        results[net_type] = {
            'simulation_results': avg_results,
            'network_properties': model.get_network_properties(),
            'peak_infection': avg_results['I_prop'].max(),
            'time_to_peak': avg_results.loc[avg_results['I_prop'].idxmax(), 'day'],
            'final_size': 1 - avg_results['S_prop'].iloc[-1]
        }
    
    return results


class NetworkInterventionScenario:
    
    def __init__(self, base_model: NetworkEpidemicModel, intervention_fraction: float = 0.1):
        self.base_model = base_model
        self.intervention_fraction = intervention_fraction
        
        self.intervened_graph = self._apply_intervention()
    
    def _apply_intervention(self) -> nx.Graph:
        graph = self.base_model.graph.copy()
        
        degree_sequence = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
        
        nodes_to_remove = int(len(degree_sequence) * self.intervention_fraction)
        high_degree_nodes = [node for node, degree in degree_sequence[:nodes_to_remove]]
        
        graph.remove_nodes_from(high_degree_nodes)
        
        return graph
    
    def simulate(self, days: int = 120) -> pd.DataFrame:
        intervened_model = NetworkEpidemicModel(
            N=self.intervened_graph.number_of_nodes(),
            network_type=self.base_model.network_type,
            infection_prob=self.base_model.infection_prob,
            recovery_prob=self.base_model.recovery_prob,
            I0=min(self.base_model.I0, self.intervened_graph.number_of_nodes())
        )
        
        intervened_model.graph = self.intervened_graph
        
        return intervened_model.simulate(days)
