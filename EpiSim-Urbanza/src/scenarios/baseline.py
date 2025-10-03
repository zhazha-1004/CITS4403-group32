
import pandas as pd
from typing import Dict, Any
from ..models.sir_model import SIRModel
from ..models.seir_model import SEIRModel
from ..analysis.metrics import calculate_all_metrics


class BaselineScenario:
    
    def __init__(self, N: int = 10000, beta: float = 0.30, gamma: float = 0.10,
                 I0: int = 10, R0: int = 0):
        self.N = N
        self.beta = beta
        self.gamma = gamma
        self.I0 = I0
        self.R0 = R0
        
        self.model = SIRModel(N=N, beta=beta, gamma=gamma, I0=I0, R0=R0)
        
    def run_simulation(self, days: int = 120) -> pd.DataFrame:
        results = self.model.simulate(days)
        return results
    
    def get_metrics(self, results: pd.DataFrame = None) -> Dict[str, Any]:
        if results is None:
            results = self.run_simulation()
        
        metrics = calculate_all_metrics(results, self.N)
        metrics['scenario_type'] = 'baseline'
        metrics['R0'] = self.model.get_basic_reproduction_number()
        
        return metrics
    
    def get_description(self) -> str:
        return (f"Baseline scenario: No interventions\n"
                f"Parameters: N={self.N}, β={self.beta}, γ={self.gamma}, "
                f"R0={self.model.get_basic_reproduction_number():.2f}")


class BaselineSEIRScenario:
    
    def __init__(self, N: int = 10000, beta: float = 0.30, sigma: float = 0.20,
                 gamma: float = 0.10, I0: int = 10, E0: int = 0, R0: int = 0):
        self.N = N
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.I0 = I0
        self.E0 = E0
        self.R0 = R0
        
        self.model = SEIRModel(N=N, beta=beta, sigma=sigma, gamma=gamma,
                              I0=I0, E0=E0, R0=R0)
        
    def run_simulation(self, days: int = 120) -> pd.DataFrame:
        results = self.model.simulate(days)
        return results
    
    def get_metrics(self, results: pd.DataFrame = None) -> Dict[str, Any]:
        if results is None:
            results = self.run_simulation()
        
        metrics = calculate_all_metrics(results, self.N)
        metrics['scenario_type'] = 'seir_baseline'
        metrics['R0'] = self.model.get_basic_reproduction_number()
        
        return metrics
    
    def get_description(self) -> str:
        return (f"SEIR baseline scenario: With incubation period, no interventions\n"
                f"Parameters: N={self.N}, β={self.beta}, σ={self.sigma}, γ={self.gamma}, "
                f"R0={self.model.get_basic_reproduction_number():.2f}")


def run_baseline_comparison() -> Dict[str, Any]:
    sir_scenario = BaselineScenario()
    seir_scenario = BaselineSEIRScenario()
    
    sir_results = sir_scenario.run_simulation()
    seir_results = seir_scenario.run_simulation()
    
    sir_metrics = sir_scenario.get_metrics(sir_results)
    seir_metrics = seir_scenario.get_metrics(seir_results)
    
    return {
        'sir_results': sir_results,
        'seir_results': seir_results,
        'sir_metrics': sir_metrics,
        'seir_metrics': seir_metrics,
        'comparison': {
            'sir_peak_day': sir_metrics['time_to_peak'],
            'seir_peak_day': seir_metrics['time_to_peak'],
            'peak_delay': seir_metrics['time_to_peak'] - sir_metrics['time_to_peak'],
            'sir_peak_proportion': sir_metrics['peak_infection_proportion'],
            'seir_peak_proportion': seir_metrics['peak_infection_proportion'],
            'sir_final_size': sir_metrics['final_epidemic_size'],
            'seir_final_size': seir_metrics['final_epidemic_size']
        }
    }
