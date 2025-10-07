
import pandas as pd
from typing import Dict, Any, Optional
from ..models.sir_model import SIRModel
from ..analysis.metrics import calculate_all_metrics


class ContactReductionScenario:
    
    def __init__(self, N: int = 10000, baseline_beta: float = 0.30, 
                 reduction_percent: float = 0.30, gamma: float = 0.10,
                 I0: int = 10, R0: int = 0):
        self.N = N
        self.baseline_beta = baseline_beta
        self.reduction_percent = reduction_percent
        self.gamma = gamma
        self.I0 = I0
        self.R0 = R0
        
        self.effective_beta = baseline_beta * (1 - reduction_percent)
        
        self.model = SIRModel(N=N, beta=self.effective_beta, gamma=gamma, I0=I0, R0=R0)
        
    def run_simulation(self, days: int = 120) -> pd.DataFrame:
        results = self.model.simulate(days)
        return results
    
    def get_metrics(self, results: pd.DataFrame = None) -> Dict[str, Any]:
        if results is None:
            results = self.run_simulation()
        
        metrics = calculate_all_metrics(results, self.N)
        metrics['scenario_type'] = 'contact_reduction'
        metrics['baseline_beta'] = self.baseline_beta
        metrics['effective_beta'] = self.effective_beta
        metrics['reduction_percent'] = self.reduction_percent
        metrics['R0'] = self.model.get_basic_reproduction_number()
        
        return metrics
    
    def get_description(self) -> str:
        return (f"Contact reduction scenario: β reduced by {self.reduction_percent*100:.0f}%\n"
                f"β: {self.baseline_beta:.3f} → {self.effective_beta:.3f}\n"
                f"R0: {self.baseline_beta/self.gamma:.2f} → {self.model.get_basic_reproduction_number():.2f}")


class FasterIsolationScenario:
    
    def __init__(self, N: int = 10000, beta: float = 0.30, 
                 baseline_gamma: float = 0.10, gamma_multiplier: float = 2.0,
                 I0: int = 10, R0: int = 0):
        self.N = N
        self.beta = beta
        self.baseline_gamma = baseline_gamma
        self.gamma_multiplier = gamma_multiplier
        self.I0 = I0
        self.R0 = R0
        
        self.effective_gamma = baseline_gamma * gamma_multiplier
        
        self.model = SIRModel(N=N, beta=beta, gamma=self.effective_gamma, I0=I0, R0=R0)
        
    def run_simulation(self, days: int = 120) -> pd.DataFrame:
        """Run faster isolation scenario simulation"""
        results = self.model.simulate(days)
        return results
    
    def get_metrics(self, results: pd.DataFrame = None) -> Dict[str, Any]:
        """Calculate key metrics for faster isolation scenario"""
        if results is None:
            results = self.run_simulation()
        
        metrics = calculate_all_metrics(results, self.N)
        metrics['scenario_type'] = 'faster_isolation'
        metrics['baseline_gamma'] = self.baseline_gamma
        metrics['effective_gamma'] = self.effective_gamma
        metrics['gamma_multiplier'] = self.gamma_multiplier
        metrics['R0'] = self.model.get_basic_reproduction_number()
        
        return metrics
    
    def get_description(self) -> str:
        baseline_infectious_period = 1 / self.baseline_gamma
        effective_infectious_period = 1 / self.effective_gamma
        return (f"Faster isolation scenario: γ increased by {(self.gamma_multiplier-1)*100:.0f}%\n"
                f"Average infectious period: {baseline_infectious_period:.1f} days → {effective_infectious_period:.1f} days\n"
                f"R0: {self.beta/self.baseline_gamma:.2f} → {self.model.get_basic_reproduction_number():.2f}")


class VaccinationScenario:
    """
    Vaccination scenario - partial population initially immune
    """
    
    def __init__(self, N: int = 10000, beta: float = 0.30, gamma: float = 0.10,
                 vaccination_rate: float = 0.40, I0: int = 10):
        """
        Initialize vaccination scenario
        
        Args:
            N: Total population size
            beta: Contact rate
            gamma: Recovery rate
            vaccination_rate: Vaccination rate (0.40 = 40% population vaccinated)
            I0: Initial infected individuals
        """
        self.N = N
        self.beta = beta
        self.gamma = gamma
        self.vaccination_rate = vaccination_rate
        self.I0 = I0
        
        self.R0 = int(N * vaccination_rate)
        
        self.model = SIRModel(N=N, beta=beta, gamma=gamma, I0=I0, R0=self.R0)
        
    def run_simulation(self, days: int = 120) -> pd.DataFrame:
        """Run vaccination scenario simulation"""
        results = self.model.simulate(days)
        return results
    
    def get_metrics(self, results: pd.DataFrame = None) -> Dict[str, Any]:
        """Calculate key metrics for vaccination scenario"""
        if results is None:
            results = self.run_simulation()
        
        metrics = calculate_all_metrics(results, self.N)
        metrics['scenario_type'] = 'vaccination'
        metrics['vaccination_rate'] = self.vaccination_rate
        metrics['initial_immune'] = self.R0
        metrics['R0'] = self.model.get_basic_reproduction_number()
        
        return metrics
    
    def get_description(self) -> str:
        return (f"Vaccination scenario: {self.vaccination_rate*100:.0f}% population initially immune\n"
                f"Initial immune population: {self.R0:,}\n"
                f"Effective susceptible population: {self.N - self.R0 - self.I0:,}")


class CombinedInterventionScenario:
    
    def __init__(self, N: int = 10000, baseline_beta: float = 0.30, 
                 reduction_percent: float = 0.30, baseline_gamma: float = 0.10,
                 gamma_multiplier: float = 2.0, I0: int = 10, R0: int = 0):
        """
        Initialize combined intervention scenario
        
        Args:
            N: Total population size
            baseline_beta: Baseline contact rate
            reduction_percent: Contact reduction percentage
            baseline_gamma: Baseline recovery rate
            gamma_multiplier: Recovery rate multiplier
            I0: Initial infected individuals
            R0: Initial recovered individuals
        """
        self.N = N
        self.baseline_beta = baseline_beta
        self.reduction_percent = reduction_percent
        self.baseline_gamma = baseline_gamma
        self.gamma_multiplier = gamma_multiplier
        self.I0 = I0
        self.R0 = R0
        
        self.effective_beta = baseline_beta * (1 - reduction_percent)
        self.effective_gamma = baseline_gamma * gamma_multiplier
        
        self.model = SIRModel(N=N, beta=self.effective_beta, gamma=self.effective_gamma, 
                             I0=I0, R0=R0)
        
    def run_simulation(self, days: int = 120) -> pd.DataFrame:
        """Run combined intervention scenario simulation"""
        results = self.model.simulate(days)
        return results
    
    def get_metrics(self, results: pd.DataFrame = None) -> Dict[str, Any]:
        """Calculate key metrics for combined intervention scenario"""
        if results is None:
            results = self.run_simulation()
        
        metrics = calculate_all_metrics(results, self.N)
        metrics['scenario_type'] = 'combined_intervention'
        metrics['baseline_beta'] = self.baseline_beta
        metrics['effective_beta'] = self.effective_beta
        metrics['reduction_percent'] = self.reduction_percent
        metrics['baseline_gamma'] = self.baseline_gamma
        metrics['effective_gamma'] = self.effective_gamma
        metrics['gamma_multiplier'] = self.gamma_multiplier
        metrics['R0'] = self.model.get_basic_reproduction_number()
        
        return metrics
    
    def get_description(self) -> str:
        baseline_R0 = self.baseline_beta / self.baseline_gamma
        effective_R0 = self.model.get_basic_reproduction_number()
        return (f"Combined intervention scenario: Contact reduction {self.reduction_percent*100:.0f}% + Faster isolation\n"
                f"β: {self.baseline_beta:.3f} → {self.effective_beta:.3f}\n"
                f"γ: {self.baseline_gamma:.3f} → {self.effective_gamma:.3f}\n"
                f"R0: {baseline_R0:.2f} → {effective_R0:.2f}")


def run_all_intervention_scenarios() -> Dict[str, Any]:
    scenarios = {
        'baseline': BaselineScenario(),
        'contact_reduction': ContactReductionScenario(),
        'faster_isolation': FasterIsolationScenario(),
        'vaccination': VaccinationScenario(),
        'combined': CombinedInterventionScenario()
    }
    
    results = {}
    metrics = {}
    
    for name, scenario in scenarios.items():
        scenario_results = scenario.run_simulation()
        scenario_metrics = scenario.get_metrics(scenario_results)
        
        results[name] = scenario_results
        metrics[name] = scenario_metrics
    
    return {
        'results': results,
        'metrics': metrics,
        'scenarios': scenarios
    }


from .baseline import BaselineScenario
