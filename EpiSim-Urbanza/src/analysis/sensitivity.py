
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable, Optional, Tuple
from ..models.sir_model import SIRModel
from ..models.seir_model import SEIRModel
from .metrics import calculate_all_metrics


class SensitivityAnalyzer:
    
    def __init__(self, base_params: Dict[str, Any]):
        self.base_params = base_params.copy()
        
    def analyze_parameter(self, parameter: str, param_range: List[float],
                         model_type: str = 'sir', days: int = 120,
                         metrics_to_track: List[str] = None) -> Dict[str, Any]:
        if metrics_to_track is None:
            metrics_to_track = ['peak_infection_proportion', 'time_to_peak', 'final_epidemic_size']
        
        results = {
            'parameter': parameter,
            'parameter_values': param_range,
            'metrics': {metric: [] for metric in metrics_to_track},
            'full_results': {},
            'R0_values': []
        }
        
        for param_value in param_range:
            params = self.base_params.copy()
            params[parameter] = param_value
            
            if model_type.lower() == 'sir':
                model = SIRModel(**params)
            elif model_type.lower() == 'seir':
                model = SEIRModel(**params)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            simulation_results = model.simulate(days)
            
            metrics = calculate_all_metrics(simulation_results, params['N'])
            for metric in metrics_to_track:
                if metric in metrics:
                    results['metrics'][metric].append(metrics[metric])
                else:
                    results['metrics'][metric].append(None)
            
            results['full_results'][param_value] = metrics
            results['R0_values'].append(model.get_basic_reproduction_number())
        
        return results
    
    def analyze_beta_sensitivity(self, beta_range: Optional[List[float]] = None,
                                model_type: str = 'sir') -> Dict[str, Any]:
        """
        Analyze sensitivity of contact rate beta
        
        Args:
            beta_range: Beta value range, uses default range if None
            model_type: Model type
            
        Returns:
            Beta sensitivity analysis results
        """
        if beta_range is None:
            base_beta = self.base_params.get('beta', 0.30)
            beta_range = np.linspace(0.1, 0.6, 11).tolist()
        
        return self.analyze_parameter('beta', beta_range, model_type)
    
    def analyze_gamma_sensitivity(self, gamma_range: Optional[List[float]] = None,
                                 model_type: str = 'sir') -> Dict[str, Any]:
        """
        Analyze sensitivity of recovery rate gamma
        
        Args:
            gamma_range: Gamma value range, uses default range if None
            model_type: Model type
            
        Returns:
            Gamma sensitivity analysis results
        """
        if gamma_range is None:
            base_gamma = self.base_params.get('gamma', 0.10)
            gamma_range = np.linspace(0.05, 0.25, 11).tolist()
        
        return self.analyze_parameter('gamma', gamma_range, model_type)
    
    def analyze_sigma_sensitivity(self, sigma_range: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Analyze sensitivity of incubation rate sigma (SEIR models only)
        
        Args:
            sigma_range: Sigma value range, uses default range if None
            
        Returns:
            Sigma sensitivity analysis results
        """
        if 'sigma' not in self.base_params:
            self.base_params['sigma'] = 0.20
        
        if sigma_range is None:
            base_sigma = self.base_params.get('sigma', 0.20)
            sigma_range = np.linspace(0.1, 0.4, 11).tolist()
        
        return self.analyze_parameter('sigma', sigma_range, 'seir')
    
    def analyze_R0_sensitivity(self, R0_range: Optional[List[float]] = None,
                              model_type: str = 'sir') -> Dict[str, Any]:
        """
        Analyze sensitivity of basic reproduction number R0 (by adjusting beta)
        
        Args:
            R0_range: R0 value range, uses default range if None
            model_type: Model type
            
        Returns:
            R0 sensitivity analysis results
        """
        if R0_range is None:
            R0_range = np.linspace(1.0, 5.0, 17).tolist()
        
        gamma = self.base_params.get('gamma', 0.10)
        beta_range = [R0 * gamma for R0 in R0_range]
        
        results = self.analyze_parameter('beta', beta_range, model_type)
        results['R0_range'] = R0_range
        results['parameter'] = 'R0'
        
        return results
    
    def comprehensive_sensitivity_analysis(self, model_type: str = 'sir') -> Dict[str, Any]:
        """
        Conduct comprehensive sensitivity analysis
        
        Args:
            model_type: Model type
            
        Returns:
            Comprehensive sensitivity analysis results
        """
        results = {
            'model_type': model_type,
            'base_params': self.base_params,
            'analyses': {}
        }
        
        results['analyses']['beta'] = self.analyze_beta_sensitivity(model_type=model_type)
        
        results['analyses']['gamma'] = self.analyze_gamma_sensitivity(model_type=model_type)
        
        results['analyses']['R0'] = self.analyze_R0_sensitivity(model_type=model_type)
        
        if model_type.lower() == 'seir':
            results['analyses']['sigma'] = self.analyze_sigma_sensitivity()
        
        return results


def create_sensitivity_summary(sensitivity_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create summary table for sensitivity analysis results
    
    Args:
        sensitivity_results: Sensitivity analysis results
        
    Returns:
        Summary DataFrame
    """
    summary_data = []
    
    for param_name, analysis in sensitivity_results['analyses'].items():
        param_values = analysis['parameter_values']
        
        for metric_name, metric_values in analysis['metrics'].items():
            if metric_values and all(v is not None for v in metric_values):
                min_val = min(metric_values)
                max_val = max(metric_values)
                range_val = max_val - min_val
                mean_val = np.mean(metric_values)
                std_val = np.std(metric_values)
                cv = std_val / mean_val if mean_val != 0 else 0
                
                summary_data.append({
                    'parameter': param_name,
                    'metric': metric_name,
                    'min_value': min_val,
                    'max_value': max_val,
                    'range': range_val,
                    'mean': mean_val,
                    'std': std_val,
                    'coefficient_of_variation': cv,
                    'sensitivity_ratio': range_val / mean_val if mean_val != 0 else 0
                })
    
    return pd.DataFrame(summary_data)


def compare_model_sensitivity(base_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare parameter sensitivity between SIR and SEIR models
    
    Args:
        base_params: Base parameters
        
    Returns:
        Model sensitivity comparison results
    """
    seir_params = base_params.copy()
    if 'sigma' not in seir_params:
        seir_params['sigma'] = 0.20
    if 'E0' not in seir_params:
        seir_params['E0'] = 0
    
    sir_analyzer = SensitivityAnalyzer(base_params)
    seir_analyzer = SensitivityAnalyzer(seir_params)
    
    sir_results = sir_analyzer.comprehensive_sensitivity_analysis('sir')
    seir_results = seir_analyzer.comprehensive_sensitivity_analysis('seir')
    
    sir_summary = create_sensitivity_summary(sir_results)
    seir_summary = create_sensitivity_summary(seir_results)
    
    return {
        'sir_analysis': sir_results,
        'seir_analysis': seir_results,
        'sir_summary': sir_summary,
        'seir_summary': seir_summary,
        'comparison': {
            'sir_peak_sensitivity': sir_summary[sir_summary['metric'] == 'peak_infection_proportion'],
            'seir_peak_sensitivity': seir_summary[seir_summary['metric'] == 'peak_infection_proportion']
        }
    }
