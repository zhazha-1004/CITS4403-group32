
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def calculate_peak_infection(results: pd.DataFrame) -> Dict[str, Any]:
    if 'I_prop' not in results.columns:
        raise ValueError("Results DataFrame must contain 'I_prop' column")
    
    peak_idx = results['I_prop'].idxmax()
    peak_proportion = results.loc[peak_idx, 'I_prop']
    peak_day = results.loc[peak_idx, 'day']
    peak_count = results.loc[peak_idx, 'I'] if 'I' in results.columns else None
    
    return {
        'peak_infection_proportion': peak_proportion,
        'peak_infection_count': peak_count,
        'time_to_peak': peak_day,
        'peak_day': peak_day
    }


def calculate_final_epidemic_size(results: pd.DataFrame, N: int) -> Dict[str, Any]:
    if 'S' not in results.columns and 'S_prop' not in results.columns:
        raise ValueError("Results DataFrame must contain 'S' or 'S_prop' column")
    
    final_state = results.iloc[-1]
    
    if 'S' in results.columns:
        final_susceptible = final_state['S']
        final_epidemic_size = (N - final_susceptible) / N
    else:
        final_susceptible_prop = final_state['S_prop']
        final_epidemic_size = 1 - final_susceptible_prop
        final_susceptible = final_susceptible_prop * N
    
    attack_rate = final_epidemic_size
    
    return {
        'final_epidemic_size': final_epidemic_size,
        'attack_rate': attack_rate,
        'final_susceptible_count': final_susceptible,
        'final_susceptible_proportion': final_susceptible / N if 'S' in results.columns else final_susceptible_prop
    }


def calculate_epidemic_duration(results: pd.DataFrame, threshold: float = 1e-3) -> Dict[str, Any]:
    """
    Calculate epidemic duration
    
    Args:
        results: Simulation results DataFrame
        threshold: Infection proportion threshold, below which epidemic is considered ended
        
    Returns:
        Dictionary containing epidemic duration metrics
    """
    if 'I_prop' not in results.columns:
        raise ValueError("Results DataFrame must contain 'I_prop' column")
    
    start_idx = results[results['I_prop'] > threshold].index
    start_day = results.loc[start_idx[0], 'day'] if len(start_idx) > 0 else 0
    
    end_idx = results[results['I_prop'] > threshold].index
    end_day = results.loc[end_idx[-1], 'day'] if len(end_idx) > 0 else results['day'].iloc[-1]
    
    duration = end_day - start_day
    
    return {
        'epidemic_start_day': start_day,
        'epidemic_end_day': end_day,
        'epidemic_duration': duration
    }


def calculate_cumulative_incidence(results: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate cumulative incidence related metrics
    
    Args:
        results: Simulation results DataFrame
        
    Returns:
        Dictionary containing cumulative incidence metrics
    """
    if 'cumulative_infections' not in results.columns:
        raise ValueError("Results DataFrame must contain 'cumulative_infections' column")
    
    final_cumulative = results['cumulative_infections'].iloc[-1]
    
    if 'new_infections' in results.columns:
        daily_new = results['new_infections']
        max_daily_new = daily_new.max()
        max_daily_new_day = results.loc[daily_new.idxmax(), 'day']
        avg_daily_new = daily_new.mean()
    else:
        max_daily_new = None
        max_daily_new_day = None
        avg_daily_new = None
    
    return {
        'final_cumulative_infections': final_cumulative,
        'max_daily_new_infections': max_daily_new,
        'max_daily_new_infections_day': max_daily_new_day,
        'average_daily_new_infections': avg_daily_new
    }


def calculate_all_metrics(results: pd.DataFrame, N: int, 
                         infection_threshold: float = 1e-3) -> Dict[str, Any]:
    """
    Calculate all key metrics
    
    Args:
        results: Simulation results DataFrame
        N: Total population size
        infection_threshold: Infection threshold
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {}
    
    try:
        peak_metrics = calculate_peak_infection(results)
        metrics.update(peak_metrics)
        
        final_metrics = calculate_final_epidemic_size(results, N)
        metrics.update(final_metrics)
        
        duration_metrics = calculate_epidemic_duration(results, infection_threshold)
        metrics.update(duration_metrics)
        
        incidence_metrics = calculate_cumulative_incidence(results)
        metrics.update(incidence_metrics)
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        if 'I_prop' in results.columns:
            metrics['peak_infection_proportion'] = results['I_prop'].max()
            metrics['time_to_peak'] = results.loc[results['I_prop'].idxmax(), 'day']
        
        if 'S_prop' in results.columns:
            metrics['final_epidemic_size'] = 1 - results['S_prop'].iloc[-1]
    
    return metrics


def compare_scenarios(scenario_results: Dict[str, pd.DataFrame], N: int) -> pd.DataFrame:
    """
    Compare key metrics across multiple scenarios
    
    Args:
        scenario_results: Mapping from scenario names to results DataFrames
        N: Total population size
        
    Returns:
        DataFrame containing comparison results
    """
    comparison_data = []
    
    for scenario_name, results in scenario_results.items():
        metrics = calculate_all_metrics(results, N)
        metrics['scenario'] = scenario_name
        comparison_data.append(metrics)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if 'scenario' in comparison_df.columns:
        cols = ['scenario'] + [col for col in comparison_df.columns if col != 'scenario']
        comparison_df = comparison_df[cols]
    
    return comparison_df
