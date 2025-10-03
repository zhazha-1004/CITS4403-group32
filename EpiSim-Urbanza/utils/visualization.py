
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_sir_timeseries(results: pd.DataFrame, title: str = "SIR Model Time Series",
                       figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    if all(col in results.columns for col in ['S', 'I', 'R']):
        ax1.plot(results['day'], results['S'], label='Susceptible (S)', linewidth=2)
        ax1.plot(results['day'], results['I'], label='Infected (I)', linewidth=2)
        ax1.plot(results['day'], results['R'], label='Recovered (R)', linewidth=2)
        ax1.set_ylabel('Population')
        ax1.set_title(f'{title} - Absolute Numbers')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    if all(col in results.columns for col in ['S_prop', 'I_prop', 'R_prop']):
        ax2.plot(results['day'], results['S_prop'], label='Susceptible Proportion', linewidth=2)
        ax2.plot(results['day'], results['I_prop'], label='Infected Proportion', linewidth=2)
        ax2.plot(results['day'], results['R_prop'], label='Recovered Proportion', linewidth=2)
        ax2.set_ylabel('Proportion')
        ax2.set_xlabel('Days')
        ax2.set_title(f'{title} - Proportions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_seir_timeseries(results: pd.DataFrame, title: str = "SEIR Model Time Series",
                        figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    if all(col in results.columns for col in ['S', 'E', 'I', 'R']):
        ax1.plot(results['day'], results['S'], label='Susceptible (S)', linewidth=2)
        ax1.plot(results['day'], results['E'], label='Exposed (E)', linewidth=2)
        ax1.plot(results['day'], results['I'], label='Infected (I)', linewidth=2)
        ax1.plot(results['day'], results['R'], label='Recovered (R)', linewidth=2)
        ax1.set_ylabel('Population')
        ax1.set_title(f'{title} - Absolute Numbers')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    if all(col in results.columns for col in ['S_prop', 'E_prop', 'I_prop', 'R_prop']):
        ax2.plot(results['day'], results['S_prop'], label='Susceptible Proportion', linewidth=2)
        ax2.plot(results['day'], results['E_prop'], label='Exposed Proportion', linewidth=2)
        ax2.plot(results['day'], results['I_prop'], label='Infected Proportion', linewidth=2)
        ax2.plot(results['day'], results['R_prop'], label='Recovered Proportion', linewidth=2)
        ax2.set_ylabel('Proportion')
        ax2.set_xlabel('Days')
        ax2.set_title(f'{title} - Proportions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_infection_comparison(scenario_results: Dict[str, pd.DataFrame], 
                            title: str = "Infection Proportion Comparison Across Scenarios",
                            figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    
    for scenario_name, results in scenario_results.items():
        if 'I_prop' in results.columns:
            ax.plot(results['day'], results['I_prop'], 
                   label=scenario_name, linewidth=2.5)
    
    ax.set_xlabel('Days')
    ax.set_ylabel('Infected Proportion')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_metrics_comparison(comparison_df: pd.DataFrame, 
                          metrics: List[str] = None,
                          figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    if metrics is None:
        metrics = ['peak_infection_proportion', 'time_to_peak', 'final_epidemic_size']
    
    available_metrics = [m for m in metrics if m in comparison_df.columns]
    
    if not available_metrics:
        raise ValueError("No available metric columns found")
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        
        bars = ax.bar(comparison_df['scenario'], comparison_df[metric])
        
        metric_names = {
            'peak_infection_proportion': 'Peak Infection Proportion',
            'time_to_peak': 'Time to Peak (days)',
            'final_epidemic_size': 'Final Epidemic Size',
            'epidemic_duration': 'Epidemic Duration (days)'
        }
        
        ax.set_title(metric_names.get(metric, metric))
        ax.set_ylabel('Value')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_sensitivity_analysis(sensitivity_results: Dict[str, Dict[str, Any]], 
                            parameter: str,
                            metric: str = 'peak_infection_proportion',
                            title: str = None,
                            figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    if title is None:
        title = f'{parameter} Parameter Sensitivity Analysis - {metric}'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    param_values = []
    metric_values = []
    
    for param_val, results in sensitivity_results.items():
        if isinstance(results, dict) and metric in results:
            param_values.append(float(param_val))
            metric_values.append(results[metric])
    
    if param_values and metric_values:
        sorted_data = sorted(zip(param_values, metric_values))
        param_values, metric_values = zip(*sorted_data)
        
        ax.plot(param_values, metric_values, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel(f'{parameter} Value')
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_summary_dashboard(scenario_results: Dict[str, pd.DataFrame],
                           comparison_df: pd.DataFrame,
                           figsize: Tuple[int, int] = (20, 15)) -> plt.Figure:
    fig = plt.figure(figsize=figsize)
    
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])
    for scenario_name, results in scenario_results.items():
        if 'I_prop' in results.columns:
            ax1.plot(results['day'], results['I_prop'], 
                    label=scenario_name, linewidth=2.5)
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Infected Proportion')
    ax1.set_title('Infected Proportion Comparison Across Scenarios')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[1, 0])
    if 'peak_infection_proportion' in comparison_df.columns:
        bars = ax2.bar(comparison_df['scenario'], comparison_df['peak_infection_proportion'])
        ax2.set_title('Peak Infection Proportion')
        ax2.set_ylabel('Proportion')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
    
    ax3 = fig.add_subplot(gs[1, 1])
    if 'time_to_peak' in comparison_df.columns:
        bars = ax3.bar(comparison_df['scenario'], comparison_df['time_to_peak'])
        ax3.set_title('Time to Peak')
        ax3.set_ylabel('Days')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom')
    
    ax4 = fig.add_subplot(gs[2, :])
    if 'final_epidemic_size' in comparison_df.columns:
        bars = ax4.bar(comparison_df['scenario'], comparison_df['final_epidemic_size'])
        ax4.set_title('Final Epidemic Size')
        ax4.set_ylabel('Proportion')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
    
    plt.suptitle('Epidemiological Modeling Results Dashboard', fontsize=16, y=0.98)
    return fig
