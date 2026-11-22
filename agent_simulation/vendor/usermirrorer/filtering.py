"""Unmodified function bodies extracted from UserMirrorer (MIT); see PROVENANCE.json."""
import numpy as np
import pandas as pd
from collections import OrderedDict

def behavior_sampling(all_data: pd.DataFrame, metrics: pd.DataFrame, filter_column: str, sampling_column: str):
    """
    Sample behaviors from the dataset based on specified criteria.
    
    This function selects one behavior per context based on the sampling criteria,
    ensuring that each context has both preferred and non-preferred behaviors.
    
    Args:
        all_data: Combined dataset with all behaviors
        metrics: DataFrame with calculated metrics
        filter_column: Column used for filtering
        sampling_column: Column used for behavior sampling
    
    Returns:
        DataFrame: Sampled behaviors with associated metrics
    """
    # Extract relevant columns from the dataset
    sampled_df = all_data.reset_index(level=2).loc[metrics.set_index(["dataset", "cid"]).index, 
                                                   ["tid", "preference", "entropy", "logloss", "behavior", "variant"]]
    sampled_df = sampled_df.set_index('preference', append=True)
    
    # Sample behaviors based on specified criteria
    if sampling_column == "random":
        # Random sampling
        sampled_df = sampled_df.sample(frac=1).groupby(level=[0,1,2]).first()
    else:
        # Deterministic sampling based on column values
        sampled_df = sampled_df.sort_values(by=sampling_column, ascending=True).groupby(level=[0,1,2]).first()
    
    sampled_df = sampled_df.reset_index(level=2)
    
    # Keep only groups that have at least one True and one False preference (chosen/rejected pair)
    has_true = sampled_df['preference'].groupby(level=[0,1]).any()
    has_false = (~sampled_df['preference']).groupby(level=[0,1]).any()
    valid_indices = has_true & has_false
    sampled_df = sampled_df.loc[valid_indices]
    
    # Join with relevant metrics
    keep_columns = ["choice_cnt"]
    if filter_column in metrics.columns:
        keep_columns.append(filter_column)
    sampled_df = sampled_df.join(metrics.set_index(["dataset", "cid"]).loc[:, keep_columns])
    return sampled_df

def data_filtering(
    metrics: pd.DataFrame,
    filter_column: str,
    group_columns: list,
    filtered_num: int,
    ascending: bool = False
) -> pd.DataFrame:
    """
    Filter data based on specified metrics and stratification criteria.
    
    This function implements stratified sampling to ensure balanced representation
    across different groups while prioritizing samples based on the filter column.
    
    Args:
        metrics: DataFrame containing metric values for filtering
        filter_column: Column name to use for filtering/sorting
        group_columns: Columns to use for stratification
        filtered_num: Maximum number of samples to select
        ascending: Sort order for the filter_column (default: False - highest values first)
        
    Returns:
        DataFrame: Filtered dataset with both chosen and rejected samples
    """
    filtered_metrics = metrics.copy()
    if filter_column == "random":
        # Random sampling without sorting
        filtered_metrics = filtered_metrics.sample(frac=1)
    else:
        # Sort based on the specified column
        filtered_metrics = filtered_metrics.sort_values(by=filter_column, ascending=ascending)
        
    # Calculate the total number of samples to select
    total_samples = min(filtered_num, len(filtered_metrics))
    
    # Create a default group if none specified
    if len(group_columns) == 0:
        filtered_metrics["group"] = "all"
        group_columns = ["group"]
        
    # Get group sizes for proportional allocation
    group_sizes = filtered_metrics.groupby(group_columns).size()
    total_original = sum(group_sizes)

    # Calculate proportional allocation for each group
    group_allocations = {}
    remaining = total_samples
    
    # Allocate samples proportionally to each group
    for group, size in group_sizes.items():
        # Calculate the proportion of samples to take from this group
        allocation = int(np.ceil((size / total_original) * total_samples))
        # Ensure we don't exceed the remaining samples or group size
        allocation = min(allocation, size, remaining)
        group_allocations[group if isinstance(group, tuple) else (group,)] = allocation
        remaining -= allocation

    # Select samples from each group according to allocations
    all_chosen = []
    all_rejected = []
    for group_name, group_df in filtered_metrics.groupby(group_columns):
        allocation = group_allocations.get(group_name if isinstance(group_name, tuple) else (group_name,), 0)
        # Split into preferred and rejected samples
        pref_df = group_df[group_df['preference']]
        rej_df = group_df[~group_df['preference']]
        if allocation > 0:
            # Take top samples based on sorting from each group
            chosen = pref_df.head(allocation)
            # Find matching rejected samples for the chosen ones
            rejected = rej_df.set_index(["dataset", "cid"]).loc[chosen.set_index(["dataset", "cid"]).index].reset_index()
            all_chosen.append(chosen.copy())
            all_rejected.append(rejected.copy())
    
    # Combine all selected samples
    sft_datasets = pd.concat([pd.concat(all_chosen), pd.concat(all_rejected)])
    return sft_datasets
