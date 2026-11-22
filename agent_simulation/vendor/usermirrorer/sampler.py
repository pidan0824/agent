"""Unmodified function bodies from UserMirrorer sampler/sampler.py (MIT).
See PROVENANCE.json. KuaiSearch adapters live outside this module.
"""
from __future__ import annotations
import random
from datetime import datetime
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch

def get_spectral_info(inter_mat: torch.Tensor, num_eigenvectors: int) -> torch.Tensor:
    """Get spectral information from interaction matrix, including node degrees and singular vectors.
    
    Args:
        inter_mat: Sparse interaction matrix
        num_eigenvectors: Number of eigenvectors to use in SVD
        
    Returns:
        V_mat: Matrix of eigenvectors scaled by degree
    """
    def _degree(mat, dim=0, exp=-0.5):
        """Get degree of nodes"""
        d_inv = torch.nan_to_num(
            torch.clip(torch.sparse.sum(mat, dim=dim).to_dense(), min=1.).pow(exp), 
            nan=1., posinf=1., neginf=1.
        )
        return d_inv
    
    # Calculate degrees and normalized adjacency
    di_isqr = _degree(inter_mat, dim=0).reshape(-1, 1)
    di_sqr = _degree(inter_mat, dim=0, exp=0.5).reshape(-1, 1)

    vals = inter_mat.values() * di_isqr[inter_mat.indices()[1]].squeeze()
    inter_mat = torch.sparse_coo_tensor(
        inter_mat.indices(),
        _degree(inter_mat, dim=1)[inter_mat.indices()[0]] * vals,
        size=inter_mat.shape, dtype=torch.float
    ).coalesce()
    # Get eigenvectors
    _, _, V_mat = torch.svd_lowrank(inter_mat, q=max(4 * num_eigenvectors, 32), niter=10)
    V_mat = V_mat[:, :num_eigenvectors]
    
    return V_mat * di_sqr

def content_predict(
        all_item_embeedding: torch.Tensor,      # (num_items, embedding_dim)
        target_item_id: torch.Tensor           # (batch_size, )
    ) -> torch.Tensor:
    """Generate content-based similarity scores between target items and all items.
    
    Args:
        all_item_embeedding: Embeddings for all items
        target_item_id: IDs of target items to compare against
        
    Returns:
        Similarity scores between target items and all items
    """
    item_id = torch.tensor(target_item_id.tolist()).to(torch.long)
    
    all_scores = torch.matmul(all_item_embeedding[item_id], all_item_embeedding.T)       # (batch_size, num_items)
    return all_scores

def collaborative_predict(
        all_item_embeedding: torch.Tensor,      # (num_items, embedding_dim)
        interaction_matrix: torch.Tensor       # (batch_size, num_items)
    ) -> torch.Tensor:
    """Generate collaborative filtering predictions based on user interactions.
    
    Args:
        all_item_embeedding: Embeddings for all items
        interaction_matrix: Matrix of user-item interactions
        
    Returns:
        Predicted scores for all items based on collaborative filtering
    """
    all_scores = torch.matmul(torch.matmul(interaction_matrix, all_item_embeedding), all_item_embeedding.T)        # (batch_size, num_items)
    return all_scores

def random_scoring(sf: SampleFormulator, sampled_inter_ids: torch.Tensor) -> torch.Tensor:
    """Generate random scores for items.
    
    Args:
        sf: SampleFormulator instance
        sampled_inter_ids: IDs of sampled interactions
        
    Returns:
        Random scores for each item
    """
    return torch.rand(sampled_inter_ids.shape[0], sf.item_df.shape[0])

def content_based_scoring(sf: SampleFormulator, sampled_inter_ids: torch.Tensor) -> torch.Tensor:
    """Score items based on content similarity using item embeddings.
    
    Args:
        sf: SampleFormulator instance
        sampled_inter_ids: IDs of sampled interactions
        
    Returns:
        Content-based similarity scores for each item
    """
    return content_predict(sf.item_content_embedding, sf.interaction_df['iid'].loc[sampled_inter_ids])

def collaborative_scoring(sf: SampleFormulator, sampled_inter_ids: torch.Tensor) -> torch.Tensor:
    """Score items based on collaborative filtering using interaction patterns.
    
    Args:
        sf: SampleFormulator instance
        sampled_inter_ids: IDs of sampled interactions
        
    Returns:
        Collaborative filtering scores for each item
    """
    user_ids = sf.interaction_df['uid'].loc[sampled_inter_ids].to_numpy()
    inter_mat = sf.user_item_interaction_matrix.index_select(0, torch.tensor(user_ids).to(torch.long)).to_dense()
    return collaborative_predict(sf.item_collaborative_embedding, inter_mat)

def topk_sampling(scores: torch.Tensor, sample_num: int) -> torch.Tensor:
    """Sample top-k items with highest scores.
    
    Args:
        scores: Scores for each item
        sample_num: Number of top items to select
        
    Returns:
        Indices of top-k scored items
    """
    return torch.topk(scores, k=sample_num).indices

def multinomial_sampling(scores: torch.Tensor, sample_num: int) -> torch.Tensor:
    """Sample items using multinomial probability distribution.
    
    Args:
        scores: Probability scores for each item
        sample_num: Number of items to sample
        
    Returns:
        Indices of sampled items
    """
    assert sample_num <= scores.shape[1], "Sample number is larger than the number of items"
    return torch.multinomial(scores, num_samples=sample_num)

def interaction_filtering(sf: SampleFormulator, sampled_inter: pd.DataFrame) -> pd.DataFrame:
    """Filter out items that user has already interacted with.
    
    Args:
        sf: SampleFormulator instance
        sampled_inter: DataFrame of sampled interactions
        
    Returns:
        Filtered DataFrame with previously interacted items removed
    """
    # sampled_inter['interaction_list'] = sf.user_interaction_list.loc[sampled_inter['user_id']]
    sampled_inter['impression_list'] = sampled_inter.apply(
        lambda x: [i for i in x['impression_list'] if i not in sf.user_interaction_list.loc[x['user_id']]],
        axis=1
    )
    return sampled_inter

def time_filtering(sf: SampleFormulator, sampled_inter: pd.DataFrame) -> pd.DataFrame:
    """Filter out items based on timestamp constraints.
    
    Args:
        sf: SampleFormulator instance
        sampled_inter: DataFrame of sampled interactions
        
    Returns:
        Filtered DataFrame with future items removed
    """
    time = sf.item_df.set_index('item_id')['item_features'].apply(lambda x: datetime.fromtimestamp(x['time']))
    sampled_inter['impression_list'] = sampled_inter.apply(
        lambda x: [i for i in x['impression_list'] if time[i] <= x['timestamp']],
        axis=1
    )
    return sampled_inter

def gt_position(ranked_list: List) -> int:
    """Randomly determine position to insert ground truth item.
    
    Args:
        ranked_list: List of ranked items
        
    Returns:
        Random position index for ground truth insertion
    """
    pos = random.randint(0, len(ranked_list) - 1)
    return pos

def add_gt_to_impression(impression_list: List, item_id: str) -> Tuple[List, int]:
    """Insert ground truth item into impression list at random position.
    
    Args:
        impression_list: List of item impressions
        item_id: Ground truth item ID to insert
        
    Returns:
        Updated impression list with ground truth item inserted
    """
    pos = gt_position(impression_list)
    impression_list.insert(pos, item_id)
    return impression_list, pos

def merge_ranking_list(ranking_lists: List[List]) -> List:
    """Merge multiple ranking lists while preserving relative ordering within each list.
    
    Args:
        ranking_lists: List of ranked item lists to merge
        
    Returns:
        Single merged list with duplicates removed
    """
    ranking_lists = [lst for lst in ranking_lists if len(lst) > 0]
    if not ranking_lists:
        return []
    length_list = np.array([len(ranking_list) for ranking_list in ranking_lists]).cumsum()
    idx_list = np.arange(length_list[-1])
    random.shuffle(idx_list)
    # Sort the idx_list in each section, split by the length of the ranking_lists
    split_idx_list = np.array_split(idx_list, length_list)
    merged_list = np.empty(length_list[-1], dtype=object)
    for ranking_list, split_idx in zip(ranking_lists, split_idx_list):
        merged_list[np.sort(split_idx)] = np.array(ranking_list)
    return list(dict.fromkeys(merged_list))
