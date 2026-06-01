"""
Common Utilities for Seq2Kd

This module provides common utility functions for data loading and processing.
"""

import torch
from torch.nn.utils.rnn import pad_sequence


def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    Handles batching of DNA and protein sequences with padding.
    
    Args:
        batch: List of (dna_seq, protein_seq) tuples
        
    Returns:
        Tuple of (dna_batch, protein_batch) tensors
    """
    dna_x, protein_x = [], []
    
    for dna_seq, protein_seq in batch:
        dna_x.append(torch.tensor(dna_seq, dtype=torch.long))
        protein_x.append(torch.tensor(protein_seq, dtype=torch.long))

    dna_x_padded = pad_sequence(dna_x, batch_first=True, padding_value=0)
    protein_x_padded = pad_sequence(protein_x, batch_first=True, padding_value=0)
    
    return (dna_x_padded, protein_x_padded)


def count_parameters(model):
    """Count total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters())
