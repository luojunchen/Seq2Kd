"""
Protein-DNA Interaction Dataset Classes

This module provides dataset classes for protein-DNA interaction prediction tasks.
Supports different data formats and preprocessing strategies for inference.
"""

import random
import numpy as np
from typing import List, Tuple
from torch.utils.data.dataset import Dataset

__all__ = ['ProteinDNAGeneratedDataset']

# =============================================================================
# Constants and Vocabulary Definitions
# =============================================================================

# DNA nucleotide vocabulary mapping
DNA_WORD2IDX = {
    '[PAD]': 0, 
    '[MASK]': 1, 
    'A': 2, 
    'T': 3, 
    'C': 4, 
    'G': 5, 
    '-': 6, 
    '[CLS]': 7, 
    '[SEP]': 8
}

# Amino acid vocabulary mapping
ACIDS = '-XACDEFGHIKLMNPQRSTVWY'


# =============================================================================
# Dataset Classes
# =============================================================================

class ProteinDNAGeneratedDataset(Dataset):
    """
    Dataset for protein-DNA binding inference.
    
    This dataset handles DNA and protein sequence encoding with padding
    for use in the Seq2Kd model during inference.
    
    Args:
        data_list: List of tuples containing (dna_seq, protein_seq)
        dna_len: Maximum DNA sequence length (default: 40)
        protein_len: Maximum protein sequence length (default: 512)
    """
    
    def __init__(
        self, 
        data_list: List[Tuple[str, str]],
        dna_len: int = 40, 
        protein_len: int = 512,
    ):
        self.data_list = data_list
        self.dna_len = dna_len
        self.protein_len = protein_len

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (dna_encoded, protein_encoded)
        """
        dna_seq, protein_seq = self.data_list[idx]
        
        dna_encoded = self._process_dna_sequence(dna_seq)
        protein_encoded = self._process_protein_sequence(protein_seq)
        
        return np.array(dna_encoded), np.array(protein_encoded)

    def _process_dna_sequence(self, dna_seq: str) -> List[int]:
        """
        Process DNA sequence: clean, add CLS token, encode, and pad.
        
        Args:
            dna_seq: Raw DNA sequence string
            
        Returns:
            Encoded and padded DNA sequence
        """
        # Replace any non-ATCG characters with random ATCG
        cleaned_seq = [s if s in ['A', 'T', 'C', 'G'] else random.choice(['A', 'T', 'C', 'G']) for s in dna_seq]
        
        # Add CLS token at the beginning
        cleaned_seq = ['[CLS]'] + cleaned_seq
        
        # Convert to indices and truncate/pad
        dna_encoded = [DNA_WORD2IDX.get(x, 6) for x in cleaned_seq][:self.dna_len]
        
        if len(dna_encoded) < self.dna_len:
            dna_encoded += [DNA_WORD2IDX['[PAD]']] * (self.dna_len - len(dna_encoded))
            
        return dna_encoded

    def _process_protein_sequence(self, protein_seq: str) -> List[int]:
        """
        Process protein sequence: encode amino acids and pad to fixed length.
        
        Args:
            protein_seq: Raw protein sequence string
            
        Returns:
            Encoded and padded protein sequence
        """
        # Encode amino acids (unknown characters mapped to 'X')
        protein_encoded = [ACIDS.index(x if x in ACIDS else 'X') for x in protein_seq]
        
        if len(protein_encoded) > self.protein_len:
            # Random truncation if too long
            start = random.randint(0, len(protein_encoded) - self.protein_len)
            protein_encoded = protein_encoded[start:start + self.protein_len]
        else:
            # Pad with zeros if too short
            protein_encoded = protein_encoded + [0] * (self.protein_len - len(protein_encoded))
            
        return protein_encoded

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_list)
