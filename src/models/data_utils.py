"""
Data Utilities for Seq2Kd

This module provides utility functions and constants for data processing
in the Seq2Kd protein-DNA binding prediction framework.
"""

__all__ = ['ACIDS', 'GENE_DIC']

# Amino acid vocabulary (including padding and unknown tokens)
ACIDS = '-XACDEFGHIKLMNPQRSTVWY'

# DNA nucleotide vocabulary
GENE_DIC = '0-123456789asdfgshaishdjATCG'
