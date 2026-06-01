"""
ESM2 Protein Language Model Embedding Module

This module provides wrapper classes for using ESM2 (Evolutionary Scale Modeling)
protein language models to generate protein sequence embeddings.

Reference: https://github.com/facebookresearch/esm
"""

import os
import ssl
import hashlib
import esm
import torch
import torch.nn as nn
from importlib.util import find_spec
from urllib.request import urlretrieve
from typing import Any, List

ssl._create_default_https_context = ssl._create_unverified_context


def flatten(lst: List[List[Any]]) -> List[Any]:
    """Flatten a nested list."""
    return [item for sublist in lst for item in sublist]


def compute_md5(file_path, chunk_size=65536) -> str:
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as fin:
        for chunk in iter(lambda: fin.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def download(url, path, save_file=None, md5=None):
    """Download file from URL if not exists."""
    if save_file is None:
        save_file = os.path.basename(url).split("?")[0]
    save_file = os.path.join(path, save_file)
    if not os.path.exists(save_file):
        print(f"Downloading {url} to {save_file}")
        urlretrieve(url, save_file)
    return save_file


class EvolutionaryScaleModeling(nn.Module):
    """
    ESM (Evolutionary Scale Modeling) protein language model wrapper.
    
    Supports various ESM model variants for protein sequence embedding.
    Models are automatically downloaded if not present locally.
    """
    
    # Model download URLs
    url = {
        "esm1b_t33_650M_UR50S": "https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt",
        "esm1v_t33_650M_UR90S_1": "https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt",
        "esm1b_t33_650M_UR50S-contact-regression": "https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt",
        "esm2_t6_8M_UR50D": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt",
        "esm2_t12_35M_UR50D": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt",
        "esm2_t30_150M_UR50D": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt",
        "esm2_t33_650M_UR50D": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
        "ESM-2-3B": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt",
        "ESM-2-15B": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt",
    }
    
    # Model file MD5 checksums
    md5 = {
        "esm1b_t33_650M_UR50S": "ba8914bc3358cae2254ebc8874ee67f6",
        "esm1v_t33_650M_UR90S_1": "1f04c2d2636b02b544ecb5fbbef8fefd",
        "esm1b_t33_650M_UR50S-contact-regression": "e7fe626dfd516fb6824bd1d30192bdb1",
        "esm2_t6_8M_UR50D": "8039fc9cee7f71cd2633b13b5a38ff50",
        "esm2_t12_35M_UR50D": "a894ddb31522e511e1273abb23b5f974",
        "esm2_t30_150M_UR50D": "229fcf8f9f3d4d442215662ca001b906",
        "esm2_t33_650M_UR50D": "ba6d997e29db07a2ad9dca20e024b102",
        "esm2_t36_3B_UR50D": "d37a0d0dbe7431e48a72072b9180b16b",
        "esm2_t48_15B_UR50D": "af61a9c0b792ae50e244cde443b7f4ac",
    }
    
    # Model output dimensions
    output_dim = {
        "esm1b_t33_650M_UR50S": 1280,
        "esm1v_t33_650M_UR90S_1": 1280,
        "esm2_t6_8M_UR50D": 320,
        "esm2_t12_35M_UR50D": 480,
        "esm2_t30_150M_UR50D": 640,
        "esm2_t33_650M_UR50D": 1280,
        "esm2_t36_3B_UR50D": 2560,
        "esm2_t48_15B_UR50D": 5120,
    }
    
    # Number of transformer layers
    num_layer = {
        "esm1b_t33_650M_UR50S": 33,
        "esm1v_t33_650M_UR90S_1": 33,
        "esm2_t6_8M_UR50D": 6,
        "esm2_t12_35M_UR50D": 12,
        "esm2_t30_150M_UR50D": 30,
        "esm2_t33_650M_UR50D": 33,
        "esm2_t36_3B_UR50D": 36,
        "esm2_t48_15B_UR50D": 48,
    }

    def __init__(self, path, model="esm2_t33_650M_UR50D", freeze_n_layers: int = 0):
        """
        Initialize ESM model.
        
        Args:
            path: Directory path to store/load model weights
            model: Model variant name (default: esm2_t33_650M_UR50D)
            freeze_n_layers: Number of layers to freeze (-1 for all layers)
        """
        super().__init__()
        if not find_spec("esm") and not find_spec("fair-esm"):
            raise ImportError("To use ESM model you must install fair-esm first")
        
        path = os.path.expanduser(path)
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.output_dim = self.output_dim[model]

        # Load model and alphabet
        _model, alphabet = self.load_weight(path, model)
        self.model = _model
        self.repr_layer = self.num_layer[model]
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        
        # Freeze layers
        if freeze_n_layers == -1:
            freeze_n_layers = len(list(self.model.named_parameters())) + 1
        print(f"[INFO] Freezing {freeze_n_layers} layers")
        for name, param in self.model.named_parameters():
            base_name, *other = name.split(".")
            param.requires_grad = base_name == "layers" and (int(other[0]) + 1 > freeze_n_layers)
    
    def load_weight(self, path, model):
        """Load model weights from file or download if not exists."""
        if model not in self.url:
            raise ValueError(f"Unknown model `{model}`")
        
        model_path = os.path.join(path, f'{model}.pt')
        if os.path.exists(path) and os.path.isdir(path) and os.path.isfile(model_path):
            if compute_md5(model_path) != self.md5[model]:
                raise ValueError(f"MD5 mismatch for {model_path}")
        else:
            model_path = download(self.url[model], path, md5=self.md5[model])
        
        checkpoint = torch.load(model_path, map_location='cuda')
        regression_data = None
        if model != "esm1v_t33_650M_UR90S_1" and not model.startswith("esm2"):
            regression_model = f"{model}-regression"
            regression_file = download(self.url[regression_model], path, md5=self.md5[regression_model])
            regression_data = torch.load(regression_file, map_location="cuda")
        return esm.pretrained.load_model_and_alphabet_core(model, checkpoint, regression_data)
    
    def convert_to_esm_input(self, sequences, device):
        """Convert sequence list to ESM model input format."""
        seq_input = [(str(idx), seq) for idx, seq in enumerate(sequences)]
        _, _, batch_tokens = self.batch_converter(seq_input)
        return batch_tokens.to(device)


class ESMAsSchNetEmbedding(EvolutionaryScaleModeling):
    """
    ESM model wrapper for generating protein embeddings.
    
    Provides a simple interface to get fixed-size embeddings 
    from protein amino acid sequences.
    """
    
    def __init__(
        self, 
        path='./checkpoints/esm', 
        model="esm2_t33_650M_UR50D", 
        freeze_n_layers=-1
    ):
        """
        Initialize ESM embedding model.
        
        Args:
            path: Directory for ESM model weights (default: ./checkpoints/esm)
            model: ESM model variant to use
            freeze_n_layers: Number of layers to freeze (-1 for all)
        """
        super().__init__(path, model, freeze_n_layers)
    
    def forward(self, batch_aa):
        """
        Generate embeddings for protein sequences.
        
        Args:
            batch_aa: List of protein amino acid sequences
            
        Returns:
            Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        device = torch.cuda.current_device()
        self.model.to(device)

        residues_lists = [[seq] for seq in batch_aa]
        seq_flatten = flatten(residues_lists)
        batch_tokens = self.convert_to_esm_input(seq_flatten, device=device)

        results = self.model(batch_tokens, repr_layers=[self.repr_layer], return_contacts=False)
        representations = results["representations"][self.repr_layer]

        # Remove BOS and EOS tokens
        if self.alphabet.prepend_bos and self.alphabet.append_eos:
            representations = representations[:, 1:-1, :]
        elif self.alphabet.prepend_bos:
            representations = representations[:, 1:, :]
        elif self.alphabet.append_eos:
            representations = representations[:, :-1, :]
        
        return representations
