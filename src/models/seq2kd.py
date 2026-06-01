"""
Seq2Kd: Protein-DNA Binding Affinity Prediction Model

A deep learning model for predicting protein-DNA binding affinities using 
cross-attention mechanisms between protein and DNA sequence representations.
"""

import math
from math import sqrt as msqrt
import os

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from models.esm_embedding import ESMAsSchNetEmbedding
from models.data_utils import ACIDS

def gelu(x):
    """Gaussian Error Linear Unit activation function."""
    return 0.5 * x * (1.0 + torch.erf(x / msqrt(2.0)))


def get_pad_mask(tokens, pad_idx=0):
    """
    Generate padding mask for attention.
    
    Args:
        tokens: Input tensor of shape [batch, seq_len]
        pad_idx: Index of padding token
        
    Returns:
        Padding mask of shape [batch, seq_len, seq_len]
    """
    batch, seq_len = tokens.size()
    pad_mask = tokens.data.eq(pad_idx).unsqueeze(1)
    pad_mask = pad_mask.expand(batch, seq_len, seq_len)
    return pad_mask


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        pos_encoding = self.pe[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        return x + pos_encoding


class Embeddings(nn.Module):
    """Token embeddings with positional encoding."""
    
    def __init__(self, d_model=768, max_vocab=5, max_len=30, p_dropout=0.1):
        super(Embeddings, self).__init__()
        self.word_emb = nn.Embedding(max_vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        word_enc = self.word_emb(x)
        pos = torch.arange(x.shape[1], dtype=torch.long).to(x.device)
        pos = pos.unsqueeze(0).expand_as(x)
        pos_enc = self.pos_emb(pos)
        x = self.norm(word_enc + pos_enc)
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""
    
    def __init__(self, d_k=64):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2) / msqrt(self.d_k))
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model=768, d_k=64, d_v=64, n_heads=12):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, Q, K, V, attn_mask):
        batch = Q.size(0)
        per_Q = self.W_Q(Q).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        per_K = self.W_K(K).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        per_V = self.W_V(V).view(batch, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context = ScaledDotProductAttention()(per_Q, per_K, per_V, attn_mask)
        context = context.transpose(1, 2).contiguous().view(
            batch, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model=768, p_dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        d_ff = d_model * 4
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p_dropout)
        self.gelu = gelu

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    """Transformer encoder layer with pre-norm."""
    
    def __init__(self, d_model=768):
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.enc_attn = MultiHeadAttention(d_model=d_model)
        self.ffn = FeedForwardNetwork(d_model=d_model)

    def forward(self, x, pad_mask):
        residual = x
        x = self.norm1(x)
        x = self.enc_attn(x, x, x, pad_mask) + residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        return x + residual


class SegmentEmbedding(nn.Module):
    """Segment embedding to distinguish DNA and protein sequences."""
    
    def __init__(self, d_model):
        super().__init__()
        self.embedding = nn.Embedding(2, d_model)  # 0: DNA, 1: Protein
    
    def forward(self, x, segment_type):
        batch_size, seq_len, _ = x.shape
        segment_ids = torch.full(
            (batch_size, seq_len), 
            segment_type, 
            device=x.device, 
            dtype=torch.long
        )
        return x + self.embedding(segment_ids)


class CustomInteractionModule(nn.Module):
    """Cross-attention based interaction module for DNA-protein features."""
    
    def __init__(self, d_model):
        super().__init__()
        self.dna_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        self.protein_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        self.cross_attention = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, dna_emb, protein_emb):
        dna_features = self.dna_transform(dna_emb)
        protein_features = self.protein_transform(protein_emb)
        
        attn_output, attn_weights = self.cross_attention(
            dna_features, protein_features, protein_features
        )
        self.attn_output = attn_output
        self.attn_weights = attn_weights
        
        combined = torch.cat([dna_features, attn_output], dim=-1)
        fused = self.fusion(combined)
        output = fused + dna_emb
        
        return output


class MLMHead(nn.Module):
    """Masked Language Modeling head for DNA sequences."""
    
    def __init__(self, d_model, max_vocab):
        super().__init__()
        self.d_model = d_model
        self.dna_fc = nn.Linear(d_model, d_model)
        self.word_classifier = nn.Linear(d_model, max_vocab, bias=False)
        self.gelu = gelu

    def forward(self, sequence_output, masked_pos):
        adjusted_masked_pos = masked_pos.unsqueeze(-1).expand(-1, -1, self.d_model)
        h_masked = torch.gather(sequence_output, dim=1, index=adjusted_masked_pos)
        combined = self.gelu(self.dna_fc(h_masked))
        logits_lm = self.word_classifier(combined)
        return torch.softmax(logits_lm, dim=-1)


class RegressionHead(nn.Module):
    """Regression head for binding affinity prediction."""
    
    def __init__(self, d_model, shared_conv=None):
        super().__init__()
        self.dna_conv = shared_conv if shared_conv is not None else None
        self.regression_pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1, bias=True)
        )
        self.regression_pooler[-1].bias.data.fill_(5.5)
    
    def apply_conv(self, x, conv_layer):
        ori_x = x
        x = x.transpose(1, 2)
        x = conv_layer(x)
        x = x.transpose(1, 2) + ori_x
        return x

    def forward(self, protein_meaning, dna_output):
        protein_features = protein_meaning
        if self.dna_conv:
            dna_features = self.apply_conv(dna_output, self.dna_conv)
        else:
            dna_features = dna_output
        
        dna_global = dna_features.mean(dim=1)
        regression_input = dna_global
        return self.regression_pooler(regression_input).squeeze(-1)


class ClassificationHead(nn.Module):
    """Classification head for binding prediction."""
    
    def __init__(self, d_model, output_class=2, dropout_rate=0.1, shared_conv=None):
        super().__init__()
        self.dna_conv = shared_conv if shared_conv is not None else None
        self.classification_pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, output_class)
        )

    def apply_conv(self, x, conv_layer):
        ori_x = x
        x = x.transpose(1, 2)
        x = conv_layer(x)
        x = x.transpose(1, 2) + ori_x
        return x

    def forward(self, protein_meaning, dna_output):
        if self.dna_conv:
            dna_features = self.apply_conv(dna_output, self.dna_conv)
        else:
            dna_features = dna_output
        
        dna_global = dna_features.mean(dim=1)
        logits = self.classification_pooler(dna_global)
        return logits


class Seq2Kd(nn.Module):
    """
    Seq2Kd: Sequence to Kd (Binding Affinity) Prediction Model
    
    A deep learning model that predicts protein-DNA binding affinities using:
    - ESM2 for protein sequence embedding
    - Cross-attention between protein and DNA representations
    
    Args:
        n_layers: Number of transformer layers
        d_model: Hidden dimension size
        max_vocab: DNA vocabulary size
        max_len: Maximum DNA sequence length
        p_dropout: Dropout probability
        freeze_layer: Number of ESM layers to freeze (-1 for all)
        num_head: Number of attention heads
        **kwargs: Additional configuration options
    """
    
    def __init__(
        self, 
        n_layers=6, 
        d_model=128, 
        max_vocab=8, 
        max_len=41,
        p_dropout=0.1, 
        freeze_layer=-1, 
        num_head=16, 
        **kwargs
    ):
        super(Seq2Kd, self).__init__()
        self.d_model = d_model
        num_cross = kwargs.get('num_cross', 1)
        
        # Configuration from kwargs
        self.esm = kwargs.get('esm', False)
        self.dna_attn = kwargs.get('dna_attn', False)
        self.dna_self_attn = kwargs.get('dna_self_attn', False)
        self.refined_conv1d = kwargs.get('refined_conv1d', True)
        self.esm_model = kwargs.get('esm_model', 'esm2')
        self.classification = kwargs.get('classification', False)
        self.segment_embedding = kwargs.get('segment_embedding', False)
        self.pretrain = kwargs.get('pretrain', True)
        self.finetune = kwargs.get('finetune', False)
        self.use_protein_len = kwargs.get('use_protein_len', 1022)
        self.finetune_attn = kwargs.get('finetune_attn', False)
        
        # ESM checkpoint path - use relative path from project root
        self.esm_path = kwargs.get('esm_path', './checkpoints/esm')

        if self.finetune_attn:
            self.finetune_attn_head = CustomInteractionModule(d_model)

        # Segment embeddings
        if self.segment_embedding:  
            self.dna_segment_embedding = SegmentEmbedding(d_model)
            self.protein_segment_embedding = SegmentEmbedding(d_model)

        # Setup encoders
        self._setup_dna_encoder(d_model, max_vocab, max_len)
        self._setup_protein_encoder(d_model, freeze_layer, num_head, num_cross)
        self._setup_cross_attention_layers(d_model, num_head, num_cross)
        
        # Task heads
        self.mlm_head = MLMHead(d_model, max_vocab)
        
        if not self.pretrain or (self.pretrain and self.finetune):
            self.regression_head = RegressionHead(d_model)

        if self.classification:
            self.classification_head = ClassificationHead(d_model)
        
        self.dropout = nn.Dropout(p_dropout * 2)
        self.gelu = gelu

    def _setup_dna_encoder(self, d_model, max_vocab, max_len):
        """Initialize DNA sequence encoder."""
        self.dna_encoder = nn.Embedding(max_vocab, d_model)
        
        if self.refined_conv1d:
            self.refined_conv = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
            )
            self.refined_conv_residual = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=1),
                nn.BatchNorm1d(d_model)
            )
            self.protein_refined_conv = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
            )
            self.protein_refined_conv_residual = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=1),
                nn.BatchNorm1d(d_model)
            )
        
        self.dna_pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        self.embedding = Embeddings(d_model=d_model, max_vocab=max_vocab, max_len=max_len)

    def _setup_protein_encoder(self, d_model, freeze_layer, num_head, num_cross):
        """Initialize protein sequence encoder using ESM2."""
        if self.esm:
            self.protein_emb = ESMAsSchNetEmbedding(
                path=self.esm_path,
                freeze_n_layers=-1
            )
            if d_model == 1280:
                self.transform_linear = None
            else:
                self.transform_linear = nn.Linear(1280, d_model)
        else:
            self.protein_emb = nn.Embedding(len(ACIDS), d_model)
            self.protein_pos_encoder = PositionalEncoding(d_model, max_len=1024)

    def _setup_cross_attention_layers(self, d_model, num_head, num_cross):
        """Initialize cross-attention layers for DNA-protein interaction."""
        if self.dna_self_attn:
            self.dna_self_attention = nn.ModuleList([
                nn.ModuleDict({
                    'attention': nn.MultiheadAttention(d_model, num_head, batch_first=True),
                    'norm1': nn.LayerNorm(d_model),
                    'ffn': FeedForwardNetwork(d_model),
                    'norm2': nn.LayerNorm(d_model)
                }) for _ in range(3)
            ])
        
        if self.dna_attn:
            self.dna_cross_layers = nn.ModuleList([
                nn.ModuleDict({
                    'attention': nn.MultiheadAttention(d_model, num_head, batch_first=True),
                    'norm1': nn.LayerNorm(d_model),
                    'ffn': FeedForwardNetwork(d_model),
                    'norm2': nn.LayerNorm(d_model)
                }) for _ in range(num_cross)
            ])

        self.protein_cross_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.MultiheadAttention(d_model, num_head, batch_first=True),
                'norm1': nn.LayerNorm(d_model),
                'ffn': FeedForwardNetwork(d_model),
                'norm2': nn.LayerNorm(d_model)
            }) for _ in range(num_cross)
        ])

    def _apply_attention_block(self, x, block, residual=None, key=None, value=None, key_padding_mask=None):
        """Apply attention block with residual connection."""
        if residual is None:
            residual = x
        if key is None:
            key = x
        if value is None:
            value = x
            
        attn_output, attn_weights = block['attention'](
            x, key, value, key_padding_mask=key_padding_mask
        )
        x = residual + attn_output
        x = block['norm1'](x)
        
        residual = x
        x = block['ffn'](x)
        x = residual + x
        x = block['norm2'](x)
        
        return x, attn_weights

    def _convert_protein_seq_to_strings(self, protein_seq):
        """Convert protein sequence tensor to amino acid string list."""
        batch_size, seq_len = protein_seq.shape
        protein_sequences = []
        
        for i in range(batch_size):
            seq_indices = protein_seq[i].cpu().numpy()
            seq_indices = seq_indices.astype(np.int64)
            seq_string = ''.join([ACIDS[idx] for idx in seq_indices if idx < len(ACIDS) and idx > 0])
            protein_sequences.append(seq_string)
        
        return protein_sequences

    def _encode_dna(self, dna_seq, padding_mask):
        """Encode DNA sequence."""
        dna_seq = dna_seq.long()
        output = self.dna_encoder(dna_seq)
        output = self.dna_pos_encoder(output)
            
        if self.segment_embedding:
            output = self.dna_segment_embedding(output, 0)
        
        return output

    def _encode_protein(self, protein_seq):
        """Encode protein sequence using ESM2."""
        if self.esm:
            protein_sequences = self._convert_protein_seq_to_strings(protein_seq)
            batch_embeddings = self.protein_emb(protein_sequences)
            
            batch_size, seq_len, _ = batch_embeddings.shape
            device = batch_embeddings.device
            
            if seq_len < self.use_protein_len:
                padded_embeddings = torch.zeros(batch_size, self.use_protein_len, 1280, device=device)
                padded_embeddings[:, :seq_len, :] = batch_embeddings
                batch_embeddings = padded_embeddings
            elif seq_len > self.use_protein_len:
                if seq_len == self.use_protein_len:
                    start_idx = 0
                else:
                    start_idx = torch.randint(0, seq_len - self.use_protein_len, (1,)).item()
                batch_embeddings = batch_embeddings[:, start_idx:start_idx+self.use_protein_len, :]
            
            if self.transform_linear:
                batch_embeddings = self.transform_linear(batch_embeddings)
            
            if self.segment_embedding:
                batch_embeddings = self.protein_segment_embedding(batch_embeddings, 1)
            
            return batch_embeddings
        else:
            protein_seq = protein_seq.long()
            output = self.protein_emb(protein_seq)
            if self.segment_embedding:
                output = self.protein_segment_embedding(output, 1)
            return self.protein_pos_encoder(output)
    
    def _apply_cross_attention(self, layer_idx, protein_meaning, output, dna_mask, protein_mask, dna_reverse):
        """Apply cross-attention between protein and DNA representations."""
        suffix = "reverse" if dna_reverse else "forward"
        
        protein_scores = None
        dna_scores = None
        
        # Protein attends to DNA
        protein_meaning, protein_scores = self._apply_attention_block(
            protein_meaning, self.protein_cross_layers[layer_idx],
            key=output, value=output, key_padding_mask=~dna_mask
        )
        setattr(self, f"protein_scores_Layer{layer_idx}_{suffix}", protein_scores)
        
        # DNA attends to protein
        if self.dna_attn:
            output, dna_scores = self._apply_attention_block(
                output, self.dna_cross_layers[layer_idx],
                key=protein_meaning, value=protein_meaning, key_padding_mask=~protein_mask
            )
            setattr(self, f"dna_scores_Layer{layer_idx}_{suffix}", dna_scores)
        
        if layer_idx == len(self.protein_cross_layers) - 1:
            setattr(self, f"protein_scores_{suffix}", protein_scores)
            if self.dna_attn and dna_scores is not None:
                setattr(self, f"dna_scores_{suffix}", dna_scores)
            else:
                setattr(self, f"dna_scores_{suffix}", None)
        
        return protein_meaning, output

    def encode(self, dna_seq, protein_seq, dna_reverse=False):
        """Encode DNA and protein sequences with cross-attention."""
        dna_padding_mask = (dna_seq != 0)
        protein_padding_mask = (protein_seq[:, :self.use_protein_len] != 0) if protein_seq.size(1) > self.use_protein_len else (protein_seq != 0)

        output = self._encode_dna(dna_seq, dna_padding_mask)
        protein_meaning = self._encode_protein(protein_seq)
        
        for i in range(len(self.protein_cross_layers)):
            protein_meaning, output = self._apply_cross_attention(
                i, protein_meaning, output, dna_padding_mask, protein_padding_mask, dna_reverse
            )

        return protein_meaning, output

    def _encode_with_complement(self, dna_seq, protein_seq):
        """Encode DNA reverse complement sequence."""
        device = dna_seq.device
        dna_complement = dna_seq.clone()
        pad_token = 0
        batch_size, seq_len = dna_seq.size()

        is_pad = (dna_seq == pad_token)
        pad_pos = is_pad.float().argmax(dim=1)
        not_have_pad = is_pad.sum(dim=1) == 0
        pad_pos[not_have_pad] = seq_len

        # Create complement (A<->T, C<->G)
        for base_from, base_to in [(2, 3), (3, 2), (4, 5), (5, 4)]:
            mask = (dna_seq[:, 1:] == base_from)
            dna_complement[:, 1:][mask] = base_to

        valid_len = (pad_pos - 1).clamp(min=0)

        # Reverse the valid region
        tmp = dna_complement[:, 1:].clone()
        for i in range(batch_size):
            vl = valid_len[i].item()
            if vl > 0:
                tmp[i, :vl] = torch.flip(tmp[i, :vl], dims=[0])
        dna_complement[:, 1:] = tmp
        
        protein_meaning_rev, output_rev = self.encode(
            dna_complement, protein_seq, dna_reverse=True
        )
        return protein_meaning_rev, output_rev

    def forward(self, dna_seq, protein_seq):
        """
        Forward pass for binding affinity prediction.
        
        Args:
            dna_seq: DNA sequence tensor [batch, seq_len]
            protein_seq: Protein sequence tensor [batch, seq_len]
            
        Returns:
            Tuple of (regression_output, classification_output)
        """
        with torch.no_grad():
            dna_seq = dna_seq.float()
            protein_seq = protein_seq.float()

            # Forward strand encoding
            protein_meaning_forward, output_forward = self.encode(
                dna_seq, protein_seq, dna_reverse=False
            )
            
            # Reverse complement encoding
            protein_meaning_reverse, output_reverse = self._encode_with_complement(
                dna_seq, protein_seq
            )

            # Average both directions
            protein_meaning = (protein_meaning_forward + protein_meaning_reverse) / 2
            output = (output_forward + output_reverse) / 2

            if self.finetune_attn:
                output = self.finetune_attn_head(output, protein_meaning)
            
            self.protein_meaning = protein_meaning
            self.output = output
        
            regression_output = self.regression_head(protein_meaning, output)
            
            classification_output = None
            if self.classification:
                classification_output = self.classification_head(protein_meaning, output)

            return regression_output, classification_output

