#!/usr/bin/env python
"""
Seq2Kd Inference Script

This script provides inference functionality for the Seq2Kd protein-DNA
binding affinity prediction model.
"""

import os
import glob
import click
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
from ruamel.yaml import YAML

from models.datasets import ProteinDNAGeneratedDataset
from models.seq2kd import Seq2Kd
from models.common import custom_collate_fn


# Vocabulary constants
ACIDS = '0XACDEFGHIKLMNPQRSTVWY'
WORD2IDX = {'[PAD]': 0, '[MASK]': 1, 'A': 2, 'T': 3, 'C': 4, 'G': 5, '-': 6, 'x': 7}
VOCAB = {idx: word for word, idx in WORD2IDX.items()}


class Seq2KdPredictor:
    """
    Seq2Kd model predictor for protein-DNA binding affinity.
    
    Args:
        model_path: Path to model checkpoint
        model_config: Model configuration dictionary
        device: Computation device ('cuda' or 'cpu')
    """
    
    def __init__(self, model_path, model_config, device='cuda'):
        self.device = device
        self.output_dir = None
        
        # Load model
        self.model = Seq2Kd(**model_config['model']).to(device)
        checkpoint = torch.load(model_path, map_location='cuda')
        state_dict = {
            k.replace('module.', ''): v for k, v in checkpoint.items()
        }
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def predict_dataset(self, data_config, input_path, output_dir):
        """
        Run inference on a dataset.
        
        Args:
            data_config: Data configuration dictionary
            input_path: Path to input file or directory
            output_dir: Output directory for results
            
        Returns:
            DataFrame with prediction results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Handle input path
        if os.path.isdir(input_path):
            path_list = list(glob.iglob(input_path + '/*.txt', recursive=True))
        else:
            path_list = [input_path]

        for path in path_list:
            file_name = os.path.basename(path).replace('.txt', '')
            output_path = os.path.join(output_dir, f"{file_name}.csv")

            print(f"Processing file: {path}")
            print(f"Output path: {output_path}")

            # Load data
            df = pd.read_csv(path, sep=' ', header=None, names=['dna_seq', 'protein_seq'])
            test_data = list(df.itertuples(index=False, name=None))
            
            test_loader = DataLoader(
                ProteinDNAGeneratedDataset(test_data, **data_config['padding']),
                batch_size=128,
                shuffle=False,
                num_workers=4,
                collate_fn=custom_collate_fn
            )

            results = []
            with torch.no_grad():
                for inputs in tqdm(test_loader, desc="Inference"):
                    input_ids, segment_ids = inputs

                    regression_output, classification_output = self.model(
                        input_ids.to(self.device),
                        segment_ids.to(self.device),
                    )

                    # Get classification probabilities
                    class_probs = torch.nn.functional.softmax(classification_output, dim=1).cpu()
                    predicted_classes = classification_output.argmax(dim=1).cpu()

                    # Class mapping
                    class_mapping = {0: -1, 1: 0, 2: 1}

                    for i in range(len(input_ids)):
                        protein_seq = ''.join([ACIDS[idx.item()] for idx in segment_ids[i] if idx.item() != 0])
                        dna_seq = ''.join([VOCAB[idx.item()] for idx in input_ids[i] if idx.item() not in [0, 1]])

                        probs = class_probs[i].tolist()
                        pred_class = class_mapping[predicted_classes[i].item()]

                        results.append({
                            'protein_sequence': protein_seq,
                            'dna_sequence': dna_seq,
                            'predicted_value': regression_output[i].item(),
                            'predicted_class': pred_class,
                            'prob_class_neg': probs[0],
                            'prob_class_pos': probs[1],
                        })

            # Save results
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)

        return df


@click.command()
@click.option('--model-path', type=click.Path(exists=True), required=True, help='Path to model checkpoint')
@click.option('--config-path', type=click.Path(exists=True), required=True, help='Path to config file')
@click.option('--input-path', type=click.Path(exists=True), required=True, help='Path to input file or directory')
@click.option('--output-dir', type=click.Path(), required=True, help='Output directory path')
def main(model_path, config_path, input_path, output_dir):
    """Seq2Kd Protein-DNA Binding Prediction Tool"""
    # Load configuration
    yaml = YAML(typ='safe')
    config = yaml.load(Path(config_path))

    # Initialize predictor
    predictor = Seq2KdPredictor(model_path, config)

    # Run prediction
    predictor.predict_dataset(config, input_path, output_dir)

    print(f"Prediction complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
