#!/home/lin/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/root/miniconda3/envs/aloha/bin/python
#!/home/lin/miniconda3/envs/aloha/bin/python
"""

import os
import numpy as np
import argparse
import json
from transformers import BertTokenizer, BertModel
from pathlib import Path
import sys
sys.path.append("../")
from models.multimodal_encoder.t5_encoder import T5Embedder
import torch


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruction', action='store', type=str, help='instruction',
                        default="null", required=False)
    parser.add_argument('--targetDir', action='store', type=str, help='targetDir',
                        default="instruction.npy", required=False)
    parser.add_argument('--encoderDir', action='store', type=str, help='encoderDir',
                        default='/home/agilex/models/google/t5-v1_1-xxl', required=False)  # 'wheel'
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    device=torch.device(f"cuda:0")
    embedder = T5Embedder(
    from_pretrained=args.encoderDir,
    model_max_length=1024,
    device=device
    )
    tokenizer, encoder = embedder.tokenizer, embedder.model
    result = tokenizer(
        args.instruction, return_tensors="pt",
        padding="longest",
        truncation=True
    )
    with torch.no_grad():
        output = encoder(result['input_ids'].to(device), result['attention_mask'].to(device))["last_hidden_state"]
    data_dict = dict()
    data_dict[f'vector'] = output[0].detach().to(torch.float32).cpu().numpy()
    data_dict[f'input_ids'] = result['input_ids'][0].detach().cpu().numpy()
    data_dict[f'attention_mask'] = result['attention_mask'][0].detach().cpu().numpy()
    np.save(args.targetDir, data_dict)
    print("Done")


if __name__ == '__main__':
    main()
