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
import h5py
sys.path.append("../")
from models.multimodal_encoder.t5_encoder import T5Embedder
import torch
import glob

embedder = None
tokenizer = None
encoder = None

class Operator:
    def __init__(self, args, hdf5_dir):

        self.args = args
        self.hdf5_dir = hdf5_dir

        self.device = torch.device(f"cuda:0")
        global embedder
        global tokenizer
        global encoder
        if embedder is None:
            # tokenizer = BertTokenizer.from_pretrained(self.args.encoderDir)
            # encoder = BertModel.from_pretrained(self.args.encoderDir)

            embedder = T5Embedder(
                from_pretrained=self.args.encoderDir,
                model_max_length=1024,
                device=self.device
            )
            tokenizer, encoder = embedder.tokenizer, embedder.model

        self.tokenizer, self.encoder = tokenizer, encoder

    def process(self):
        data_dict = {}
        data_dict[f'instructions/full_instructions/input_ids'] = []
        data_dict[f'instructions/full_instructions/attention_mask'] = []
        data_dict[f'instructions/full_instructions/vector'] = []
        with h5py.File(self.hdf5_dir, 'r') as root:
            try:
                for text in root['instructions/full_instructions/text'][()]:
                    text = text.decode('utf-8')
                    # result = self.tokenizer.encode_plus(
                    #     text,                     # 输入文本
                    #     return_attention_mask=True,  # 返回 attention mask
                    #     add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                    #     padding='max_length',
                    #     truncation=True,
                    #     max_length=32,
                    #     return_tensors='pt',     # 返回 pytorch tensor 格式的数据
                    # )
                    # output = self.encoder(result['input_ids'], result['attention_mask'])["last_hidden_state"]

                    result = self.tokenizer(
                        text, return_tensors="pt",
                        padding="longest",
                        truncation=True
                    )
                    with torch.no_grad():
                        output = self.encoder(result['input_ids'].to(self.device), result['attention_mask'].to(self.device))["last_hidden_state"]

                    data_dict[f'instructions/full_instructions/vector'].append(output[0].detach().to(torch.float32).cpu().numpy())
                    data_dict[f'instructions/full_instructions/input_ids'].append(result['input_ids'][0].detach().cpu().numpy())
                    data_dict[f'instructions/full_instructions/attention_mask'].append(result['attention_mask'][0].detach().cpu().numpy())
            except Exception as e:
                pass
            try:
                for start_time, end_time in zip(root['instructions/segment_instructions/start_time'][()], root['instructions/segment_instructions/end_time'][()]):
                    start_time = start_time
                    end_time = end_time
                    data_dict[f'instructions/segment_instructions/{start_time}-{end_time}/vector'] = []
                    data_dict[f'instructions/segment_instructions/{start_time}-{end_time}/input_ids'] = []
                    data_dict[f'instructions/segment_instructions/{start_time}-{end_time}/attention_mask'] = []
                    for text in root[f'instructions/segment_instructions/{start_time}-{end_time}/text'][()]:
                        text = text.decode('utf-8')

                        # result = self.tokenizer.encode_plus(
                        #     text,                     # 输入文本
                        #     return_attention_mask=True,  # 返回 attention mask
                        #     add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                        #     padding='max_length',
                        #     truncation=True,
                        #     max_length=32,
                        #     return_tensors='pt',     # 返回 pytorch tensor 格式的数据
                        # )
                        # output = self.encoder(result['input_ids'], result['attention_mask'])["last_hidden_state"]

                        result = self.tokenizer(
                            text, return_tensors="pt",
                            padding="longest",
                            truncation=True
                        )
                        with torch.no_grad():
                            output = self.encoder(result['input_ids'], result['attention_mask'])["last_hidden_state"]

                        data_dict[f'instructions/segment_instructions/{start_time}-{end_time}/vector'].append(output[0].detach().cpu().numpy())
                        data_dict[f'instructions/segment_instructions/{start_time}-{end_time}/input_ids'].append(result['input_ids'][0].detach().cpu().numpy())
                        data_dict[f'instructions/segment_instructions/{start_time}-{end_time}/attention_mask'].append(result['attention_mask'][0].detach().cpu().numpy())
            except Exception as e:
                pass
        with h5py.File(self.hdf5_dir, 'a') as root:
            for key in data_dict:
                if key in root:
                    del root[key]
                root.create_dataset(key, data=data_dict[key])


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetDir', action='store', type=str, help='datasetDir',
                        default="/home/agilex/data", required=False)
    parser.add_argument('--encoderDir', action='store', type=str, help='encoderDir',
                        default='/home/agilex/models/google/t5-v1_1-xxl', required=False)  # 'wheel'
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    hdf5_files = []
    for f in os.listdir(args.datasetDir):
        if f.endswith(".hdf5"):
            hdf5_files.append(os.path.join(args.datasetDir, f))
        if os.path.isdir(os.path.join(args.datasetDir, f)):
            hdf5_files.extend(glob.glob(os.path.join(args.datasetDir, f, "*.hdf5")))
    for hdf5_file in hdf5_files:
        print(f"{hdf5_file} processing")
        operator = Operator(args, hdf5_file)
        operator.process()
        print(f"{hdf5_file} done")
    print("Done")


if __name__ == '__main__':
    main()
