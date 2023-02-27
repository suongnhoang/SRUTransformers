#!/usr/bin/env python3

"""Processors for different tasks."""
import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import copy
import codecs as cd
import numpy as np
import operator
from .datasetloader_utils import ExampleStruct


class MultiTargetProcessor(Dataset):
    """Processor for the Sentihood data set."""
    def __init__(self, dataset, data_dir, 
                 random_mask=False):
        super().__init__()
        self.dataset = dataset.lower()
        assert self.dataset in ["quinhon"],\
            'dataset required, one of ["quinhon"]'

        self.random_mask = random_mask
        self.data_dir = data_dir
        self.extened_type = "csv"
        if self.dataset == "quinhon":
            self.text_colums = "Review"

    def load_label_map(self):
        lb_path = os.path.join(self.data_dir, "labels.txt")
        label_map = {}
        with cd.open(lb_path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip().split(":")
                label_text = line[0].strip().lower()
                label_id = int(line[1].strip())
                label_map[label_text] = label_id
        return label_map


    def set_type(self, set_type):
        set_type = set_type.lower()
        assert set_type in ["train", "dev", "test"]
        self.__type = set_type
        self.data = pd.read_csv(os.path.join(self.data_dir, f"{self.__type}_multitarget.{self.extened_type}"))
        self.context_id_map = self.get_targets()
        self.label_map = self.load_label_map()
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        class_repr = self.__type
        return class_repr+"\n"+ super().__repr__()
    
    def get_labels(self):
        """See base class."""
        return list(self.load_label_map().keys())
    
    def get_targets(self):
        targets_col = self.data.drop(columns=[self.text_colums])
        return dict([(v.lower(),k) for k,v in enumerate(targets_col.columns.tolist())])

    def get_examples(self):
        """See base class."""
        data = self.run_data.values
        return self._create_examples(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = row[self.text_colums].lower()
        label = np.array(row.values[1:].astype(np.float))
        return {"text":text, "label":label}

    def convert_examples_to_features(self, examples, tokenizer, max_seq_length):
        """Loads a data file into a list of `InputBatch`s."""

        batch_sentences, batch_labels = [],[]
        for example in examples:
            example = ExampleStruct(**example)
            batch_sentences.append(example.text)
            batch_labels.append(example.label)
        
        features = {"input_ids": None,
                    "attention_mask": None,
                    "label_ids": None,
                    }
        encoded = tokenizer.batch_encode_plus(batch_sentences, max_length=max_seq_length, padding='max_length', truncation=True)
        features["input_ids"] = torch.LongTensor(encoded["input_ids"])
        features["attention_mask"] = torch.LongTensor(encoded["attention_mask"])
        features["label_ids"] = torch.LongTensor(np.array(batch_labels))
        return features