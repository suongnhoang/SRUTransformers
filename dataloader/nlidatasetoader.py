#!/usr/bin/env python3

"""Processors for different tasks."""
import os
import pandas as pd
import six
from torch.utils.data import Dataset
from .datasetloader_utils import ExampleStruct, convert_example_to_feature
import torch
import copy
import codecs as cd
import numpy as np
import operator



def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python 3")


class NLI_M_Processor(Dataset):
    """Processor for the Sentihood data set."""
    def __init__(self, dataset, data_dir, 
                 balance_load=False,
                 random_mask=False):
        super().__init__()
        self.dataset = dataset.lower()
        assert self.dataset in ["semeval", "visd4sa", "sentihood", "quinhon"],\
            'dataset required, one of ["semeval", "visd4sa", "sentihood", "quinhon"]'

        self.balance_load = balance_load
        self.random_mask = random_mask

        self.data_dir = data_dir

        self.extened_type = "csv" if self.dataset in ["semeval", "visd4sa", "quinhon"] else "tsv"
        self.order = [3,2,1] if self.dataset == "semeval" else [1,2,3] #sentihood and visd4sa have same orders


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
        self.data = pd.read_csv(os.path.join(self.data_dir, f"{self.__type}_NLI_M.{self.extened_type}"),sep="\t")
        self.context_id_map = self.get_targets()
        self.label_map = self.load_label_map()
        self.data_class_len = self.create_data_len()
        self.refresh_data()
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        class_repr = self.__type
        class_repr += "\n"+str(self.data_class_len)
        return class_repr+"\n"+ super().__repr__()

    def create_data_len(self):
        label_col_id = self.order[-1]
        label_col_name = self.data.columns[label_col_id]
        labels_count, compare_scores = {}, {}

        for label in self.label_map.keys():
            num_count = len(self.data[self.data[label_col_name] == label])
            labels_count[label] = num_count
        labels_count = dict( sorted(labels_count.items(), key=operator.itemgetter(1),reverse=True))

        if self.balance_load and self.__type == "train":
            labels = list(labels_count.keys())
            for label in copy.deepcopy(labels):
                row_ = []
                for x in labels:
                    if label != x:
                        row_.append(labels_count[label]/labels_count[x] if labels_count[x]!= 0 else 0.0)
                compare_mean_score = np.array(row_).mean()
                if len(labels) not in list(compare_scores.keys()):
                    compare_scores[len(labels)] = [{label:compare_mean_score}]
                else:
                    compare_scores[len(labels)].append({label:compare_mean_score})
                if compare_mean_score > 2.0:
                    labels.remove(label)
            group_len = np.array([len(compare_scores[x]) for x in compare_scores])
            max_group = [list(x.keys())[0] for x in compare_scores[list(compare_scores.keys())[np.argmax(group_len)]]]
            max_group_max_len = np.array([labels_count[x] for x in max_group]).max()
            
            for x in labels_count.keys():
                if labels_count[x] > max_group_max_len:
                    labels_count[x] = max_group_max_len
        return labels_count

    def refresh_data(self):
        label_col_id = self.order[-1]
        label_col_name = self.data.columns[label_col_id]
        data = []
        for label in self.data_class_len:
            samples_num = self.data_class_len[label]
            data.append(self.data[self.data[label_col_name]==label].sample(samples_num))
        self.run_data = pd.concat(data)
        self.countdown = len(self.run_data)
        if not self.verify_run_data():
            raise

    def verify_run_data(self):
        label_col_id = self.order[-1]
        label_col_name = self.run_data.columns[label_col_id]
        for label in self.label_map.keys():
            if self.data_class_len[label] != len(self.run_data[self.run_data[label_col_name] == label]):
                return False
        return True
    
    def get_labels(self):
        """See base class."""
        label_col_id = self.order[-1]
        label_col_name = self.data.columns[label_col_id]
        return list(self.data[label_col_name].unique())
    
    def get_targets(self):
        targets_col_id = self.order[-2]
        targets_col_name = self.data.columns[targets_col_id]
        return dict([(v.lower(),k) for k,v in enumerate(self.data[targets_col_name].unique())])


    def get_examples(self):
        """See base class."""
        data = self.run_data.values
        return self._create_examples(data)

    def create_example(self, line, i=0):
        guid = "%s-%s" % (self.__type, i)
        self.text_a, self.text_b, self.label = None, None, None
        for idx, var in zip(self.order,["text_a", "text_b", "label"]):
            setattr(self,var,convert_to_unicode(str(line[idx])))
        return {"guid":guid, "text_a":self.text_a, "text_b":self.text_b, "label":self.label}

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            example = self.create_example(line, i)
            examples.append(example)
            del(self.text_a, self.text_b, self.label)
        return examples

    def __len__(self):
        return len(self.run_data)

    def __getitem__(self, index):
        data = self.run_data.values[index]
        example = self.create_example(data, index)
        self.countdown -= 1
        if self.countdown == 0:
            self.refresh_data()
        return example

    def convert_examples_to_features(self, examples, tokenizer, max_seq_length,
                                    max_context_length=1, 
                                    context_standalone=False, is_roberta=False):
        """Loads a data file into a list of `InputBatch`s."""
        features = {"input_ids": [],
                    "attention_mask": [],
                    "token_type_ids": [],
                    "context_ids": [],
                    "label_ids": [],
                    "seq_len": [],
                    }
        for example in examples:
            example = ExampleStruct(**example)
            input_ids, attention_mask, token_type_ids, context_ids = convert_example_to_feature(example, self.context_id_map,
                                                                                    max_seq_length, tokenizer, max_context_length,
                                                                                    context_standalone, is_roberta)
            seq_len = len(input_ids)
            label_ids = self.label_map[example.label]

            features["input_ids"].append(input_ids)
            features["attention_mask"].append(attention_mask)
            features["token_type_ids"].append(token_type_ids)
            features["context_ids"].append(context_ids)
            features["label_ids"].append(label_ids)
            features["seq_len"].append(seq_len)

        for key in features.keys():
            features[key] = torch.LongTensor(features[key])
        return features