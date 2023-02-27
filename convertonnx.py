import os
import torch.onnx
import torch.nn as nn 

# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, RandomSampler
import functools

from dataloader.multitargetdataloader import MultiTargetProcessor
from util.systems_settings import get_device, seed_everything

from model.Mymodel import BertForSequenceMultiTargetsClassification
from tqdm import tqdm
from sklearn.metrics import f1_score
from util.evaluation import QNAI_accuracy_calc


root_path = "./"
model_name_or_path = root_path+"models/phoBERTbase/"
data_dir = root_path+"datasets/QuiNhonAI"

seed_everything(369)
device = get_device()

max_seq_length = 256
max_context_length = 1 # ko thể khác 1
context_standalone = False
LR = 2e-6
n_epochs = 100
batch_size = 8
acum_step = 1
warmup_rate = 0.3
is_roberta = True
quasi_model = False
balance_load = True
att_activate = "tanh"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=None)

semval_processor = MultiTargetProcessor(dataset="QuiNhon", data_dir=data_dir)
train_data = semval_processor.set_type("train")
test_data = semval_processor.set_type("test")

model_collate_fn = functools.partial(lambda x: x)

sampler = RandomSampler(train_data)
train_data_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, collate_fn=model_collate_fn)

sampler = RandomSampler(test_data)
test_data_loader = DataLoader(test_data, batch_size=batch_size, sampler=sampler, collate_fn=model_collate_fn)

num_class = len(train_data.get_labels())
num_target = len(train_data.get_targets())

model = BertForSequenceMultiTargetsClassification(num_labels = num_class, num_target=num_target, model_path = model_name_or_path, 
                                                init_weight=True, att_activate = att_activate, 
                                                context_size = num_target, log_full=False, quasi_model=quasi_model, use_center_loss=False)
model.load_state_dict(torch.load("./models/check_points/best_no_centerloss.bin", map_location=torch.device('cpu')))
model.to(device)
model.eval()

def Convert_ONNX(model, data_input, logs_dir = "./logs"):
    # Export the model
    os.makedirs(logs_dir, exist_ok=True)
    torch.onnx.export(model,               # model being run
                    (data_input["input_ids"], data_input["attention_mask"]), # model input (or a tuple for multiple inputs)
                    os.path.join(logs_dir, "model.onnx"),   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=12,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input_ids', 'attention_mask'],   # the model's input names
                    output_names = ['logits'], # the model's output names
                    dynamic_axes={'input_ids' : {0 : 'batch_size', 1: 'sequence_len'},    # variable length axes
                                    'attention_mask' : {0 : 'batch_size', 1: 'sequence_len'},
                                    'logits' : {0 : 'batch_size'}
                                }
                    )

e_step = 0
tk2 = tqdm(test_data_loader)
with torch.no_grad():
    sentence_predicts_dev, sentence_labels_dev = [], []
    for dev_batch in tk2:
        s_ids = {}
        feauture_tensor = train_data.convert_examples_to_features( dev_batch, tokenizer, max_seq_length)
        s_ids["input_ids"] = feauture_tensor["input_ids"].to(device)
        s_ids["attention_mask"] = feauture_tensor["attention_mask"].to(device)
        label_ids = feauture_tensor["label_ids"].to(device)
        
        logits  = model(**s_ids)
        
        predict = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
        sentence_predicts_dev.append(predict)
        sentence_labels_dev.append(label_ids)
        
        tk2.set_postfix(Epoch=0, step=e_step, state="eval")

    sentence_labels_dev = torch.concat(sentence_labels_dev, dim=0)
    sentence_predicts_dev = torch.concat(sentence_predicts_dev, dim=0)
    final_score_dev = QNAI_accuracy_calc(sentence_predicts_dev, sentence_labels_dev)

    flatten_labels_dev = sentence_labels_dev.reshape(-1).cpu().detach().numpy()
    flatten_predicts_dev = sentence_predicts_dev.reshape(-1).cpu().detach().numpy()
    f1=f1_score(flatten_labels_dev, flatten_predicts_dev, average='macro')
    print(f1, final_score_dev)

for batch in train_data_loader:
    break
s_ids = {}
feauture_tensor = train_data.convert_examples_to_features(batch, tokenizer, max_seq_length)
s_ids["input_ids"] = feauture_tensor["input_ids"].to(device)
s_ids["attention_mask"] = feauture_tensor["attention_mask"].to(device)
if quasi_model:
    s_ids["context_ids"] = feauture_tensor["context_ids"].to(device)
Convert_ONNX(model, s_ids, logs_dir = "./checkpoints")