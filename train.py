# %%
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, RandomSampler
import functools

from dataloader.multitargetdataloader import MultiTargetProcessor
from util.systems_settings import get_device, seed_everything

from model.Mymodel import BertForSequenceMultiTargetsClassification
from model.optimization import AdamW, get_linear_schedule_with_warmup #transformers
from model.losses import CenterLoss
from torch.optim import Adam
from util.evaluation import QNAI_accuracy_calc

from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# %%
root_path = "./"
model_name_or_path = root_path+"weights/phoBERTbase/"
data_dir = root_path+"datasets/QuiNhonAI"

seed_everything(369)
device = get_device()

max_seq_length = 64
max_context_length = 1 # ko thể khác 1
context_standalone = False
LR = 2e-6
n_epochs = 200
batch_size = 8
acum_step = 1
warmup_rate = 0.3
patient_num = 20

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
print(train_data.get_labels(), train_data.get_targets(), num_class, num_target, test_data, train_data)

# %%
from model.Mymodel import RobertaModel
from transformers import RobertaConfig
import torch
from torch import nn
from model.poolers import AttentionPooling
from model.losses import MultiSofmaxLoss, MultiBCELoss
from torch.nn import CrossEntropyLoss, BCELoss
from model.Mymodel import MyTopModel


class BertForSequenceMultiTargetsClassification(nn.Module):
    """Proposed Context-Aware Bert Model for Sequence Classification
    """
    def __init__(self, bert, num_labels, num_target, hidden_size, model_path=None, config=None, att_activate="tanh", use_center_loss=True):
        super(BertForSequenceMultiTargetsClassification, self).__init__()
        self.model_path = model_path
        self.num_labels = num_labels
        self.num_targets = num_target
        self.att_activate = att_activate
        self.hidden_size = hidden_size
        
        self.bert = bert
        self.dropout_in = nn.Dropout(0.3)
        self.arr_module = MyTopModel(self.num_targets, self.num_labels, self.hidden_size, self.att_activate)
        self.dropout_out = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

        if use_center_loss:
            self.center_loss_fct = CenterLoss(self.num_labels, self.hidden_size)
        else:
            self.center_loss_fct = None

    def forward(self, input_ids, attention_mask, label_ids=None, tau=1.0):
        sequence_output = self.bert(input_ids=input_ids, attention_mask = attention_mask)[0]
        
        sequence_output = self.dropout_in(sequence_output)
        arred_output, target_prob = self.arr_module(sequence_output)
        arred_output = self.dropout_out(arred_output)
        logits = self.classifier(arred_output)
        logits = logits*target_prob.unsqueeze(dim=-1)

        if label_ids is not None:
            target_labels = torch.where(label_ids == 0.0, 0.0, 1.0).to(label_ids.device)
            sentimnet_loss_fct = MultiSofmaxLoss(self.num_labels)
            target_loss_fct = MultiBCELoss()
            loss = {
                    "target_sentimnet_loss" : sentimnet_loss_fct(logits/tau, label_ids),
                    "target_loss" : target_loss_fct(target_prob, target_labels)
                   }
            if self.center_loss_fct is not None:
                loss["center_loss"] = self.center_loss_fct(arred_output.reshape(-1, self.hidden_size), label_ids.reshape(-1))
            return loss, logits, target_prob
        else:
            return logits


# Initializing a RoBERTa configuration
config = RobertaConfig.from_pretrained(model_name_or_path)
bert = RobertaModel(config, model_path=model_name_or_path, use_sru=True)
model = BertForSequenceMultiTargetsClassification(bert,num_labels=num_class, num_target=num_target, hidden_size=config.hidden_size)
count_parameters(model)

# %%
model.to(device)
model.train()

model_params = list(model.named_parameters()) # included all params from pooler and transformers
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
model_params = [{'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0001},
                {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
               ]

model_optimizer = AdamW(model_params, lr=LR)
model_scheduler = get_linear_schedule_with_warmup(model_optimizer,
                                                  num_warmup_steps=len(train_data_loader)*int(n_epochs)*warmup_rate, 
                                                  num_training_steps=len(train_data_loader)*n_epochs)

for n, p in list(model.bert.named_parameters()):
    if not any(nd in n for nd in ["bias", "LayerNorm.bias", "LayerNorm.weight"]):
        p.requires_grad = False

from tqdm import tqdm
writer = SummaryWriter("./runs")

step, e_step, best_F1, best_score = 0, 0, 0, 0
patient = 0
swith_trainable_mode_flag = False
# step, e_step, best_F1 = step, e_step, best_F1
tk_g = tqdm(range(n_epochs))
for e in tk_g:
    tk1 = tqdm(train_data_loader, leave=False)
    model.train()
    for batch in tk1:
        s_ids = {}
        feauture_tensor = train_data.convert_examples_to_features(batch, tokenizer, max_seq_length)
        s_ids["input_ids"] = feauture_tensor["input_ids"].to(device)
        s_ids["attention_mask"] = feauture_tensor["attention_mask"].to(device)
        s_ids["label_ids"] = feauture_tensor["label_ids"].to(device)
        
        # if step == 0:
        #     inp = (s_ids["input_ids"], s_ids["attention_mask"], s_ids["label_ids"])
        #     writer.add_graph(model, inp)
        #     model_optimizer.zero_grad()

        loss, logits, targets_pred  = model(**s_ids)
        loss = 0.5*loss["target_sentimnet_loss"] + 0.5*loss["target_loss"] + 0.0000001*loss["center_loss"]
        loss.backward()

        if ((step%acum_step == 0) and step!=0) or s_ids["label_ids"].shape[0] < batch_size:
            model_optimizer.step()
            model_scheduler.step()
            model_optimizer.zero_grad()

        with torch.no_grad():
            sentence_predict = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
            sentence_labels = s_ids["label_ids"]

            target_labels = torch.where(sentence_labels == 0.0, 0.0, 1.0)
            target_pred = torch.where(targets_pred <= 0.7, 0.0, 1.0)
            
            flatten_labels = sentence_labels.reshape(-1).cpu().detach().numpy()
            flatten_predicts = sentence_predict.reshape(-1).cpu().detach().numpy()
            f1_sentiments = f1_score(flatten_labels, flatten_predicts, average='macro')

            target_labels_flatten = target_labels.reshape(-1).cpu().detach().numpy()
            target_pred_flatten = target_pred.reshape(-1).cpu().detach().numpy()
            f1_target=f1_score(target_labels_flatten, target_pred_flatten, average='macro')

            final_score = QNAI_accuracy_calc(sentence_predict, sentence_labels)

            tk1.set_postfix(Epoch=e, step=step, loss=loss.data.item(),
                            F1_sentiments_macro=f1_sentiments, F1_targets=f1_target, 
                            score=final_score)
            writer.add_scalar('train/loss', loss.data.item(), step)
            writer.add_scalar('train/F1_sentiments_macro', f1_sentiments, step)
            writer.add_scalar('train/F1_target_macro', f1_target, step)
            writer.add_scalar('train/score', final_score, step)
        step += 1
        
    tk2 = tqdm(test_data_loader, leave=False)
    model.eval()
    with torch.no_grad():
        sentence_predicts_dev, sentence_labels_dev = [], []
        for dev_batch in tk2:
            s_ids = {}
            feauture_tensor = train_data.convert_examples_to_features( dev_batch, tokenizer, max_seq_length)
            s_ids["input_ids"] = feauture_tensor["input_ids"].to(device)
            s_ids["attention_mask"] = feauture_tensor["attention_mask"].to(device)
            if quasi_model:
                s_ids["context_ids"] = feauture_tensor["context_ids"].to(device)
            label_ids = feauture_tensor["label_ids"].to(device)
            
            logits  = model(**s_ids)
            
            predict = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
            sentence_predicts_dev.append(predict)
            sentence_labels_dev.append(label_ids)
            
            tk2.set_postfix(Epoch=e, step=e_step, state="eval")

        sentence_labels_dev = torch.concat(sentence_labels_dev, dim=0)
        sentence_predicts_dev = torch.concat(sentence_predicts_dev, dim=0)
        final_score_dev = QNAI_accuracy_calc(sentence_predicts_dev, sentence_labels_dev)

        flatten_labels_dev = sentence_labels_dev.reshape(-1).cpu().detach().numpy()
        flatten_predicts_dev = sentence_predicts_dev.reshape(-1).cpu().detach().numpy()
        f1_sentiment_dev = f1_score(flatten_labels_dev, flatten_predicts_dev, average='macro')

        writer.add_scalar('dev/F1 sentiments macro', f1_sentiment_dev, e_step)
        writer.add_scalar('dev/score', final_score_dev, e_step)
        
        if best_F1 < f1_sentiment_dev:
            patient = 0
            best_F1 = f1_sentiment_dev
            best_score = final_score_dev
            torch.save(model.state_dict(), f"./best.bin")
        else:
            patient += 1
            if patient == patient_num:
                torch.save(model.state_dict(), f"./last.bin")
                break
        if (f1_sentiment_dev >= 0.5 or e >= int(n_epochs*0.65)) and not swith_trainable_mode_flag:
            swith_trainable_mode_flag = True
            print(f"Trainable BERT model in {e} epoch with F1:{f1_sentiment_dev}, score:{final_score_dev} in dev set.")
            for n, p in list(model.bert.named_parameters()):
                if not any(nd in n for nd in ["bias", "LayerNorm.bias", "LayerNorm.weight"]):
                    p.requires_grad = True
        tk_g.set_postfix(Epoch=e, dev_F1_macro=f1_sentiment_dev, dev_score=final_score_dev, best_F1 = best_F1, best_score=best_score, patient=patient)
        e_step += 1


