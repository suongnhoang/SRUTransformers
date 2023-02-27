import re
from typing import OrderedDict
import torch
import torch.nn as nn

from .config import BertConfig
from .poolers import AttentionPooling

from .losses import CenterLoss, MultiSofmaxLoss, MultiBCELoss
from .QACGPhoBERT import RobertaModel
from .QACGPhoBERT import init_weights
from .utils import dot_attention_function, cos_sim

from transformers import AutoModel, AutoConfig


class BertForSequenceMultiTargetsClassification(nn.Module):
    """Proposed Context-Aware Bert Model for Sequence Classification
    """
    def __init__(self, num_labels, num_target, model_path=None, config=None, init_weight=False,
                att_activate="tanh", 
                context_size = None, log_full = False, quasi_model=False, use_center_loss=True):
        super(BertForSequenceMultiTargetsClassification, self).__init__()
        assert model_path is not None or config is not None, "model_path and config can not be None at the same time"
        self.log_full = log_full
        self.model_path = model_path
        self.quasi_model = quasi_model
        self.num_labels = num_labels
        self.num_targets = num_target
        self.att_activate = att_activate
        self.config = self.create_config(model_path, config)
        self.config.context_size = context_size
        self.create_model(num_labels, init_weight)
        if use_center_loss:
            self.center_loss_fct = CenterLoss(self.num_labels, self.config.hidden_size)
        else:
            self.center_loss_fct = None
        
        

    def create_config(self, model_path, config):
        if model_path is not None:
            if self.quasi_model:
                config = self.__load_config()
            else:
                config = AutoConfig.from_pretrained(model_path)
        elif isinstance(config, BertConfig):
            config  = config
        return config

    def create_model(self, num_labels, init_weight):
        if self.quasi_model:
            self.bert = RobertaModel(self.config)
        else:
            self.bert = AutoModel.from_pretrained(self.model_path, config=self.config, cache_dir=None)
        self.dropout_in = nn.Dropout(self.config.hidden_dropout_prob)
        self.arr_module = MyTopModel(self.num_targets, self.num_labels, self.config.hidden_size, self.att_activate)
        self.dropout_out = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        init_weights.set_config(self.config)
        if init_weight:
            self.init_weights()
        self.load_trained_weights()

    def __load_config(self):
        config  = BertConfig().from_json_file(self.model_path+"config.json")
        self.config = config
        return self.config
        
    def load_trained_weights(self):
        model_prefix = f"^{self.config.model_type}."
        checkpoint = OrderedDict([(re.sub(model_prefix,"", k), v) \
                                    for k, v in  torch.load(self.model_path+"pytorch_model.bin", map_location='cpu').items()\
                                        if re.sub(model_prefix,"", k)])
        print(f"[INFO]--LOADING: loading {len(checkpoint)} weights from checkedpoint file.")
        incompatibleKeys = self.bert.load_state_dict(checkpoint, strict=False)
        unexpected_keys = incompatibleKeys.unexpected_keys
        missing_keys = incompatibleKeys.missing_keys
        if len(unexpected_keys):
            log_string = f"We found {len(unexpected_keys)} {f'unexpected_keys='.split('=')[0]}, "
            log_string += f"try to init new params for {unexpected_keys[:3]}..." if not self.log_full \
                        else f"try to init new params for {unexpected_keys}"
            print(log_string)
        if len(missing_keys):
            log_string = f"We found {len(missing_keys)} {f'missing_keys='.split('=')[0]}, "
            log_string += f"try to init new params for {missing_keys[:3]}..." if not self.log_full \
                        else f"try to init new params for {missing_keys}"
            print(log_string)
    
    def init_weights(self):
        """Initialize the weights"""
        print("Initializing new weights...")
        self.apply(init_weights)
        #######################################################################
        # TODO: initialize the weights to be a close diagonal identity matrix
        # Let's do special handling of initialization of newly added layers
        # for newly added layer, we want it to be "invisible" to the training process in the beginning and slightly diverge from the
        # original model. To illustrate, let's image we add a linear layer in the middle of a pretrain network. If we initialize the weight
        # randomly by default, then it will effect the output of the pretrain model largely so that it lost the point of importing pretrained weights.
        #
        # What we do instead is that we initialize the weights to be a close diagonal identity matrix, so that, at the beginning, for the 
        # network, it will be bascially copying the input hidden vectors as the output, and slowly diverge in the process of fine tunning. 
        # We turn the bias off to accomedate this as well.
        if self.model_path is not None and self.quasi_model:
            init_perturbation = 1e-2
            for layer_module in self.bert.encoder.layer:
                layer_module.attention.self.lambda_q_context_layer.weight.data.normal_(mean=0.0, std=init_perturbation)
                layer_module.attention.self.lambda_k_context_layer.weight.data.normal_(mean=0.0, std=init_perturbation)
                layer_module.attention.self.lambda_q_query_layer.weight.data.normal_(mean=0.0, std=init_perturbation)
                layer_module.attention.self.lambda_k_key_layer.weight.data.normal_(mean=0.0, std=init_perturbation)

    def __resize_embeddings(self, new_num_tokens=None, embeddings_type=None):
        if self.quasi_model:
            assert embeddings_type in ["word_embeddings","position_embeddings","token_type_embeddings"], \
                        "embeddings_type must be 'word_embeddings', 'position_embeddings' or 'token_type_embeddings'"
            _ = self.bert.resize_embeddings(new_num_tokens, embeddings_type)
            # Update base model and current model config
            if embeddings_type == "word_embeddings":
                self.config.vocab_size = new_num_tokens
            elif embeddings_type == "position_embeddings":
                self.config.max_position_embeddings = new_num_tokens
            elif embeddings_type == "token_type_embeddings":
                self.config.type_vocab_size = new_num_tokens
        else:
            print("Warning: __resize_embeddings is not support for huggingface model.")
            pass

    def resize_word_embeddings(self, new_num_tokens=None):
        self.__resize_embeddings(new_num_tokens, "word_embeddings")

    def resize_position_embeddings(self, new_num_tokens=None):
        self.__resize_embeddings(new_num_tokens, "position_embeddings")

    def resize_token_type_embeddings(self, new_num_tokens=None):
        self.__resize_embeddings(new_num_tokens, "token_type_embeddings")

    def forward(self, input_ids, attention_mask, context_ids=None, label_ids=None, tau=1.0):
        if self.quasi_model:
            sequence_output, _ = self.bert(input_ids=input_ids, attention_mask = attention_mask, context_ids = context_ids)
        else:
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
                loss["center_loss"] = self.center_loss_fct(arred_output.reshape(-1, self.config.hidden_size), label_ids.reshape(-1))
            return loss, logits, target_prob
        else:
            return logits


class MacaronFFN(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.L_1 = nn.Linear(hidden_dim, hidden_dim*4)
        self.L_2 = nn.Linear(hidden_dim*4, hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.L_1(x))
        x = self.L_2(x)
        return x

class MacaronAttention(nn.Module):
    def __init__(self, hidden_dim, att_activate="tanh", eps=1e-05) -> None:
        super().__init__()

        self.Wq = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.Wk = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.Wv = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.att_activate = att_activate
        self.att_laynorm = nn.LayerNorm(hidden_dim, eps=eps)

        self.FFN1 = MacaronFFN(hidden_dim)
        self.FFN1_laynorm = nn.LayerNorm(hidden_dim, eps=eps)
        self.FFN2 = MacaronFFN(hidden_dim)
        self.FFN2_laynorm = nn.LayerNorm(hidden_dim, eps=eps)

    def forward(self, query, key, value):
        # query
        q = self.FFN1(query)
        q = self.FFN1_laynorm(q + 0.5*query)
        q = self.Wq(q)
        # Key
        k = self.Wk(key)
        #value
        v = self.Wv(value)
        att_out = dot_attention_function( q=q, k=k, v=v,
                                   activation=self.att_activate)
        att_out = self.att_laynorm(att_out + 0.5*q)
        Out = self.FFN2(att_out)
        Out = self.FFN2_laynorm(Out + 0.5*att_out)
        return Out 

class MyTopModel(nn.Module):
    def __init__(self, targets_num, class_num, hidden_dim, att_activate="tanh", use_one_transformers=False) -> None:
        super().__init__()
        self.targets_num = targets_num
        self.class_num = class_num
        self.use_one_transformers = use_one_transformers
        self.macaron_att = MacaronAttention(hidden_dim, att_activate)
        if use_one_transformers:
            self.macaron_att_1 = MacaronAttention(hidden_dim, att_activate)
        self.targets_emb = nn.Parameter(torch.normal(0.0, 1e-2,(targets_num, hidden_dim)), requires_grad=True)

    def forward(self, hidden_states):
        sentiment_aspect_f = self.macaron_att(query = self.targets_emb, key = hidden_states, value = hidden_states)
        
        if self.use_one_transformers:
            target_seq_hidden = self.macaron_att_1(query = hidden_states, key = self.targets_emb, value = self.targets_emb)
        else:
            target_seq_hidden = self.macaron_att(query = hidden_states, key = self.targets_emb, value = self.targets_emb)
        target_seq_hidden = target_seq_hidden[:,0,:]
        # target_att_prop = (target_seq_hidden*self.targets_emb).sum(dim=-1)
        target_att_prop = torch.matmul(target_seq_hidden, self.targets_emb.transpose(1,0))
        targets_normed = torch.sigmoid(target_att_prop)
        return sentiment_aspect_f, targets_normed