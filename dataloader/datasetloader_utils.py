#!/usr/bin/env python3

import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class ExampleStruct:
    def __init__(self, **arg):
        self.__dict__.update(arg)

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def __check_valid_seqlen(feautures, max_seq_length, max_context_length):
    input_ids, attention_mask, token_type_ids, context_ids = feautures
    return len(input_ids) == max_seq_length and \
            len(attention_mask) == max_seq_length and \
            len(token_type_ids) == max_seq_length and \
            len(context_ids) == max_context_length

def convert_example_to_feature(example, context_id_map, max_seq_length,
                                 tokenizer, max_context_length,
                                 context_standalone, is_roberta):
    sep_token = getattr(tokenizer, "sep_token")
    cls_token = getattr(tokenizer, "cls_token")

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b and not context_standalone:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length. Account for [CLS], [SEP], [SEP] with "-3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length-2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0 0 0 0 0 0 0 0 1 1 1 1 1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0 0 0 0 0 0 0
    #
    # Where "type_ids" are used to indicate whether this is the first sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens , token_type_ids = [], []
    tokens.append(cls_token)
    token_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        token_type_ids.append(0)
    
    tokens.append(sep_token)
    token_type_ids.append(0)

    if tokens_b and not context_standalone:
        for token in tokens_b:
            tokens.append(token)
            if is_roberta:
                token_type_ids.append(0)
            else:
                token_type_ids.append(1)
        tokens.append(sep_token)
        if is_roberta:
            token_type_ids.append(0)
        else:
            token_type_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    context_ids = []
    if example.text_b:
        context_ids = [context_id_map[example.text_b.lower()]]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    attention_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        attention_mask.append(0)
        token_type_ids.append(0)

    while len(context_ids) < max_context_length:
        context_ids.append(0)

    assert __check_valid_seqlen((input_ids, attention_mask, token_type_ids, context_ids), 
                                max_seq_length, max_context_length)
    return input_ids, attention_mask, token_type_ids, context_ids
