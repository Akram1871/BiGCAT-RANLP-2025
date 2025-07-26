from collections import defaultdict

import torch
import torch.nn.functional as F
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.dataset_readers.dataset_utils.span_utils import \
    bio_tags_to_spans
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
import pandas as pd

import networkx as nx
labels = ['PER','LOC','ORG']
# labels = ['PER','LOC','ORG','MISC']

def construct_ov_graph(max_N, batch_size, max_seg):

    spans_idx = []
    for i in range(max_N):
        spans_idx.extend([(i, i + j) for j in range(max_seg)])

    N = len(spans_idx)
    G = nx.interval_graph(spans_idx)
    # neg edges
    neg = - nx.adjacency_matrix(G, spans_idx).todense()
    graph = torch.from_numpy(neg + np.eye(N, N)).long()
    graph = graph.view(1, N, N).repeat(batch_size, 1, 1)
    return graph




model_name = 'philschmid/distilroberta-base-ner-conll2003'

map_lab = dict((j, i) for i, j in enumerate(labels, start=1))

max_span_width = 1

tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(tokens):

    # if tags is None:
    tags = ['O' for _ in range(len(tokens))]

    def span_to_label(tags):
        """Convert tags to spans

        Args:
            tags (list[str]): BIO tags

        Returns:
            defaultdict: Mapping span idx to labels
        """

        dict_tag = defaultdict(int)

        spans_with_labels = bio_tags_to_spans(tags)

        for label, span in spans_with_labels:
            dict_tag[span] = map_lab[label]

        return dict_tag

    tokenized = [tokenizer.encode(
        t, add_special_tokens=False) for t in tokens]  # tokenization

    tokenized_2 = []
    tags_2 = []
    subword_lenghts = []
    i=0
    for tok, tg in zip(tokenized, tags):
        subword_lenghts.append(i)
        i += len(tok)
        if i < 512:
            tokenized_2.append(tok)
            tags_2.append(tg)

    # # compute span from BIO
    dict_map = span_to_label(tags_2)  # dict containing 'span'->'label'

    # all possible spans
    span_ids = []
    for i in range(len(tokenized_2)):
        span_ids.extend([(i, i + j) for j in range(max_span_width)])

    # span lengths
    span_lengths = []
    for idxs in span_ids:
        sid, eid = idxs
        slen = eid - sid
        span_lengths.append(slen)

    # sword boundary => span boundary
    mapping = dict(zip(range(len(subword_lenghts)), subword_lenghts))

    subword_boundaries = []
    for idxs in span_ids:
        try:
            subword_boundaries.append((mapping[idxs[0]], mapping[idxs[1]]))
        except:
            subword_boundaries.append((0, 0))

    # span labels
    span_labels = torch.LongTensor(
        [dict_map[i] for i in span_ids]
    )

    original_spans = torch.LongTensor(span_ids)  # [num_spans, 2]

    valid_span_mask = original_spans[:, 1] > len(tokenized_2) - 1

    span_labels = span_labels.masked_fill(valid_span_mask, -1)

    input_ids, span_ids, span_lengths = map(torch.LongTensor, [
                                            [i for k in tokenized for i in k], subword_boundaries, span_lengths])

    return {'input_ids': input_ids, 'span_ids': span_ids, 'span_lengths': span_lengths,'span_labels':span_labels, 'original_spans': original_spans}

def collate_fn( batch_list):

    # preprocess batch
    batch = [preprocess(tokens) for tokens in batch_list]
    # print(batch)

    # Char batch
    input_ids = pad_sequence([b['input_ids'] for b in batch], batch_first=True,
                              padding_value=tokenizer.pad_token_id)  # [B, W, C]

    # span mask
    span_ids = pad_sequence(
        [b['span_ids'] for b in batch], batch_first=True, padding_value=-1)
    # print('span_ids shape',span_ids.shape)
    # print('span_ids before',span_ids)

    span_ids = span_ids.masked_fill(span_ids == -1, 0)
    # print('span_ids after',span_ids)

    original_spans = pad_sequence(
        [b['original_spans'] for b in batch], batch_first=True, padding_value=0
    )

    # attention_mask
    attention_mask = (input_ids != tokenizer.pad_token_id).float()

    # span label
    span_labels = pad_sequence(
        [b['span_labels'] for b in batch], batch_first=True, padding_value=-1
    )

    span_lengths = pad_sequence(
        [b['span_lengths'] for b in batch], batch_first=True, padding_value=0
    )



    max_N = max([len(tokens) for tokens in batch_list])
    batch_size = len(batch_list)

    graph = construct_ov_graph(max_N, batch_size, max_span_width)
    span_mask = span_labels != -1

    # print('graph shape',graph.shape)
    # print('span_mask shape',span_mask.shape)

    graph = graph * span_mask.unsqueeze(-1)

    return {'input_ids': input_ids, 'span_ids': span_ids, 'attention_mask': attention_mask,'span_labels':span_labels, 'span_mask': span_mask, 'span_lengths': span_lengths, 'graph': graph, 'original_spans': original_spans}

def validate(true, pred, mask):
    """validate true and pred
    Args:
        true : [Batch_size, num_spans]
        pred : [Batch_size, num_spans]
        mask : [Batch_size, num_spans]

    Returns:
        masked true and pred
    """

    true = true.view(-1)
    pred = pred.view(-1)
    mask = mask.view(-1)

    valid = (true != 0) + (pred != 0)
    valid = valid * mask

    true = true[valid].tolist()
    pred = pred[valid].tolist()

    return true, pred

def extract_spans(predictions, span_mask, span_idx):
    """Extract spans + labels

    Args:
        predictions (torch.Tensor): predictions [B, num_spans]
        span_mask (torch.Tensor): mask per batch [B, num_spans]
        span_idx (torch.Tensor): spans [B, num_spans]

    Returns:
        [type]: [description]
    """

    lengths = span_mask.sum(-1)  # [B,]

    all_pred = []

    for pred, l, span in zip(predictions, lengths, span_idx):

        pred = pred[:l]

        idx = torch.where(pred > 0)[0]

        # number if non-O spans
        if idx.nelement() == 0:
            all_pred.append([])
            continue

        idx = idx.tolist()

        spans = []

        for i in idx:
            # start, end
            spi = span[i].tolist()

            # append label
            spi.append(pred[i].item())

            # to tuple
            spans.append(tuple(spi))

        all_pred.append(spans)

    return all_pred

def remove_padded_predictions(span_mask, predictions):
    """
    Remove padded predictions based on the given span mask.

    Args:
    - span_mask (torch.Tensor): Boolean tensor indicating the span positions.
    - predictions (torch.Tensor): Tensor containing the predicted values.

    Returns:
    - relevant_predictions (torch.Tensor): Filtered predictions without padded values.
    """
    relevant_predictions = []

    # Iterate through each row of the span mask and corresponding predictions
    for i in range(span_mask.shape[0]):
        # Get the indices of True values in the span mask
        true_indices = torch.nonzero(span_mask[i]).squeeze()

        # Extract the relevant predictions using the true indices
        row_predictions = predictions[i, true_indices].tolist()
        if not isinstance(row_predictions, list):
            row_predictions = [row_predictions]

        relevant_predictions.append(row_predictions)

    return relevant_predictions

def predict_ner(input_text, model):
  batch_list = np.array(input_text)

  # Convert the input to a tensor and move it to the appropriate device
  input_tensor =collate_fn(batch_list)
  # print(input_tensor)

  # Run inference
  with torch.no_grad():
      output = model(input_tensor)['logits'].argmax(-1)
      print(output.shape)
      true = input_tensor['span_labels']
      mask = input_tensor['span_mask']

      spans = extract_spans(output, mask, input_tensor['original_spans'])
      output = remove_padded_predictions(mask,output)
  return output

def map_to_entity_labels(relevant_predictions):
    """
    Map relevant predictions to entity labels using BIO format.

    Args:
    - relevant_predictions (list of lists): Filtered predictions without padded values.

    Returns:
    - entity_labels (list of lists): Entity labels in BIO format.
    """
    entity_type_mapping = {1: 'PER', 2: 'LOC', 3: 'ORG'}
    entity_labels = []

    for row_predictions in relevant_predictions:
        if not isinstance(row_predictions, list):
            row_predictions = [row_predictions]

        bio_labels = ['O'] * len(row_predictions)

        for i, predictionr in enumerate(row_predictions):
            if predictionr in entity_type_mapping:
                entity_type = entity_type_mapping[predictionr]

                # Assign B-<entity_type> for the first token of the entity
                if i == 0 or (i > 0 and row_predictions[i - 1] != predictionr):
                    bio_labels[i] = f'{entity_type}'
                else:
                    bio_labels[i] = f'{entity_type}'

        entity_labels.append(bio_labels)

    return entity_labels

def display_entities_with_tags(tokens, predictions):
    """
    Displays tokens along with their predicted entity tags.
    """
    for i, token_list in enumerate(tokens):
        print(f"Sentence {i + 1}:")
        for j, token in enumerate(token_list):
            if isinstance(token, list):
              for k, sub_token in enumerate(token):
                print(f"  Token: {sub_token}, Tag: {predictions[i][j][k]}")

            else:
                print(f"  Token: {token}, Tag: {predictions[i][j]}")
        print("---")
