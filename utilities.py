import os
import re
import csv
import numpy as np
from itertools import groupby
from copy import deepcopy
import numpy as np


def numpy_kl_div(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def flatten(nested_elems):
    return [elem for elems in nested_elems for elem in elems]

def stratify(data, classes, ratios, one_hot=False):
    """Stratifying procedure. Borrowed from https://vict0rs.ch/2018/05/24/sample-multilabel-dataset/
    data is a list of lists: a list of labels, for each sample.
        Each sample's labels should be ints, if they are one-hot encoded, use one_hot=True
    
    classes is the list of classes each label can take
    ratios is a list, summing to 1, of how the dataset should be split
    """
    # one-hot decoding
    if one_hot:
        temp = [[] for _ in range(len(data))]
        indexes, values = np.where(np.array(data).astype(int) == 1)
        for k, v in zip(indexes, values):
            temp[k].append(v)
        data = temp

    # Organize data per label: for each label l, per_label_data[l] contains the list of samples
    # in data which have this label
    per_label_data = {c: set() for c in classes}
    for i, d in enumerate(data):
        for l in d:
            per_label_data[l].add(i)

    # number of samples
    size = len(data)

    # In order not to compute lengths each time, they are tracked here.
    subset_sizes = [r * size for r in ratios]
    target_subset_sizes = deepcopy(subset_sizes)
    per_label_subset_sizes = {
        c: [r * len(per_label_data[c]) for r in ratios]
        for c in classes
    }

    # For each subset we want, the set of sample-ids which should end up in it
    stratified_data_ids = [set() for _ in range(len(ratios))]

    # For each sample in the data set
    while size > 0:
        # Compute |Di|
        lengths = {
            l: len(label_data)
            for l, label_data in per_label_data.items()
        }
        try:
            # Find label of smallest |Di|
            label = min(
                {k: v for k, v in lengths.items() if v > 0}, key=lengths.get
            )
        except ValueError:
            # If the dictionary in `min` is empty we get a Value Error. 
            # This can happen if there are unlabeled samples.
            # In this case, `size` would be > 0 but only samples without label would remain.
            # "No label" could be a class in itself: it's up to you to format your data accordingly.
            break
        current_length = lengths[label]

        # For each sample with label `label`
        while per_label_data[label]:
            # Select such a sample
            current_id = per_label_data[label].pop()

            subset_sizes_for_label = per_label_subset_sizes[label]
            # Find argmax clj i.e. subset in greatest need of the current label
            largest_subsets = np.argwhere(
                subset_sizes_for_label == np.amax(subset_sizes_for_label)
            ).flatten()

            if len(largest_subsets) == 1:
                subset = largest_subsets[0]
            # If there is more than one such subset, find the one in greatest need
            # of any label
            else:
                largest_subsets = np.argwhere(
                    subset_sizes == np.amax(subset_sizes)
                ).flatten()
                if len(largest_subsets) == 1:
                    subset = largest_subsets[0]
                else:
                    # If there is more than one such subset, choose at random
                    subset = np.random.choice(largest_subsets)

            # Store the sample's id in the selected subset
            stratified_data_ids[subset].add(current_id)

            # There is one fewer sample to distribute
            size -= 1
            # The selected subset needs one fewer sample
            subset_sizes[subset] -= 1

            # In the selected subset, there is one more example for each label
            # the current sample has
            for l in data[current_id]:
                per_label_subset_sizes[l][subset] -= 1
            
            # Remove the sample from the dataset, meaning from all per_label dataset created
            for l, label_data in per_label_data.items():
                if current_id in label_data:
                    label_data.remove(current_id)

    # Create the stratified dataset as a list of subsets, each containing the orginal labels
    stratified_data_ids = [sorted(strat) for strat in stratified_data_ids]
    stratified_data = [
        [data[i] for i in strat] for strat in stratified_data_ids
    ]

    # Return both the stratified indexes, to be used to sample the `features` associated with your labels
    # And the stratified labels dataset
    return stratified_data_ids, stratified_data

def get_utterance_cmi(utterance_labels, languages={'mixed', 'lang1', 'lang2', 'fw'}):
    token_count = len(utterance_labels)
    label_counts = {'other': 0}

    utterance_CMI = 0.0
    for label in utterance_labels:
        if label in languages:
            label_counts[label] = label_counts.get(label, 0) + 1
        else:
            # u -> 'ne', 'other', 'ambiguous', 'unk', etc.
            label_counts['other'] = label_counts.get('other', 0) + 1

    lang_label_counts = [label_counts[key] for key in label_counts if key not in 'other']
    
    if lang_label_counts: # if has language labels
        max_lang_count = float(max(lang_label_counts))
    else:
        max_lang_count = 0.0

    if token_count > label_counts['other']:
        tmp = max_lang_count / float(token_count - label_counts['other']) # max{wi}/n-u
        cmi = (1 - tmp) * 100
    else:
        cmi = 0.0
    return cmi

def assert_cmi_implementation():
    """Test cases taken from the CMI paper: https://pdfs.semanticscholar.org/c82c/9ea0073129904738fbc051c06188c02f4f6b.pdf"""
    test1_langs = {'hi', 'en'}  # 'univ' and 'acro' are treated as 'other'
    test1_labels = 'hi hi univ hi hi hi univ en hi en en univ univ hi hi hi hi univ en acro acro acro acro acro acro univ hi en en hi acro acro acro univ en univ en hi hi hi hi hi hi univ en en hi hi hi hi univ hi en hi acro acro en en en en hi hi en en hi hi en en en univ en hi hi hi univ univ univ en hi hi en univ en en en hi hi acro univ hi hi acro acro hi hi en hi univ univ hi hi hi acro acro hi en en acro acro hi univ '.split()
    assert 39.19 == round(get_utterance_cmi(test1_labels, test1_langs), 2), 'Test #1 failed'
    
    test2_langs = {'hi', 'en'}
    test2_labels = 'hi hi hi univ hi hi univ'.split()
    assert 0 == get_utterance_cmi(test2_labels, test2_langs), 'Test #2 failed'
    
def is_code_mixed(utterance_labels, langs):
    lang_in_utterance = [lang for lang in langs if lang in utterance_labels]    
    return len(lang_in_utterance) >= 2 # at least two languages 

def get_dataset_cmi(dataset_labels, langs):
    all_sents_count = float(len(dataset_labels))
    all_tokens_count = sum([len(sents) for sents in dataset_labels])
    cm_sents_count = float(len([labels for labels in dataset_labels if is_code_mixed(labels, langs)]))
    
    CMI = 0.0
    for labels in dataset_labels:
        CMI += get_utterance_cmi(labels, langs)
        
    CMI_all = round(CMI / all_sents_count, 3)
    CMI_cm = round(float(CMI) / cm_sents_count, 3)
    
    stats = {'cmi_all': CMI_all, 
             'cmi_cm': CMI_cm, 
             'all_sents': all_sents_count, 
             'all_tokens': all_tokens_count,
             'cm_sents': cm_sents_count}
    
    return stats

def get_corpus_cmi(corpus, langs):
    return {split: get_dataset_cmi(corpus[split]['lid'], langs) for split in corpus}