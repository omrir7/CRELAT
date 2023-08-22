# import re
# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
# import Prepare_Text
# import pickle



def lower_known_names_dct(known_names):
    '''
    The purpose of this function is to add the key into the list of values
    as a lowered key
    '''
    if type(known_names) == list:
        known_names = {k: [k] for k in known_names}

    known_names_full = known_names.copy()
    for k, v in known_names_full.items():
        if k.lower not in v:
            v.append(k.lower())
    return known_names_full




def generate_idx_dict(text, found_names_full):
    '''
    create an idx_dct where the keys are the idx of the names in
    the text and the values are the names
    '''
    res = dict()
    for i, word in enumerate(text):
        for k, v in found_names_full.items():
            if word not in v:
                continue
            res[i] = k
    return res




def find_interactions(idx_dct, N):
    res = dict()
    names = list(idx_dct.keys())

    for i, na in enumerate(names):
        # given an index, get the sublist of all indicies greater than the current index
        if i < len(names) - 1:
            kl = names[i + 1:]
        else:
            kl = []

        # for each idx greater than the current, check if its found in the range of N
        for k in kl:
            if k - na < N:
                # get names found in current position (na) and index greater than current but in rnage N (k)
                n1 = idx_dct[na]
                n2 = idx_dct[k]

                key = tuple(sorted([n1, n2]))
                if n1!=n2:
                    if not(key in res.keys()):
                        res[key]=0
                    res[key]+=1
    return res




def CoOcCount(entities, book_text,window_size):
    known_names = dict()
    for ent in entities:
        known_names[ent[0]] = ent[1:]
    text = book_text
    found_names = known_names
    found_names_full = lower_known_names_dct(found_names)
    idx_dct = generate_idx_dict(text, found_names_full)

    interactions_dct = find_interactions(idx_dct, window_size)
    interactions_lst = list([(*k, v) for (k, v) in interactions_dct.items()])
    interactions_lst = [list(i) for i in interactions_lst]
    return interactions_lst