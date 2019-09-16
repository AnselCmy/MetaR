from collections import defaultdict
import numpy as np
import json
import argparse

args = argparse.ArgumentParser()
args.add_argument("-path", "--dataset_path", default="./NELL", type=str)  # ./Wiki
args.add_argument("-data", "--dataset_name", default="NELL-One", type=str)  # Wiki-One
params = args.parse_args()

dire = params.dataset_path
data = params.dataset_name

path = {
    'train_tasks': '/train_tasks.json',
    'test_tasks': '/test_tasks.json',
    'dev_tasks': '/dev_tasks.json',
    'rel2candidates': '/rel2candidates.json',
    'e1rel_e2': '/e1rel_e2.json',
    'path_graph': '/path_graph',
    'ent2emb': '/entity2vec.TransE'
}

print('Start')
print('Process {} in {}'.format(data, dire))

print("Loading jsons ... ...")
train_tasks = json.load(open(dire+path['train_tasks']))
test_tasks = json.load(open(dire+path['test_tasks']))
dev_tasks = json.load(open(dire+path['dev_tasks']))
e1rel_e2 = json.load(open(dire+path['e1rel_e2']))
path_graph_lines = open(dire+path['path_graph']).readlines()
rel2candidates = json.load(open(dire+path['rel2candidates']))
ent2emb = np.loadtxt(dire+path['ent2emb'], dtype=np.float32)

# convert entity2vec to .npy
np.save('ent2vec.npy', ent2emb)

entity = set()
path_graph = []
for line in path_graph_lines:
    triple = line.strip().split()
    entity.add(triple[0])
    entity.add(triple[2])
    path_graph.append(triple)
json.dump(path_graph, open(dire+'/path_graph.json', 'w'))

# train_tasks_in_train
print("Writing train_tasks_in_train.json ... ...")
path_graph_tasks = defaultdict(list)
for p in path_graph:
    path_graph_tasks[p[1]].append(p)
train_tasks_in_train = {**train_tasks, **path_graph_tasks}
json.dump(train_tasks_in_train, open(dire+'/train_tasks_in_train.json', 'w'))

# rel2candidates_in_train
if data == 'NELL-One':
    print("Writing rel2candidates_in_train.json ... ...")
    entity_dict = defaultdict(list)
    for ent in entity:
        s = ent.split(':')
        if len(s) != 3:
            entity_dict['num'].append(ent)
        else:
            entity_dict[s[1]].append(ent)

    rel2candidates_in_train = defaultdict(list)

    for rel, task in path_graph_tasks.items():
        types = []
        cands = []
        for i in task:
            e1, r, e2 = i
            s = e2.split(':')
            if len(s) != 3:
                types.append('num')
            else:
                types.append(s[1])
        types = set(types)
        for t in types:
            cands.extend(entity_dict[t])
        cands = list(set(cands))
        rel2candidates_in_train[rel] = cands

    rel2candidates_in_train = {**rel2candidates, **rel2candidates_in_train}
else:
    print("Writing rel2candidates_in_train.json ... ...")
    rel2candidates_in_train = defaultdict(list)
    for k, v in path_graph_tasks.items():
        cands = []
        for tri in v:
            cands.append(tri[2])
        cands = list(set(cands))
        rel2candidates_in_train[k] = cands
    rel2candidates_in_train = {**rel2candidates, **rel2candidates_in_train}

    for rel, cands in rel2candidates_in_train.items():
        if len(cands) == 1:
            one_cand = cands[0]
            for k, v in train_tasks_in_train.items():
                for tri in v:
                    h, r, t = tri
                    if t == one_cand:
                        cands.extend(rel2candidates_in_train[r])
                        break
                if len(cands) > 1:
                    break
            rel2candidates_in_train[rel] = list(set(cands))

json.dump(rel2candidates_in_train, open(dire + '/rel2candidates_in_train.json', 'w'))


# e1rel_e2_in_train
print("Writing e1rel_e2_in_train.json ... ...")
e1rel_e2_in_train = defaultdict(list)
for k, v in path_graph_tasks.items():
    for triple in v:
        e1, r, e2 = triple
        e1rel_e2_in_train[e1+r].append(e2)

e1rel_e2_in_train = {**e1rel_e2, **e1rel_e2_in_train}
json.dump(e1rel_e2_in_train, open(dire+'/e1rel_e2_in_train.json', 'w'))

print('End')


