import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, dataset, parameter):
        super(Embedding, self).__init__()
        self.device = parameter['device']
        self.ent2id = dataset['ent2id']
        self.es = parameter['embed_dim']

        num_ent = len(self.ent2id)
        self.embedding = nn.Embedding(num_ent, self.es)

        if parameter['data_form'] == 'Pre-Train':
            self.ent2emb = dataset['ent2emb']
            self.embedding.weight.data.copy_(torch.from_numpy(self.ent2emb))
        elif parameter['data_form'] in ['In-Train', 'Discard']:
            nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, triples):
        idx = [[[self.ent2id[t[0]], self.ent2id[t[2]]] for t in batch] for batch in triples]
        idx = torch.LongTensor(idx).to(self.device)
        return self.embedding(idx)



