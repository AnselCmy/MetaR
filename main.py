from trainer import *
from params import *
from data_loader import *
import json

if __name__ == '__main__':
    params = get_params()

    print("---------Parameters---------")
    for k, v in params.items():
        print(k + ': ' + str(v))
    print("----------------------------")

    # control random seed
    if params['seed'] is not None:
        SEED = params['seed']
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        np.random.seed(SEED)
        random.seed(SEED)

    # select the dataset
    for k, v in data_dir.items():
        data_dir[k] = params['data_path']+v

    tail = ''
    if params['data_form'] == 'In-Train':
        tail = '_in_train'

    dataset = dict()
    print("loading train_tasks{} ... ...".format(tail))
    dataset['train_tasks'] = json.load(open(data_dir['train_tasks'+tail]))
    print("loading test_tasks ... ...")
    dataset['test_tasks'] = json.load(open(data_dir['test_tasks']))
    print("loading dev_tasks ... ...")
    dataset['dev_tasks'] = json.load(open(data_dir['dev_tasks']))
    print("loading rel2candidates{} ... ...".format(tail))
    dataset['rel2candidates'] = json.load(open(data_dir['rel2candidates'+tail]))
    print("loading e1rel_e2{} ... ...".format(tail))
    dataset['e1rel_e2'] = json.load(open(data_dir['e1rel_e2'+tail]))
    print("loading ent2id ... ...")
    dataset['ent2id'] = json.load(open(data_dir['ent2ids']))

    if params['data_form'] == 'Pre-Train':
        print('loading embedding ... ...')
        dataset['ent2emb'] = np.load(data_dir['ent2vec'])

    print("----------------------------")

    # data_loader
    train_data_loader = DataLoader(dataset, params, step='train')
    dev_data_loader = DataLoader(dataset, params, step='dev')
    test_data_loader = DataLoader(dataset, params, step='test')
    data_loaders = [train_data_loader, dev_data_loader, test_data_loader]

    # trainer
    trainer = Trainer(data_loaders, dataset, params)

    if params['step'] == 'train':
        trainer.train()
        print("test")
        print(params['prefix'])
        trainer.reload()
        trainer.eval(istest=True)
    elif params['step'] == 'test':
        print(params['prefix'])
        if params['eval_by_rel']:
            trainer.eval_by_relation(istest=True)
        else:
            trainer.eval(istest=True)
    elif params['step'] == 'dev':
        print(params['prefix'])
        if params['eval_by_rel']:
            trainer.eval_by_relation(istest=False)
        else:
            trainer.eval(istest=False)

