import os
import json
import random
import numpy as np
from tqdm import tqdm
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from utils import LoadDataset

DATA_PATH = '../data/new_data/'
SAVE_PATH = '../final_output/weight_blending8_1130'
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

def save_json(save_path, dic):
    with open(save_path, 'w') as f:
        json.dump(dic, f)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed=42)
dataset = LoadDataset(data_path=DATA_PATH, debug=False)

file_list = {
    '../final_output/LightECTNet/': 0.3,
    '../final_output/light_lxmert1009/': 0.2,
    '../final_output/inception_lxmert1001/': 0.1,
    '../final_output/gru_lxmert1001/': 0.1,

    '../final_output/enhancedrcnn1001/': 0.1,
    '../final_output/esim1011_5/': 0.1,

    '../final_output/lstm1001/': 0.05,
    '../final_output/lxmert1001/': 0.05,
}

test_emb = []
for path in tqdm(file_list, desc='Load Embeddings'):
    test_emb_slice = np.load(f'{path}/distill_test_pred.npy')
    test_emb.append(test_emb_slice)
res = dict()
for key in tqdm(range(test_emb[0].shape[0])):
    this_id = dataset.test_ids[key]
    for idx in range(len(test_emb)):
        weight = list(file_list.values())[idx]
        # print('weight: ', weight)
        if idx == 0:
            res[this_id] = test_emb[idx][key] * weight
        else:
            res[this_id] += test_emb[idx][key] * weight
    res[this_id] = list(res[this_id])

with open('result.json', 'w') as f:
    json.dump(res, f)
with ZipFile(f'{SAVE_PATH}/weight_blend8.zip', 'w', compression=ZIP_DEFLATED) as zip_file:
    zip_file.write('result.json')
