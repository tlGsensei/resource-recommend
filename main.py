import os
import dgl
import random
import numpy as np
import setproctitle
from time import time
from prettytable import PrettyTable
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.parser import parse_args
from utils.load_data import load_data
from modules.model import Recommender
from utils.evaluate import evaluate
from utils.helper import early_stopping, create_loss
from utils.dataset import *

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


if __name__ == '__main__':

    """fix the random seed"""
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    """read args"""
    global args, device
    args = parse_args()

    """set process name"""
    setproctitle.setproctitle("cczhao's graduation-" + str(args.exp_name))

    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
    Ks = eval(args.Ks)
    sim_decay = args.sim_regularity

    """build dataset"""
    train_dataset, test_dataset, n_params, graph, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    
    n_users, n_items, n_entities, n_relations, n_nodes = n_params['n_users'], \
                                                         n_params['n_items'], \
                                                         n_params['n_entities'], \
                                                         n_params['n_relations'], \
                                                         n_params['n_nodes']

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    """define model"""
    model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    writer = SummaryWriter('./training_log/tensorboard')

    cur_best_pre_0 = 0
    best_res = None
    best_model = None
    stopping_step = 0
    should_stop = False

    print("start training ...")
    for epoch in range(args.epoch):
        mean_loss, cor_loss = 0., 0.
        train_s_t = time()
        model.train()
        for batch in train_loader:
            batch_g, labels = prepare_batch(batch, device)
            logits, cor, self_supervised_loss = model(batch_g)
            batch_loss, batch_cor = create_loss(logits, labels, cor, self_supervised_loss, sim_decay)
            batch_loss = batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            mean_loss += batch_loss.item() / args.batch_size
            cor_loss += batch_cor.item() / args.batch_size

        train_e_t = time()

        writer.add_scalar('Loss', round(mean_loss, 4), epoch)

        # if epoch % 10 == 0:
        """testing"""
        test_s_t = time()
        ret = evaluate(model, test_loader, device, Ks=Ks)
        test_e_t = time()

        
        # HRs = [round(ret['HR@' + str(k)], 5) for k in Ks]
        # MRRs = [round(ret['MRR@' + str(k)], 5) for k in Ks]
        # NDCGs = [round(ret['MRR@' + str(k)], 5) for k in Ks]
        HRs = [round(ret['HR@' + str(10)], 5), round(ret['HR@' + str(20)], 5)]
        MRRs = [round(ret['MRR@' + str(10)], 5), round(ret['MRR@' + str(20)], 5)]
        NDCGs = [round(ret['MRR@' + str(10)], 5), round(ret['MRR@' + str(20)], 5)]
        test_res = PrettyTable()
        # 将用户最后交互的学习资源作为测试集，该数据集划分方式下，Recall 和 HR 指标值相同
        test_res.field_names = ["Epoch", "training time", "testing time", "Loss", "Recall", "MRR", "NDCG"]
        test_res.add_row(
            [epoch, round(train_e_t - train_s_t, 3), round(test_e_t - test_s_t, 3), round(mean_loss, 4), \
                HRs, MRRs, NDCGs]
        )
        print(test_res)

        # for i, k in enumerate(Ks):
        #     writer.add_scalar('HR@' + str(k), round(HRs[i], 4), epoch)
        #     writer.add_scalar('MRR@' + str(k), round(MRRs[i], 4), epoch)
        #     writer.add_scalar('NDCG@' + str(k), round(NDCGs[i], 4), epoch)

        if NDCGs[1] > cur_best_pre_0:
            best_res = test_res

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        cur_best_pre_0, stopping_step, should_stop = early_stopping(NDCGs[1],
                                                                    cur_best_pre_0,
                                                                    stopping_step,
                                                                    expected_order='acc',
                                                                    flag_step=10)
        if should_stop:
            break

        """save weight"""
        if NDCGs[1] == cur_best_pre_0 and args.save:
            best_model = model

        # else:
        #     print('using time %.4f, training loss at epoch %d: %.4f, cor: %.6f' % (train_e_t - train_s_t, epoch, mean_loss, cor_loss))
    if args.save:
        torch.save(best_model.cpu().state_dict(), args.out_dir + 'model_' + args.dataset + '_' + str(args.exp_name) + '.ckpt')
    print('early stopping at %d, best result:' % (epoch))
    print(best_res)