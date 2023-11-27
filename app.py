import json
import torch
import pickle as pkl
from flask import Flask, Response
from flask import request
import numpy as np
import dgl

from utils.parser_http import parse_args
from utils.load_data import load_data

from modules.model_http import Recommender


app = Flask(__name__)

global uid_iid_dict, user_behavior_dict, iid_dict, video_course_dict, vv_list

base_data_path = './data/mooper/'


def init():
    global uid_iid_dict, user_behavior_dict, iid_dict, video_course_dict, vv_list
    
    uid_iid_dict = get_user_dict()
    user_behavior_dict = get_user_behavior_dict()
    iid_dict = get_item_dict()
    video_course_dict = get_video_course_dict()
    vv_list = get_vv_list()

    """read args"""
    global args, device
    args = parse_args()

    sim_decay = args.sim_regularity

    """build dataset"""
    train_dataset, test_dataset, n_params, graph, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    
    n_users, n_items, n_entities, n_relations, n_nodes = n_params['n_users'], \
                                                         n_params['n_items'], \
                                                         n_params['n_entities'], \
                                                         n_params['n_relations'], \
                                                         n_params['n_nodes']
    global model
    model = Recommender(n_params, args, graph, mean_mat_list[0])
    model.load_state_dict(torch.load('./weights/model_mooper_http_cpu.ckpt'))
    model.eval()


def get_user_dict():
    uid_iid_dict = {}
    lines = open(base_data_path + 'user_list.txt', 'r').readlines()
    for i, l in enumerate(lines):
        if i == 0:
            continue
        user_id = [int(i) for i in l.strip().split(" ")]
        uid, uiid = user_id[0], user_id[1]
        uid_iid_dict[uid] = uiid
    return uid_iid_dict


def get_user_behavior_dict():
    user_behavior_dict = {}
    lines = open(base_data_path + 'train.txt', 'r').readlines()
    for l in lines:
        inters = [int(i) for i in l.strip().split(" ")]
        uiid, behavior = inters[0], inters[1:]
        user_behavior_dict[uiid] = behavior
    return user_behavior_dict


def get_item_dict():
    iid_dict = {}
    lines = open(base_data_path + 'video_list.txt', 'r').readlines()
    for i, l in enumerate(lines):
        if i == 0:
            continue
        item_id = [int(i) for i in l.strip().split(" ")]
        id, iid = item_id[0], item_id[1]
        iid_dict[iid] = id
    return iid_dict


def get_video_course_dict():
    with open(base_data_path + 'video_course_dict.pkl', 'rb') as f:
        video_course_dict = pkl.load(f)
        return video_course_dict


def get_vv_list():
    vv_list = []
    with open(base_data_path + 'video_video_list.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n').split(' ')
            vv_list.append([int(x) for x in line])

    return vv_list



def label_last(g, last_nid):
    is_last = torch.zeros(g.number_of_nodes(), dtype=torch.int32)
    is_last[last_nid] = 1
    g.ndata['last'] = is_last
    return g


def seq_to_graph(seq, ii_pair):
    items = np.unique(seq)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    num_nodes = len(items)

    if len(seq) > 1:
        seq_nid = [iid2nid[iid] for iid in seq]
        ii_pair_nid = [[iid2nid[x[0]], iid2nid[x[1]]] for x in ii_pair]
        src = seq_nid[:-1]
        dst = seq_nid[1:]
        src.extend([x[0] for x in ii_pair_nid])
        dst.extend([x[1] for x in ii_pair_nid])
    else:
        src = torch.LongTensor([])
        dst = torch.LongTensor([])
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g.ndata['video_id'] = torch.tensor(items, dtype=torch.long)
    label_last(g, iid2nid[seq[-1]])
    return g


def get_input_data(user_id, seq):
    global vv_list, video_course_dict
    # item-item pair in course
    item_set = set(seq)
    ii_list = [x for x in vv_list if x[0] in item_set and x[1] in item_set]

    if len(seq) < 4: # cold start user when evaluation
        sub_seq, ssub_seq = seq, seq
        sub_ii_list, ssub_ii_list = ii_list, ii_list
    else:
        # behavior in the last course
        if seq[-1] in video_course_dict.keys():
            target_cid = video_course_dict[seq[-1]]
            for i in range(len(seq)):
                if seq[i] in video_course_dict.keys() and video_course_dict[seq[i]] == target_cid:
                    break
        else:
            i = len(seq) - 1
        ssub_seq = seq[i:] if i != len(seq) - 1 else [seq[-1]]
        ssub_item_set = set(ssub_seq)
        ssub_ii_list = [x for x in vv_list if x[0] in ssub_item_set and x[1] in ssub_item_set]

        # behavior in the last two course
        if i != 0:
            if seq[i - 1]  in video_course_dict.keys():
                target_cid = video_course_dict[seq[i - 1]]
                for i in range(len(seq)):
                    if seq[i] in video_course_dict.keys() and video_course_dict[seq[i]] == target_cid:
                        break
            else:
                i -= 1
            sub_seq = seq[i:]
        else:
            sub_seq = ssub_seq
        sub_item_set = set(sub_seq)
        sub_ii_list = [x for x in vv_list if x[0] in sub_item_set and x[1] in sub_item_set]

    user_id = torch.LongTensor([user_id])

    graphs = seq_to_graph(seq, ii_list)
    # bg = dgl.batch(graphs)
    bg = graphs
    sub_graphs = seq_to_graph(sub_seq, sub_ii_list)
    # sub_bg = dgl.batch(sub_graphs)
    sub_bg = sub_graphs
    ssub_graphs = seq_to_graph(ssub_seq, ssub_ii_list)
    # ssub_bg = dgl.batch(ssub_graphs)
    ssub_bg = ssub_graphs

    lens = [len(seq)]
    seqs_tensor = torch.LongTensor([seq])
    # seq_pad = torch.nn.utils.rnn.pad_sequence(seqs_tensor, batch_first=True, padding_value=0)

    sub_lens = [len(sub_seq)]
    sub_seqs_tensor = torch.LongTensor([sub_seq])
    # sub_seq_pad = torch.nn.utils.rnn.pad_sequence(sub_seqs_tensor, batch_first=True, padding_value=0)

    ssub_lens = [len(ssub_seq)]
    ssub_seqs_tensor = torch.LongTensor([ssub_seq])
    # ssub_seq_pad = torch.nn.utils.rnn.pad_sequence(ssub_seqs_tensor, batch_first=True, padding_value=0)

    return  (user_id, graphs, sub_graphs, ssub_graphs, seq, sub_seq, ssub_seq,
            lens, sub_lens, ssub_lens)


def load_and_predict(user_id, topk=5):
    user_id, topk = int(user_id), int(topk)
    global uid_iid_dict, user_behavior_dict, iid_dict, video_course_dict, vv_list
    
    default_rec_result = ['2050', '2054', '2068', '2069', '4117', '2072', '2075', '2858', '2096', '2102']

    # 新用户, 走兜底策略
    if user_id not in uid_iid_dict.keys():
        return default_rec_result

    uid = uid_iid_dict[user_id]

    # 取用户历史行为序列
    if uid not in user_behavior_dict.keys():
        return default_rec_result
    else:
        seq = user_behavior_dict[uid]

    input_data = get_input_data(user_id, seq)

    global model
    item_list = model.predict(input_data)
    selected_items = item_list[:(min(topk, len(item_list)))]
    result = [iid_dict[x] for x in  selected_items]
    return result

""" 学习资源推荐 """
@app.route('/src_rec', methods=['POST'])
def source_rec():
    status = 200
    result = {}
    global uid_iid_dict, user_behavior_dict, video_course_dict, vv_list

    # 接收参数
    user_id = request.form.get('user_id', type=str, default='')
    topk = request.form.get('topk', type=str, default=5)      

    # 参数校验
    if user_id.strip() == '':
        status = 4001
        result = {
            "status": str('False'),
            "msg": str('参数错误: 缺少user_id')
        }
        return Response(json.dumps(result, ensure_ascii=False), mimetype='application/json')
    elif user_id.isnumeric() == False or int(user_id) < 0:
        result = {
            "status": str('False'),
            "error_msg": str('user_id不合法')
        }
        return Response(json.dumps(result, ensure_ascii=False), mimetype='application/json')
    elif topk.isnumeric() == False or int(topk) <= 0:
        result = {
            "status": str('False'),
            "error_msg": str('topk不合法')
        }
        return Response(json.dumps(result, ensure_ascii=False), mimetype='application/json')

    # 调用模型，得到推荐结果
    recommend_results = load_and_predict(user_id, topk)

    # 接口响应数据
    result = {
        "status": str(status),
        "user_id": str(user_id),
        "recommend_count": str(len(recommend_results)),
        "recommend_results": recommend_results,
    }
    return Response(json.dumps(result), mimetype='application/json')

if __name__ == '__main__':
    init()
    app.run(host='0.0.0.0', port=9000, debug=True, use_reloader=False)




