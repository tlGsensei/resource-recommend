import itertools
import dgl
import torch
import numpy as np


class ArgumentDataset:
    def __init__(self, data, vv_list, video_course_dict, is_argument=False, sort_by_length=True):
        self.data = data
        self.vv_list = vv_list
        self.video_course_dict = video_course_dict
        self.is_argue = is_argument
        if is_argument:
            index = self._create_index(data)
            if sort_by_length:
                ind = np.argsort(index[:, 1])[::-1]
                index = index[ind]
            self.index = index
        

    def __getitem__(self, index):
        if self.is_argue:
            seq_index, label_index = self.index[index]
            user_id = self.data[seq_index][0]
            seq = self.data[seq_index][1:label_index]
            label = self.data[seq_index][label_index]
        else:
            user_id = self.data[index][0]
            seq = self.data[index][1:-1]
            label = self.data[index][-1]
        
        # item-item pair in course
        item_set = set(seq)
        ii_list = [x for x in self.vv_list if x[0] in item_set and x[1] in item_set]

        if len(seq) < 4: # cold start user when evaluation
            sub_seq, ssub_seq = seq, seq
            sub_ii_list, ssub_ii_list = ii_list, ii_list
        else:
            # behavior in the last course
            if seq[-1] in self.video_course_dict.keys():
                target_cid = self.video_course_dict[seq[-1]]
                for i in range(len(seq)):
                    if seq[i] in self.video_course_dict.keys() and self.video_course_dict[seq[i]] == target_cid:
                        break
            else:
                i = len(seq) - 1
            ssub_seq = seq[i:] if i != len(seq) - 1 else [seq[-1]]
            ssub_item_set = set(ssub_seq)
            ssub_ii_list = [x for x in self.vv_list if x[0] in ssub_item_set and x[1] in ssub_item_set]

            # behavior in the last two course
            if i != 0:
                if seq[i - 1]  in self.video_course_dict.keys():
                    target_cid = self.video_course_dict[seq[i - 1]]
                    for i in range(len(seq)):
                        if seq[i] in self.video_course_dict.keys() and self.video_course_dict[seq[i]] == target_cid:
                            break
                else:
                    i -= 1
                sub_seq = seq[i:]
            else:
                sub_seq = ssub_seq
            sub_item_set = set(sub_seq)
            sub_ii_list = [x for x in self.vv_list if x[0] in sub_item_set and x[1] in sub_item_set]

        return user_id, seq, sub_seq, ssub_seq, ii_list, sub_ii_list, ssub_ii_list, label

    def __len__(self):
        return len(self.index) if self.is_argue else len(self.data)

    def _create_index(self, data):
        """
        sequence argument

        for example:
            origin data:
            data = [[u_id, i_id1, i_id2, i_id3, i_id4],
                    [...],
                    ...,
                   ]
            data[0] after argument:
                seq_index = [0, 0, 0]
                label_index = [2, 3, 4]
        """
        lens = np.fromiter(map(len, data), dtype=np.long) - 1 # exclude user_id
        seq_index = np.repeat(np.arange(len(data)), lens - 1)
        label_index = map(lambda l: range(2, l + 1), lens)
        label_index = itertools.chain.from_iterable(label_index)
        label_index = np.fromiter(label_index, dtype=np.long)
        index = np.column_stack((seq_index, label_index))
        return index


def label_last(g, last_nid):
    is_last = torch.zeros(g.number_of_nodes(), dtype=torch.int32)
    is_last[last_nid] = 1
    g.ndata['last'] = is_last
    return g


def seq_to_graph(seq_struct):
    seq, ii_pair = seq_struct
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


def collate_fn(samples):
    user_ids, seqs, sub_seqs, ssub_seqs, ii_list, sub_ii_list, ssub_ii_list, labels = zip(*samples)
    user_ids = torch.LongTensor(user_ids)

    graphs = list(map(seq_to_graph, zip(seqs, ii_list)))
    bg = dgl.batch(graphs)
    sub_graphs = list(map(seq_to_graph, zip(sub_seqs, sub_ii_list)))
    sub_bg = dgl.batch(sub_graphs)
    ssub_graphs = list(map(seq_to_graph, zip(ssub_seqs, ssub_ii_list)))
    ssub_bg = dgl.batch((ssub_graphs))

    lens = np.fromiter(map(len, seqs), dtype=np.long)
    seqs_tensor = list(map(lambda x : torch.LongTensor(x), seqs))
    seq_pad = torch.nn.utils.rnn.pad_sequence(seqs_tensor, batch_first=True, padding_value=0)

    sub_lens = np.fromiter(map(len, sub_seqs), dtype=np.long)
    sub_seqs_tensor = list(map(lambda x : torch.LongTensor(x), sub_seqs))
    sub_seq_pad = torch.nn.utils.rnn.pad_sequence(sub_seqs_tensor, batch_first=True, padding_value=0)

    ssub_lens = np.fromiter(map(len, ssub_seqs), dtype=np.long)
    ssub_seqs_tensor = list(map(lambda x : torch.LongTensor(x), ssub_seqs))
    ssub_seq_pad = torch.nn.utils.rnn.pad_sequence(ssub_seqs_tensor, batch_first=True, padding_value=0)

    labels = torch.LongTensor(labels)
    return (user_ids, bg, sub_bg, ssub_bg,
            seq_pad, sub_seq_pad, ssub_seq_pad,
            lens, sub_lens, ssub_lens,  labels)


def prepare_batch(batch, device):
    user_ids, graphs, sub_graphs, ssub_graphs, \
    seqs, sub_seqs, ssub_seqs, \
    lens, sub_lens, ssub_lens, labels = batch

    user_ids_g = user_ids.to(device)
    graphs_g = graphs.to(device)
    sub_graphs_g = sub_graphs.to(device)
    ssub_graphs_g = ssub_graphs.to(device)

    seqs_g = seqs.to(device)
    sub_seqs_g = sub_seqs.to(device)
    ssub_seqs_g = ssub_seqs.to(device)
    labels_g = labels.to(device)
    return (user_ids_g, graphs_g, sub_graphs_g, ssub_graphs_g,
            seqs_g, sub_seqs_g, ssub_seqs_g,
            lens, sub_lens, ssub_lens), labels_g