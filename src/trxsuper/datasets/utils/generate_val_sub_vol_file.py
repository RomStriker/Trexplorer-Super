import os.path
import glob
import pickle
import random
from bigtree import levelordergroup_iter, find_name


random.seed(37)


def create_sub_vol_eval_dataset(annot_path, bifur_prob, end_prob, root_prob,
                                seq_len, num_samples_per_annot, seq_range_half):
    if seq_range_half:
        seq_len = seq_len // 2
    annot_id = int(os.path.basename(annot_path).split('.')[0])

    annot_samples = []
    with open(annot_path, 'rb') as f:
        annot = pickle.load(f)

    num_bifur_samples = int(num_samples_per_annot * bifur_prob)
    num_end_samples = int(num_samples_per_annot * end_prob)
    num_root_samples = int(num_samples_per_annot * root_prob)
    num_rest_samples = num_samples_per_annot - num_bifur_samples - num_end_samples - num_root_samples

    # Get bifucation and end points
    types = ['bifur', 'end', 'root', 'interm']
    types_to_point_ids = {'bifur': 'bifur_ids', 'end': 'endpts_ids', 'root': 'root_id', 'interm': 'interm_ids'}
    num_type_samples = [num_bifur_samples, num_end_samples, num_root_samples, num_rest_samples]
    for t_i, type in enumerate(types):
        if type == 'root':
            type_points = [[id] for id in annot[types_to_point_ids[type]]]
        else:
            type_points = annot[types_to_point_ids[type]]
        total_type_points = sum([len(tr) for tr in type_points])

        # Cumulative index
        cumulative_index = -1
        index_mapping = {}
        for tree_id, lst in enumerate(type_points):
            for index_within_list, value in enumerate(lst):
                cumulative_index += 1
                index_mapping[cumulative_index] = [tree_id, index_within_list]

        # Generate num_type_samples random indices for type_points
        if type == 'root':
            type_samples_idx = random.choices(range(total_type_points), k=num_type_samples[t_i])
        else:
            type_samples_idx = random.sample(range(total_type_points), num_type_samples[t_i])

        for idx in type_samples_idx:
            tree_id, index_within_list = index_mapping[idx]
            node_name = type_points[tree_id][index_within_list]
            node_point_id = node_name.split('-')[1]
            while node_point_id == '0' and type == 'end':
                idx += 1
                tree_id, index_within_list = index_mapping[idx]
                node_name = type_points[tree_id][index_within_list]
                node_point_id = node_name.split('-')[1]
            if type == 'bifur':
                # sample a random index between (node_point_id - seq_len//2) and node_point_id
                index = random.randint(max(0, int(node_point_id) - seq_len), int(node_point_id))
            elif type == 'end':
                # sample a random index between (node_point_id - seq_len//2) and node_point_id - 1
                index = random.randint(max(0, int(node_point_id) - seq_len), int(node_point_id) - 1)
            elif type == 'root':
                branch_id = node_name.split('-')[0]
                branch_idx = annot['branch_ids'][tree_id].index(branch_id)
                branch_length = annot['num_points'][tree_id][branch_idx]
                # sample a random index between node_point_id and (node_point_id + seq_len//2)
                index = random.randint(int(node_point_id), min(branch_length - 1, int(node_point_id) + seq_len))
            else:
                index = int(node_point_id)
            distance = abs(index - int(node_point_id))
            new_node_name = node_name.split('-')[0] + '-' + str(index)
            assert new_node_name in annot['all_ids'][tree_id], 'new_node_name not in all_ids'

            # structure of annot_samples: [node_name, tree_id, annot_id, point_type, sample_point_id]
            annot_samples.append({'node_id': new_node_name, 'tree_id': tree_id, 'sample_id': annot_id,
                                  'point_type': type, 'distance': distance})

    for idx, item in enumerate(annot_samples):
        item['index'] = idx

    return annot_samples


if __name__ == '__main__':
    bifur_prob = 0.5
    end_prob = 0.3
    root_prob = 0.2
    num_samples = 1280
    seq_len = 10
    seq_range_half = False

    dataset = 'syntrx'  # 'atm22', 'parse2022', 'syntrx'
    annot_dir = f'PATH/TO/DATASET/{dataset}/annots_val_sub_vol'
    out_dir = f'./data/{dataset}/'
    save_path = out_dir + 'annots_val_sub_vol.pickle'
    annot_paths = glob.glob(os.path.join(annot_dir, "*.pickle"))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    num_samples_per_annot = num_samples // len(annot_paths)
    samples_list = []
    for annot_path in annot_paths:
        samples_list += create_sub_vol_eval_dataset(annot_path, bifur_prob, end_prob, root_prob,
                                                    seq_len, num_samples_per_annot, seq_range_half)
    for sample in samples_list:
        print(sample)

    with open(save_path, 'wb') as f:
        pickle.dump(samples_list, f)