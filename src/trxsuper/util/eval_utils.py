import numpy as np
import torch
import src.trxsuper.util.misc as utils
from scipy.spatial import cKDTree
import networkx as nx
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from src.trxsuper.datasets.graph_utils import get_branches


def calculate_scores(gt_points, pred_points, gt_num_child=None, pred_num_child=None, threshold=1.5):
    gt_tree = cKDTree(gt_points)
    if len(pred_points):
        pred_tree = cKDTree(pred_points)
    else:
        return 0, 0, 0

    dis_gt2pred, dis_gt2pred_idxs = pred_tree.query(gt_points, k=1)
    dis_pred2gt, _ = gt_tree.query(pred_points, k=1)

    min_distances = {}
    for i in range(len(dis_gt2pred_idxs)):
        idx = dis_gt2pred_idxs[i]
        distance = dis_gt2pred[i]
        if idx not in min_distances or distance < min_distances[idx]:
            min_distances[idx] = distance

    filtered_dis_gt2pred_idxs = [idx for idx, _ in min_distances.items()]
    filtered_dis_gt2pred = [min_distances[idx] for idx in filtered_dis_gt2pred_idxs]
    filtered_dis_gt2pred = np.array(filtered_dis_gt2pred)

    # Compute Acc, Recall, F1
    # Bipartite (BP) version does not count duplicate predictions as true positives
    true_positives = [x for x in dis_gt2pred if x < threshold]
    filtered_true_positives = [x for x in filtered_dis_gt2pred if x < threshold]
    recall = len(true_positives) / len(dis_gt2pred)  # Num true positive / num all ground truths
    recall_nd = len(filtered_true_positives) / len(dis_gt2pred)  # Num true positive / num all ground truths
    acc = len([x for x in dis_pred2gt if x < threshold]) / len(dis_pred2gt)  # Num true positives including duplicate counts / num all predictions
    acc_nd = len(filtered_true_positives) / len(dis_pred2gt)  # Num true positives / num all predictions
    r_f = 0
    r_f_nd = 0
    if acc * recall:
        r_f = 2 * recall * acc / (acc + recall)
        r_f_nd = 2 * recall_nd * acc_nd / (acc_nd + recall_nd)

    # compute the distance and l1-distance of the num of children for predicted bifurcation points
    if gt_num_child is not None and pred_num_child is not None:
        recalled_points_gt_nchild = [gt_num_child[idx]
                                     for idx, dist in enumerate(dis_gt2pred) if dist < threshold]
        recalled_points_pred_nchild = [pred_num_child[dis_gt2pred_idxs[idx]]
                                       for idx, dist in enumerate(dis_gt2pred) if dist < threshold]
        # L1 distance
        if len(recalled_points_gt_nchild) and len(recalled_points_pred_nchild):
            arr1 = np.array(recalled_points_gt_nchild)
            arr2 = np.array(recalled_points_pred_nchild)
            nchild_dist = np.sum(arr1 - arr2) / len(recalled_points_gt_nchild)
            nchild_dist_l1 = np.sum(np.abs(arr1 - arr2)) / len(recalled_points_gt_nchild)
        else:
            nchild_dist, nchild_dist_l1 = np.nan, np.nan
    else:
        return acc, acc_nd, recall, recall_nd, r_f, r_f_nd

    return acc, acc_nd, recall, recall_nd, r_f, r_f_nd, nchild_dist_l1, nchild_dist


def calculate_scores_v2(gt_points, pred_points, gt_radii, pred_radii, threshold=1.5):
    gt_tree = cKDTree(gt_points)
    if len(pred_points):
        pred_tree = cKDTree(pred_points)
    else:
        return 0, 0, 0

    dis_gt2pred, dis_gt2pred_idxs = pred_tree.query(gt_points, k=1)
    dis_pred2gt, _ = gt_tree.query(pred_points, k=1)
    min_distances = {}
    for i in range(len(dis_gt2pred_idxs)):
        tgt_idx = i
        pred_idx = dis_gt2pred_idxs[i]
        distance = dis_gt2pred[i]
        if pred_idx not in min_distances or distance < min_distances[pred_idx][0]:
            min_distances[pred_idx] = (distance, tgt_idx)
    filtered_dis_gt2pred_pred_idxs = np.array([idx for idx, _ in min_distances.items()])
    filtered_dis_gt2pred_tgt_idxs = np.array([min_distances[idx][1] for idx in filtered_dis_gt2pred_pred_idxs])
    filtered_dis_gt2pred = np.array([min_distances[idx][0] for idx in filtered_dis_gt2pred_pred_idxs])

    # Compute Radius MAE
    pred_radii = np.array(pred_radii)
    gt_radii = np.array(gt_radii)
    matched_pred_radii = pred_radii[filtered_dis_gt2pred_pred_idxs]
    matched_tgt_radii = gt_radii[filtered_dis_gt2pred_tgt_idxs]
    radius_mae = np.mean(np.abs(matched_pred_radii - matched_tgt_radii))

    # Compute Acc, Recall, F1
    # Bipartite (BP) version does not count duplicate predictions as true positives
    true_positives = [x for x in dis_gt2pred if x < threshold]
    filtered_true_positives = [x for x in filtered_dis_gt2pred if x < threshold]
    recall = len(true_positives) / len(dis_gt2pred)  # Num true positive / num all ground truths
    recall_nd = len(filtered_true_positives) / len(dis_gt2pred)  # Num true positive / num all ground truths
    acc = len([x for x in dis_pred2gt if x < threshold]) / len(dis_pred2gt)  # Num true positives including duplicate counts / num all predictions
    acc_nd = len(filtered_true_positives) / len(dis_pred2gt)  # Num true positives / num all predictions
    r_f = 0
    r_f_nd = 0
    if acc * recall:
        r_f = 2 * recall * acc / (acc + recall)
        r_f_nd = 2 * recall_nd * acc_nd / (acc_nd + recall_nd)

    return acc, acc_nd, recall, recall_nd, r_f, r_f_nd, radius_mae


def get_recall(gt_points, pred_points, threshold=1.5):
    gt_tree = cKDTree(gt_points)
    if len(pred_points):
        pred_tree = cKDTree(pred_points)
    else:
        return 0, 0, 0

    dis_gt2pred, dis_gt2pred_idxs = pred_tree.query(gt_points, k=1)
    dis_pred2gt, _ = gt_tree.query(pred_points, k=1)
    min_distances = {}
    for i in range(len(dis_gt2pred_idxs)):
        idx = dis_gt2pred_idxs[i]
        distance = dis_gt2pred[i]
        if idx not in min_distances or distance < min_distances[idx]:
            min_distances[idx] = distance

    filtered_dis_gt2pred_idxs = [idx for idx, _ in min_distances.items()]
    filtered_dis_gt2pred = [min_distances[idx] for idx in filtered_dis_gt2pred_idxs]
    filtered_dis_gt2pred = np.array(filtered_dis_gt2pred)

    # accuracy and recall:
    filtered_true_positives = [x for x in filtered_dis_gt2pred if x < threshold]
    recall = len(filtered_true_positives) / len(dis_gt2pred)  # Num true positive / num all ground truths

    return recall


def get_nearest_pred_branches(target_positions, target_idx, pred_positions,
                              pred_branches, pred_ids, threshold=1.5):
    kdt = cKDTree(target_positions)
    dist, idx = kdt.query(pred_positions, k=1)
    pred_idx = np.where(dist < threshold)[0]
    nearest_pred_ids = pred_ids[pred_idx]

    if len(nearest_pred_ids) == 0:
        return target_idx, []

    pred_branch_counts = np.array([np.sum(np.isin(branch, nearest_pred_ids)) for branch in pred_branches])
    k = 10
    nonzero_indices = np.nonzero(pred_branch_counts)[0]
    nonzero_values = pred_branch_counts[nonzero_indices]
    k = min(k, len(nonzero_values))
    topk_indices = nonzero_indices[np.argsort(nonzero_values)[-k:][::-1]]

    return target_idx, topk_indices.tolist()


def compute_score_for_pair(pred, target):
    pred_branches = get_branches(pred.to_undirected())
    target_branches = get_branches(target.to_undirected())
    if len(pred_branches) and len(target_branches):
        return list(calculate_scores_branch_parallel(target, target_branches, pred, pred_branches))
    return None


def get_score_nx_branch(preds, targets, dist=False):
    scores_dict = {'total_branches': [],
                   'num_branches': [],
                   'scores': []}

    with ProcessPoolExecutor() as executor:
        futures = []
        for i, (pred, target) in enumerate(zip(preds, targets)):
            future = executor.submit(compute_score_for_pair, pred, target)
            futures.append((i, future))

        for i, future in tqdm(futures, desc="Overall Progress", unit="task"):
            try:
                result = future.result()
                if result is not None:
                    scores_dict['scores'].append(result)
                else:
                    scores_dict['scores'].append([0, 0, 0, 0, 0, 0])
                pred_branches = get_branches(preds[i].to_undirected())
                target_branches = get_branches(targets[i].to_undirected())
                scores_dict['num_branches'].append(len(pred_branches))
                scores_dict['total_branches'].append(len(target_branches))
            except Exception as e:
                print(f"Error processing future for index {i}: {e}")
                scores_dict['scores'].append([0, 0, 0, 0, 0, 0])
                scores_dict['num_branches'].append(0)
                scores_dict['total_branches'].append(0)

    stats_t = {'num_branches': torch.FloatTensor(scores_dict['num_branches'])}
    stats_t['total_branches'] = torch.FloatTensor(scores_dict['total_branches'])
    stats_t['scores'] = torch.FloatTensor(scores_dict['scores'])

    if dist:
        for key in stats_t:
            stats_t[key] = stats_t[key].to('cuda')
        torch.cuda.synchronize()
        torch.distributed.barrier()
        stats_reduced = utils.gather_stats(stats_t)
    else:
        stats_reduced = stats_t

    stats_reduced['avg_num_branches'] = torch.mean(stats_reduced['num_branches'], dim=0)
    stats_reduced['avg_total_branches'] = torch.mean(stats_reduced['total_branches'], dim=0)
    stats_reduced['avg_scores'] = torch.mean(stats_reduced['scores'], dim=0)

    return stats_reduced


def calculate_scores_branch_parallel(target, target_branches, pred, pred_branches, threshold=1.5):
    num_target, num_pred = len(target_branches), len(pred_branches)
    target_positions_list = [np.array([target.nodes[node]['position'] for node in branch]) for branch in target_branches]
    pred_positions_list = [np.array([pred.nodes[node]['position'] for node in branch]) for branch in pred_branches]
    pred_positions_dict = nx.get_node_attributes(pred, 'position')
    pred_positions = np.array(list(pred_positions_dict.values()))
    pred_ids = np.array(list(pred_positions_dict.keys()))

    target_matches = {}
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(get_nearest_pred_branches, target_positions_list[i], i, pred_positions, pred_branches, pred_ids, threshold)
            for i in range(num_target)
        ]
        for future in tqdm(futures, desc="Target Matching Progress", unit="task"):
            try:
                target_idx, nearest_pred_branches = future.result()  # Catch errors in result()
                target_matches[target_idx] = nearest_pred_branches
            except Exception as e:
                print(f"Error during target matching for index {target_idx}: {e}")
                target_matches[target_idx] = []  # Assign empty match if error occurs

    targets_matched = set()
    with ProcessPoolExecutor() as executor:
        tasks = []
        for i, matches in target_matches.items():
            if i in targets_matched:
                continue
            for j in matches:
                tasks.append(executor.submit(recall_task_for_match, i, j,
                                             target_positions_list, pred_positions_list, threshold))

        for future in tqdm(tasks, desc="Recall Task Progress", unit="task"):
            try:
                is_tp, i = future.result()
                if is_tp:
                    targets_matched.add(i)
            except Exception as e:
                print(f"Error during recall task processing: {e}")

    TP = len(targets_matched)
    FP = num_pred - TP
    FN = num_target - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return TP, FP, FN, precision, recall, f1_score


def calculate_scores_branch(target, target_branches, pred, pred_branches, threshold=1.5):
    num_target, num_pred = len(target_branches), len(pred_branches)

    # Precompute positions
    target_positions_list = [np.array([target.nodes[node]['position'] for node in branch]) for branch in target_branches]
    pred_positions_list = [np.array([pred.nodes[node]['position'] for node in branch]) for branch in pred_branches]

    # Get all pred points
    pred_positions_dict = nx.get_node_attributes(pred, 'position')
    pred_positions = np.array(list(pred_positions_dict.values()))
    pred_ids = np.array(list(pred_positions_dict.keys()))

    # Use ProcessPoolExecutor to parallelize the target matching
    target_matches = {}
    for i in tqdm(range(num_target), desc="Target Matching Progress", unit="task"):
        target_idx, nearest_pred_branches = get_nearest_pred_branches(target_positions_list[i], i, pred_positions, pred_branches, pred_ids, threshold)
        target_matches[target_idx] = nearest_pred_branches

    # Use a set to track matched targets for faster membership check
    targets_matched = set()
    for i, matches in tqdm(target_matches.items(), desc="Recall Task Progress", unit="task"):
        for j in matches:
            recall = get_recall(target_positions_list[i], pred_positions_list[j], threshold)
            if recall > 0.8:
                targets_matched.add(i)
                break

    # Compute TP, FP, FN
    TP = len(targets_matched)
    FP = num_pred - TP
    FN = num_target - TP

    # Compute metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return TP, FP, FN, precision, recall, f1_score


def recall_task_for_match(target_idx, pred_idx, target_positions_list, pred_positions_list, threshold):
    recall = get_recall(target_positions_list[target_idx], pred_positions_list[pred_idx], threshold)
    is_tp = recall > 0.8  # True if recall > 0.8, else False
    return is_tp, target_idx


def get_score_nx_single(preds, targets, elapsed_time, dist=False):
    stats_reduced_sv = {}
    stats_reduced_sv['selected_node'] = get_score_nx(preds, targets, elapsed_time, dist)

    return stats_reduced_sv


def get_score_nx_single_v2(preds, targets, dist=False):
    stats_reduced_sv = {}
    stats_reduced_sv['selected_node'] = get_score_nx_v2(preds, targets, dist)

    return stats_reduced_sv


def get_score_nx_single_branch(preds, targets, dist=False):
    stats_reduced_sv = {}
    stats_reduced_sv['selected_node'] = get_score_nx_branch(preds, targets, dist)

    return stats_reduced_sv


def get_score_nx(preds, targets, elapsed_time, dist=False):
    data_dict = {'preds_nchild_list': [],
                 'targets_nchild_list': [],
                 'preds_bifur_list': [],
                 'targets_bifur_list': [],
                 'preds_list': [],
                 'targets_list': [],}

    for i, (pred, target) in enumerate(zip(preds, targets)):
        for item in data_dict:
            data_dict[item].append([])
        data_dict['preds_list'][i] = list(nx.get_node_attributes(pred, 'position').values())
        out_degrees = dict(pred.out_degree())
        bifur_nodes = [node for node in pred.nodes if out_degrees[node] > 1]
        data_dict['preds_bifur_list'][i] = [pred.nodes[node]['position'] for node in bifur_nodes]
        data_dict['preds_nchild_list'][i] = [out_degrees[node] for node in bifur_nodes]
        data_dict['targets_list'][i] = list(nx.get_node_attributes(target, 'position').values())
        out_degrees = dict(target.out_degree())
        bifur_nodes = [node for node in target.nodes if out_degrees[node] > 1]
        data_dict['targets_bifur_list'][i] = [target.nodes[node]['position'] for node in bifur_nodes]
        data_dict['targets_nchild_list'][i] = [out_degrees[node] for node in bifur_nodes]

    scores_dict = {'total_points': [],
                   'total_bifur_points': [],
                   'num_bifur_points': [],
                   'num_points': [],
                   'bifur_scores': [],
                   'scores': [], }

    # calculate_scores
    for (pred_points, target_points) in zip(data_dict['preds_list'], data_dict['targets_list']):
        if len(pred_points) and len(target_points):
            scores_dict['scores'].append(list(calculate_scores(target_points, pred_points)))
        scores_dict['num_points'].append(len(pred_points))
        scores_dict['total_points'].append(len(target_points))

    for (pred_points, target_points, pred_nchild, target_nchild) in (
            zip(data_dict['preds_bifur_list'], data_dict['targets_bifur_list'],
                data_dict['preds_nchild_list'], data_dict['targets_nchild_list'])):
        if len(pred_points) and len(target_points):
            scores_dict['bifur_scores'].append(list(calculate_scores(target_points, pred_points, target_nchild, pred_nchild, threshold=3.5)))
        elif len(target_points) or len(pred_points):
            scores_dict['bifur_scores'].append([0, 0, 0, 0, 0, 0, np.nan, np.nan])
        else:
            scores_dict['bifur_scores'].append([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.nan, np.nan])
        scores_dict['num_bifur_points'].append(len(pred_points))
        scores_dict['total_bifur_points'].append(len(target_points))

    stats_t = {'num_points': torch.FloatTensor(scores_dict['num_points'])}
    stats_t['scores'] = torch.FloatTensor(scores_dict['scores'])
    stats_t['total_points'] = torch.FloatTensor(scores_dict['total_points'])
    stats_t['num_bifur_points'] = torch.FloatTensor(scores_dict['num_bifur_points'])
    stats_t['bifur_scores'] = torch.FloatTensor(scores_dict['bifur_scores'])
    stats_t['total_bifur_points'] = torch.FloatTensor(scores_dict['total_bifur_points'])
    stats_t['elapsed_time'] = torch.FloatTensor(elapsed_time)

    if dist:
        torch.cuda.synchronize()
        torch.distributed.barrier()
        stats_reduced = utils.gather_stats(stats_t)
    else:
        stats_reduced = stats_t

    stats_reduced['avg_num_points'] = torch.mean(stats_reduced['num_points'], dim=0)
    stats_reduced['avg_scores'] = torch.mean(stats_reduced['scores'], dim=0)
    stats_reduced['avg_total_points'] = torch.mean(stats_reduced['total_points'], dim=0)
    stats_reduced['avg_num_bifur_points'] = torch.mean(stats_reduced['num_bifur_points'], dim=0)
    stats_reduced['avg_bifur_scores'] = torch.nanmean(stats_reduced['bifur_scores'], dim=0)
    stats_reduced['avg_total_bifur_points'] = torch.mean(stats_reduced['total_bifur_points'], dim=0)
    stats_reduced['avg_elapsed_time'] = torch.mean(stats_reduced['elapsed_time'], dim=0)

    return stats_reduced


def get_score_nx_v2(preds, targets, dist=False):
    data_dict = {'preds_list': [],
                 'preds_radii_list': [],
                 'targets_list': [],
                 'targets_radii_list': []}

    for i, (pred, target) in enumerate(zip(preds, targets)):
        for item in data_dict:
            data_dict[item].append([])
        data_dict['preds_list'][i] = list(nx.get_node_attributes(pred, 'position').values())
        data_dict['preds_radii_list'][i] = list(nx.get_node_attributes(pred, 'radius').values())
        data_dict['targets_list'][i] = list(nx.get_node_attributes(target, 'position').values())
        data_dict['targets_radii_list'][i] = list(nx.get_node_attributes(target, 'radius').values())

    scores_dict = {'total_points': [],
                   'num_points': [],
                   'scores': []}
    for (pred_points, pred_radii,
         target_points, targets_radii) in zip(data_dict['preds_list'], data_dict['preds_radii_list'],
                                            data_dict['targets_list'], data_dict['targets_radii_list']):
        if len(pred_points) and len(target_points):
            scores_dict['scores'].append(list(calculate_scores_v2(target_points, pred_points,
                                                                  targets_radii, pred_radii)))
        scores_dict['num_points'].append(len(pred_points))
        scores_dict['total_points'].append(len(target_points))

    stats_t = {'num_points': torch.FloatTensor(scores_dict['num_points'])}
    stats_t['scores'] = torch.FloatTensor(scores_dict['scores'])
    stats_t['total_points'] = torch.FloatTensor(scores_dict['total_points'])

    if dist:
        for key in stats_t:
            stats_t[key] = stats_t[key].to('cuda')
        torch.cuda.synchronize()
        torch.distributed.barrier()
        stats_reduced = utils.gather_stats(stats_t)
    else:
        stats_reduced = stats_t

    stats_reduced['avg_num_points'] = torch.mean(stats_reduced['num_points'], dim=0)
    stats_reduced['avg_scores'] = torch.mean(stats_reduced['scores'], dim=0)
    stats_reduced['avg_total_points'] = torch.mean(stats_reduced['total_points'], dim=0)

    return stats_reduced


def get_stats_message(stats_dict, averages_only=False):
    message = ""
    for key in stats_dict:
        message += f"\n{key} Metrics:"
        stats = stats_dict[key]

        message += "\nAll points\n"
        if not averages_only:
            for i, score in enumerate(stats['scores']):
                message += (f"ID: {i} \t | \t Acc: {score[0].item():.3f} \t "
                            f"| \t Acc-ND: {score[1].item():.3f} \t "
                            f"| \t Recall: {score[2].item():.3f} \t "
                            f"| \t Recall-ND: {score[3].item():.3f} \t "
                            f"| \t F1: {score[4].item():.3f} \t "
                            f"| \t F1-ND: {score[5].item():.3f} \n ")
                message += (f"| \t Num Points: {stats['num_points'][i].item():.3f} \t "
                            f"| \t Tot Points: {int(stats['total_points'][i].item())} \t "
                            f"| \t Time taken: {stats['elapsed_time'][i].item():.3f}\n")

        message += "Average\n"
        message += (f"Acc: {stats['avg_scores'][0]:.3f} \t "
                    f"| \t Acc-ND: {stats['avg_scores'][1]:.3f} \t "
                    f"| \t Recall: {stats['avg_scores'][2]:.3f} \t "
                    f"| \t Recall-ND: {stats['avg_scores'][3]:.3f} \t "
                    f"| \t F1: {stats['avg_scores'][4]:.3f} \t "
                    f"| \t F1-ND: {stats['avg_scores'][5]:.3f} \n ")
        message += (f"| \t Avg Points: {int(stats['avg_num_points'].item())} \t "
                    f"| \t Tot Points: {int(stats['avg_total_points'].item())} \t "
                    f"| \t Avg Time: {stats['avg_elapsed_time'].item():.3f}\n")

        message += "\nBifur points\n"
        if (not averages_only) and (not key == 'average'):
            for i, score in enumerate(stats['bifur_scores']):
                message += (f"ID: {i} \t | \t Acc: {score[0].item():.3f} \t "
                            f"| \t Acc-ND: {score[1].item():.3f} \t "
                            f"| \t Recall: {score[2].item():.3f} \t "
                            f"| \t Recall-ND: {score[3].item():.3f} \t "
                            f"| \t F1: {score[4].item():.3f} \t "
                            f"| \t F1-ND: {score[5].item():.3f} \n "
                            f"| \t NC Error: {score[5].item():.3f} \t "
                            f"| \t Num Points: {int(stats['num_bifur_points'][i].item())} \t "
                            f"| \t Tot Points: {int(stats['total_bifur_points'][i].item())} \t "
                            f"| \t Time taken: {stats['elapsed_time'][i].item():.3f}\n")

        message += "Average\n"
        message += (f"Acc: {stats['avg_bifur_scores'][0]:.3f} \t "
                    f"| \t Acc-ND: {stats['avg_bifur_scores'][1]:.3f} \t "
                    f"| \t Recall: {stats['avg_bifur_scores'][2]:.3f} \t "
                    f"| \t Recall-ND: {stats['avg_bifur_scores'][3]:.3f} \t "
                    f"| \t F1: {stats['avg_bifur_scores'][4]:.3f} \t "
                    f"| \t F1-ND: {stats['avg_bifur_scores'][5]:.3f} \n "
                    f"| \t NC Error: {stats['avg_bifur_scores'][6]:.3f} \t "
                    f"| \t Avg Points: {int(stats['avg_num_bifur_points'].item())} \t "
                    f"| \t Tot Points: {int(stats['avg_total_bifur_points'].item())} \t "
                    f"| \t Avg Time: {stats['avg_elapsed_time'].item():.3f}\n")

    return message


def get_stats_message_v2_branch(stats_dict, averages_only=False):
    message = ""
    for key in stats_dict:
        stats = stats_dict[key]
        if not averages_only:
            for i, score in enumerate(stats['scores']):
                message += (f"ID: {i} \t | \t TP: {score[0].item():.3f} \t "
                            f"| \t FP: {score[1].item():.3f} \t "
                            f"| \t FN: {score[2].item():.3f} \t "
                            f"| \t Precision: {score[3].item():.3f} \t "
                            f"| \t Recall: {score[4].item():.3f} \t "
                            f"| \t F1: {score[5].item():.3f} \n ")
                message += (f"| \t Num Branches: {stats['num_branches'][i].item():.3f} \t "
                            f"| \t Tot Branches: {int(stats['total_branches'][i].item())}\n")

        message += "Average\n"
        message += (f"TP: {stats['avg_scores'][0]:.3f} \t "
                    f"| \t FP: {stats['avg_scores'][1]:.3f} \t "
                    f"| \t FN: {stats['avg_scores'][2]:.3f} \t "
                    f"| \t Precision: {stats['avg_scores'][3]:.3f} \t "
                    f"| \t Recall: {stats['avg_scores'][4]:.3f} \t "
                    f"| \t F1: {stats['avg_scores'][5]:.3f} \n ")
        message += (f"| \t Avg Branches: {int(stats['avg_num_branches'].item())} \t "
                    f"| \t Tot Branches: {int(stats['avg_total_branches'].item())}\n")

    return message


def get_stats_message_v2(stats_dict, sample_ids=None, averages_only=False):
    message = ""
    for key in stats_dict:
        stats = stats_dict[key]
        if not averages_only:
            for i, score in enumerate(stats['scores']):
                if sample_ids:
                    id = sample_ids[i]
                else:
                    id = i
                message += (f"ID: {id} \t | \t Acc: {score[0].item():.3f} \t "
                            f"| \t Acc-ND: {score[1].item():.3f} \t "
                            f"| \t Recall: {score[2].item():.3f} \t "
                            f"| \t Recall-ND: {score[3].item():.3f} \t "
                            f"| \t F1: {score[4].item():.3f} \t "
                            f"| \t F1-ND: {score[5].item():.3f} \n "
                            f"| \t Radius MAE: {score[5].item():.3f} \n ")
                message += (f"| \t Num Points: {stats['num_points'][i].item():.3f} \t "
                            f"| \t Tot Points: {int(stats['total_points'][i].item())}\n")

        message += "Average\n"
        message += (f"TP: {stats['avg_scores'][0]:.3f} \t "
                    f"| \t Acc-ND: {stats['avg_scores'][1]:.3f} \t "
                    f"| \t Recall: {stats['avg_scores'][2]:.3f} \t "
                    f"| \t Recall-ND: {stats['avg_scores'][3]:.3f} \t "
                    f"| \t F1: {stats['avg_scores'][4]:.3f} \t "
                    f"| \t F1-ND: {stats['avg_scores'][5]:.3f} \n "
                    f"| \t Radius MAE: {stats['avg_scores'][6]:.3f} \n ")
        message += (f"| \t Avg Points: {int(stats['avg_num_points'].item())} \t "
                    f"| \t Tot Points: {int(stats['avg_total_points'].item())}\n")

    return message


def betti(G):
    """
    Extract branches from an undirected graph
    """
    num_cc = nx.number_connected_components(G)
    num_cycles = len(list(nx.simple_cycles(G)))
    betti_numbers = [num_cc, num_cycles]

    return betti_numbers
