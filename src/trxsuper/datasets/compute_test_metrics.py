import os
import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.trxsuper.util.eval_utils import (get_score_nx,
                                          get_score_nx_branch,
                                          get_stats_message,
                                          get_stats_message_branch,
                                          betti)
from src.trxsuper.datasets.graph_utils import (get_resampled_graph,
                                               rename_node_names, add_edge_lengths)


def load_graphs(pred_dir, gt_dir, dataset='atm22'):
    """ Load graphs from prediction and ground truth directories """
    pred_files = sorted(list(pred_dir.glob("*.pickle")) + list(pred_dir.glob("*.pkl")))

    if dataset == 'atm22':
        assert len(pred_files) == 60, f"There should be 60 samples in the test set, but found {len(pred_files)}"
    elif dataset == 'parse2022':
        assert len(pred_files) == 20, f"There should be 20 samples in the test set, but found {len(pred_files)}"
    elif dataset == 'syntrx':
        assert len(pred_files) == 100, f"There should be 100 samples in the test set, but found {len(pred_files)}"
    else:
        raise ValueError("Dataset not recognized")

    pred_files = [f for f in pred_files if "temp" not in os.path.basename(f)]

    # load pred and target files
    preds = []
    targets = []
    sample_ids = []
    for pred_file in pred_files:
        sample_id = os.path.basename(pred_file).split('.')[0]
        sample_ids.append(sample_id)
        target_file = gt_dir / (sample_id + '.pickle')
        with open(target_file, 'rb') as handle:
            target = pickle.load(handle)
        targets.append(target['networkx'][0])
        pred = torch.load(pred_file)
        preds.append(pred['preds'][0])

    return preds, targets, sample_ids


def load_graphs_parse_lr(pred_dir):
    """ Load graphs from prediction and ground truth directories """
    pred_files = sorted(list(pred_dir.glob("*.pickle")) + list(pred_dir.glob("*.pkl")))
    if len(pred_files) != 20:
        print(f"There should be 20 samples in the test set, but found {len(pred_files)}")
    assert len(pred_files) == 20, f"There should be 20 samples in the test set, but found {len(pred_files)}"

    pred_files = [f for f in pred_files if "temp" not in os.path.basename(f)]

    # load pred and target files
    preds = []
    targets = []
    sample_ids = []
    for i, pred_file in enumerate(pred_files):
        sample_id = os.path.basename(pred_file).split('.')[0]
        sample_ids.append(sample_id)
        output = torch.load(pred_file)
        preds.append(output['preds'][0])
        targets.append(output['targets'][0])

        if 'resampled' not in output:
            print(f"Resampling graph {i}")
            pred = preds[i]
            target = targets[i]
            old_spacing = 0.8
            new_spacing = 0.5

            # rescale the graph nodes
            for node in pred.nodes:
                pred.nodes[node]['position'] = (np.array(pred.nodes[node]['position']) * old_spacing) / new_spacing
                pred.nodes[node]['radius'] = (pred.nodes[node]['radius'] * old_spacing) / new_spacing
            for node in target.nodes:
                target.nodes[node]['position'] = (np.array(target.nodes[node]['position']) * old_spacing) / new_spacing
                target.nodes[node]['radius'] = (target.nodes[node]['radius'] * old_spacing) / new_spacing

            # resample the graph nodes
            pred = add_edge_lengths(pred)
            pred = get_resampled_graph(pred, points_dist=1.0, smooth=False)
            pred = rename_node_names(pred, "0-0")
            target = add_edge_lengths(target)
            target = get_resampled_graph(target, points_dist=1.0, smooth=False)
            target = rename_node_names(target, "0-0")
            preds[i] = pred
            targets[i] = target
            output['resampled'] = True
            output['preds'][0] = pred
            output['targets'][0] = target

            torch.save(output, pred_file)

    return preds, targets, sample_ids


if __name__ == '__main__':
    dataset = 'atm22'  # 'parse2022' or 'parse2022_lr' or 'syntrx'
    dataset_path = f'./data/{dataset}'
    pred_dirs = []  # List of paths with stored predictions
    mask_dir = Path(f"{dataset_path}/masks_test")
    gt_dir = Path(f"{dataset_path}/annots_test")

    f1 = True
    betti_error = True
    branch_f1 = True

    if f1:
        prec_list = []
        recall_list = []
        f1_list = []
        radii_list = []
        for pred_dir in pred_dirs:
            # load pred and target files
            if dataset == 'parse2022_lr':
                preds, targets, sample_ids = load_graphs_parse_lr(pred_dir)
            else:
                preds, targets, sample_ids = load_graphs(pred_dir, gt_dir, dataset)
            # compute precision, recall and f1 score
            stats = get_score_nx(preds, targets, False)
            prec_list.append(stats['avg_scores'][1])
            recall_list.append(stats['avg_scores'][3])
            f1_list.append(stats['avg_scores'][5])
            radii_list.append(stats['avg_scores'][6])
            stats_message = get_stats_message(stats, sample_ids=sample_ids, averages_only=True)
            print(stats_message)

        prec_list = np.array(prec_list)
        mean_prec = np.mean(prec_list)
        std_prec = np.std(prec_list, ddof=1)  # Sample standard deviation
        recall_list = np.array(recall_list)
        mean_recall = np.mean(recall_list)
        std_recall = np.std(recall_list, ddof=1)  # Sample standard deviation
        f1_list = np.array(f1_list)
        mean_f1 = np.mean(f1_list)
        std_f1 = np.std(f1_list, ddof=1)  # Sample standard deviation
        radii_list = np.array(radii_list)
        mean_radii = np.mean(radii_list)
        std_radii = np.std(radii_list, ddof=1)  # Sample standard deviation
        print(f"Precision: {mean_prec:.4f} ± {std_prec:.4f}")
        print(f"Recall: {mean_recall:.4f} ± {std_recall:.4f}")
        print(f"F1: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"Radii: {mean_radii:.4f} ± {std_radii:.4f}")

    if betti_error:
        betti_mean_list = []
        for pred_dir in pred_dirs:
            betti_list = []
            preds, targets, _ = load_graphs(pred_dir, gt_dir, dataset)
            for pred, target in tqdm(zip(preds, targets)):
                beta = np.array([1, 0])
                beta_pred = np.array(betti(pred.to_undirected()))
                betti_list.append(np.abs(beta_pred - beta))
            mean_betti = np.mean(np.array(betti_list), axis=0)
            betti_mean_list.append(mean_betti)
            print(f"Betti error: {mean_betti}")

        betti_mean_list = np.array(betti_mean_list)
        betti_0_list = betti_mean_list[:, 0]
        betti_1_list = betti_mean_list[:, 1]
        mean_betti_0 = np.mean(betti_0_list)
        mean_betti_1 = np.mean(betti_1_list)
        std_betti_0 = np.std(betti_0_list, ddof=1)
        std_betti_1 = np.std(betti_1_list, ddof=1)
        print(f"Betti 0: {mean_betti_0:.4f} ± {std_betti_0:.4f}")
        print(f"Betti 1: {mean_betti_1:.4f} ± {std_betti_1:.4f}")

    if branch_f1:
        prec_list = []
        recall_list = []
        f1_list = []
        for pred_dir in pred_dirs:
            # load pred and target files
            if dataset == 'parse2022_lr':
                preds, targets, _ = load_graphs_parse_lr(pred_dir)
            else:
                preds, targets, _ = load_graphs(pred_dir, gt_dir, dataset)
            # compute precision, recall and f1 score
            stats = get_score_nx_branch(preds, targets, False)
            prec_list.append(stats['avg_scores'][3])
            recall_list.append(stats['avg_scores'][4])
            f1_list.append(stats['avg_scores'][5])
            stats_message = get_stats_message_branch(stats, averages_only=True)
            print(stats_message)

        prec_list = np.array(prec_list)
        mean_prec = np.mean(prec_list)
        std_prec = np.std(prec_list, ddof=1)  # Sample standard deviation
        recall_list = np.array(recall_list)
        mean_recall = np.mean(recall_list)
        std_recall = np.std(recall_list, ddof=1)  # Sample standard deviation
        f1_list = np.array(f1_list)
        mean_f1 = np.mean(f1_list)
        std_f1 = np.std(f1_list, ddof=1)  # Sample standard deviation
        print(f"Precision: {mean_prec:.4f} ± {std_prec:.4f}")
        print(f"Recall: {mean_recall:.4f} ± {std_recall:.4f}")
        print(f"F1: {mean_f1:.4f} ± {std_f1:.4f}")


