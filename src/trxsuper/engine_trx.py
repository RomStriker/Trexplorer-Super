# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import os
import numpy as np
from typing import Iterable
import time
import networkx as nx
from tqdm import tqdm
import torch
from src.trxsuper.util import misc as utils
from monai.transforms import (
    Compose,
    LoadImage,
    ThresholdIntensity,
    NormalizeIntensity)
from src.trxsuper.datasets.transforms_vtl import (
    CropAndPad,
    LoadAnnotPickle)


class TrexplorerSuper:
    def __init__(self, args, logger, device):
        self.args = args
        self.logger = logger
        self.device = device

    @staticmethod
    def get_single_step_targets(targets, step):
        """
        Extracting targets for current step
        """
        curr_step_targets = []
        for target in targets:
            curr_step_targets_dict = {'labels': torch.tensor(target['labels'][step]),
                                      'directions': torch.tensor(target['rel_positions'][step]).float(),
                                      'radii': torch.tensor(target['radii'][step]).float(),
                                      'parent_idxs': torch.tensor(target['parent_idxs'][step]).long()}
            curr_step_targets.append(curr_step_targets_dict)

        return curr_step_targets

    def train_one_epoch_vtl(self, model: torch.nn.Module, criterion: torch.nn.Module,
                             data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler,
                             scaler: torch.cuda.amp.GradScaler):
        """
        Training function for one epoch
        """
        model.train()
        criterion.train()
        skip_lr_step = False
        epoch_metrics = {'scaled_losses': 0, 'unscaled_losses': 0, 'total_loss': 0, }
        for i, batch in enumerate(data_loader):
            for j, sub_batch in enumerate(batch):
                node = 'selected_node' if j == 0 else 'pair_node'
                losses = []
                batch_metrics = {'scaled_losses': [], 'unscaled_losses': []}
                sample_imgs, sample_past_trs, targets = (sub_batch["image"], sub_batch["past_tr"], sub_batch["label"])

                if sample_imgs.device != self.device:
                    sample_imgs = sample_imgs.to(self.device)
                if sample_past_trs is not None:
                    sample_past_trs = sample_past_trs.to(self.device)

                # prev_step_info
                if node == 'pair_node':
                    # create new indices based on tracking in the previous sub volume
                    indices = prev_step_info['indices']
                    new_indices = []

                    # find the targets in the selected node that are used as a starting point for the pair node
                    for sample_idx in range(len(targets)):
                        prev_sv_last_step_node_ids = batch[j-1]["label"][sample_idx]['node_ids'][-1]
                        curr_sv_first_step_node_id = targets[sample_idx]['node_ids'][0][0]
                        target_idx = prev_sv_last_step_node_ids.index(curr_sv_first_step_node_id)
                        target_idx_in_indices = (indices[sample_idx][1] == target_idx).nonzero().item()
                        query_idx = indices[sample_idx][0][target_idx_in_indices].item()
                        new_indices.append([torch.tensor([query_idx]), torch.tensor([0])])

                    prev_step_info['first_step'] = True
                    prev_step_info['memory'] = None
                    prev_step_info['pos'] = None
                    prev_step_info['bifur_list'] = None
                    prev_step_info['out'] = None
                    prev_step_info['old_indices'] = None
                    prev_step_info['node_type'] = node
                    prev_step_info['indices'] = new_indices
                    prev_step_info['hs_without_norm'] = prev_step_info['hs_without_norm'].detach()
                    prev_step_targets = self.get_single_step_targets(targets, 0)
                    prev_step_targets = [utils.nested_dict_to_device(t, self.device) for t in prev_step_targets]
                else:
                    prev_step_info = {"first_step": True,
                                      "indices": [],
                                      "out": None,
                                      "bifur_list": None,
                                      "node_type": node}
                    prev_step_targets = None

                # predicting for the number of steps in our sequence
                for step in range(1, self.args.seq_len):
                    curr_step_targets = self.get_single_step_targets(targets, step)
                    curr_step_targets = [utils.nested_dict_to_device(t, self.device) for t in curr_step_targets]

                    with torch.cuda.amp.autocast(enabled=self.args.amp):
                        norm_step = torch.tensor((step - 1) / (self.args.seq_len - 1)).unsqueeze(0).unsqueeze(0).to(self.device)
                        outputs, prev_step_info = model(sample_imgs, sample_past_trs, prev_step_info, norm_step,
                                                        curr_step_targets, prev_step_targets)

                        # compute loss
                        loss_dict, prev_step_info['indices'] = criterion(outputs, curr_step_targets, prev_step_targets,
                                                                         prev_step_info)
                        weight_dict = criterion.weight_dict
                        losses.append(sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict))

                    # reduce losses over all GPUs for logging purposes
                    loss_dict_reduced = utils.reduce_dict(loss_dict)
                    loss_dict_reduced_unscaled = {
                        f'{k}_unscaled': v for k, v in loss_dict_reduced.items() if k in weight_dict}
                    losses_reduced_unscaled = sum(loss_dict_reduced_unscaled.values())
                    loss_dict_reduced_scaled = {
                        k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
                    losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

                    batch_metrics['scaled_losses'].append(losses_reduced_scaled.item())
                    batch_metrics['unscaled_losses'].append(losses_reduced_unscaled.item())
                    for item in loss_dict_reduced:
                        if item in batch_metrics:
                            batch_metrics[item].append(loss_dict_reduced[item].item())
                        else:
                            batch_metrics.update({item: [loss_dict_reduced[item].item()]})

                    prev_step_targets = curr_step_targets

                for item in batch_metrics:
                    if item in epoch_metrics:
                        # normalize by number of steps to get the average loss of a query/object per step
                        epoch_metrics[item] += sum(batch_metrics[item]) / (self.args.seq_len - 1)
                    else:
                        epoch_metrics.update({item: sum(batch_metrics[item]) / (self.args.seq_len - 1)})

                # sum the losses for all steps
                total_losses = sum(losses)
                epoch_metrics['total_loss'] += total_losses.item()
                epoch_metrics['scaled_losses'] /= self.args.dec_layers
                epoch_metrics['unscaled_losses'] /= self.args.dec_layers

                if self.args.amp:
                    optimizer.zero_grad()
                    scaler.scale(total_losses).backward()
                    if self.args.clip_max_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_max_norm)
                    scaler.step(optimizer)
                    scale = scaler.get_scale()
                    scaler.update()
                    skip_lr_step = (scale > scaler.get_scale())
                else:
                    optimizer.zero_grad()
                    total_losses.backward()
                    if self.args.clip_max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_max_norm)
                    optimizer.step()

            if skip_lr_step:
                self.logger("Skipping LR Scheduler step.")
            else:
                lr_scheduler.step()

        for item in epoch_metrics:
            epoch_metrics[item] = epoch_metrics[item] / len(data_loader)
        epoch_metrics['lr'] = lr_scheduler.get_last_lr()[0]

        return epoch_metrics

    def create_node_batch_nx_sv(self, targets):
        """
        Create a list of nodes from the targets
        """
        node_batch = [nx.DiGraph() for _ in range(len(targets))]
        for i, target in enumerate(targets):
            node_pos = np.array([self.args.sub_vol_size/2, self.args.sub_vol_size/2,
                                 self.args.sub_vol_size/2]) + np.array(target['rel_positions'][0][0])*self.args.sub_vol_size/2
            node_batch[i].add_node('0-0',
                                   position=node_pos.tolist(),
                                   radius=target['radii'][0][0],
                                   label=target['labels'][0][0],
                                   level=0, depth=0, rel_pos=[0, 0, 0],
                                   query_index=-1, original_query_index=-1,
                                   step=-1)
        return node_batch

    def get_global_pred_root_node_nx(self, pred_tree, target_tree):
        """
        Create a Node for the root point
        """
        in_degrees = dict(target_tree.in_degree())
        root_node_id = [node for node, in_degree in in_degrees.items() if in_degree == 0][0]
        root_node_info = self.get_node(root_node_id, target_tree)
        root_out_degree = target_tree.out_degree(root_node_id)

        root_pos = root_node_info['position']
        rel_pos = np.array(root_pos) - np.array(root_pos).astype(int)
        radius = root_node_info['radius']
        pred_tree.add_node(root_node_id, position=root_pos,
                           rel_pos=rel_pos, radius=radius, level=0, depth=0, 
                           query_index=-1, original_query_index=-1, score=1.0,
                           label=root_out_degree)

        return root_node_info

    def get_classes_radii_positions(self, outputs):
        """
        Obtain the classes, radii, and positions for all the queries
        """
        query_classes = outputs['class_logits']
        query_classes = torch.argmax(query_classes, dim=2)
        query_radii = outputs['radius_logits']
        query_radii = query_radii * (self.args.sub_vol_size // 2)
        query_positions = outputs['direc_logits']
        query_positions = query_positions * (self.args.sub_vol_size // 2)

        return query_classes, query_radii, query_positions

    def get_cont_next_node_ar2_nx(self, query_index, step_node, pred_tree, curr_node_root_pos, query_positions,
                                  query_radii, query_classes, level, step=None, point_hidden_state=None):
        """
        Get the continuing node
        """
        branch_id = int(step_node.split("-")[0])
        point_id = int(step_node.split("-")[1]) + 1
        query_index_t = torch.tensor(query_index).to(query_positions.device)
        rel_pos = query_positions.index_select(1, query_index_t).squeeze()
        radius = query_radii.index_select(1, query_index_t).squeeze()
        label = query_classes.index_select(1, query_index_t).squeeze()
        position = curr_node_root_pos + rel_pos

        next_node = str(branch_id) + "-" + str(point_id)
        step_node_info = self.get_node(step_node, pred_tree)
        pred_tree.add_node(next_node,
                           position=position.tolist(),
                           rel_pos=rel_pos.tolist(),
                           radius=radius.item(),
                           query_index=query_index,
                           original_query_index=query_index,
                           level=level,
                           depth=step_node_info['depth'] + 1,
                           step=step,
                           label=label.item(),
                           hidden_state=point_hidden_state,
                           )

        parent_pos = torch.tensor(step_node_info['position']).to(query_positions.device)
        child_pos = torch.tensor(self.get_node(next_node, pred_tree)['position']).to(query_positions.device)
        length = torch.norm(child_pos - parent_pos).item()
        pred_tree.add_edge(step_node, next_node, length=length)

        return next_node

    def get_new_next_nodes_ar2_nx(self, new_branches, parent_node, pred_tree, query_positions, query_radii,
                                  query_classes, curr_node_root_pos, global_branch_id, level, step=None, point_hidden_state=None):
        """
        Get the new nodes originating at a bifurcation point
        """
        # initialize a list for nodes of new branches
        new_branch_node_list = []
        # Add new branches to the tree whose parents are in the list parent_indices_filter
        for query_index in new_branches:
            query_index_t = torch.tensor(query_index).to(query_positions.device)
            rel_pos = query_positions.index_select(1, query_index_t).squeeze()
            radius = query_radii.index_select(1, query_index_t).squeeze()
            label = query_classes.index_select(1, query_index_t).squeeze()
            position = curr_node_root_pos + rel_pos
            global_branch_id += 1
            branch_id = global_branch_id
            point_id = 0

            new_node = str(branch_id) + "-" + str(point_id)
            pred_tree.add_node(new_node,
                               position=position.tolist(),
                               rel_pos=rel_pos.tolist(),
                               radius=radius.item(),
                               query_index=query_index,
                               original_query_index=query_index,
                               level=level,
                               depth=self.get_node(parent_node, pred_tree)['depth'] + 1,
                               step=step,
                               label=label.item(),
                               hidden_state=point_hidden_state[query_index] if point_hidden_state is not None else None,)
            pred_tree.add_edge(parent_node, new_node)
            parent_pos = torch.tensor(self.get_node(parent_node, pred_tree)['position']).to(query_positions.device)
            child_pos = torch.tensor(self.get_node(new_node, pred_tree)['position']).to(query_positions.device)
            length = torch.norm(child_pos - parent_pos).item()
            pred_tree.add_edge(parent_node, new_node, length=length)
            new_branch_node_list.append(new_node)

        return new_branch_node_list, global_branch_id

    def get_updated_indices_nx(self, indices, curr_step, pred_tree):
        if len(indices) == 0 or sum(len(lst) for lst in indices) == 0:
            branch_indices = [int(node.split("-")[0]) for node in curr_step]
            query_indices = [self.get_node(node, pred_tree)['query_index'] for node in curr_step]
            indices = [[torch.tensor(query_indices), torch.tensor(branch_indices)]]
        else:
            indices_list = [indices[0][0].tolist(), indices[0][1].tolist()]
            query_indices = [self.get_node(node_i, pred_tree)['query_index'] for node_i in curr_step
                             if self.get_node(node_i, pred_tree)['query_index'] not in indices_list[0]]
            branch_indices = [int(node_i.split("-")[0]) for node_i in curr_step
                              if ((int(node_i.split("-")[0]) not in indices_list[1]) or (int(node_i.split("-")[0]) == 10 ** 6))]
            indices = [[torch.tensor(indices_list[0] + query_indices),
                        torch.tensor(indices_list[1] + branch_indices)]]

        for sample in indices:
            sorted_targets, sort_perm = torch.sort(sample[1])
            if (sample[1] != sorted_targets).any():
                for i in range(2):
                    sample[i] = sample[i][sort_perm]

        if indices[0][0].size(0) == 0:
            indices[0] = []

        return indices

    def log_progress(self, tree_id):
        if self.args.eval_only:
            self.logger.info(f'Tree ID: {tree_id}')

    def check_finished(self, targets, tree_id, curr_level, level):
        break_loop = False
        if level >= self.args.max_inference_levels and self.args.eval_limit_levels:
            self.logger.info(f"Max inference levels reached! Sample: {str(targets[0]['index'])}, Tree ID: {tree_id:d}")
            break_loop = True

        if len(curr_level) > self.args.max_nodes_per_level and self.args.eval_limit_nodes_per_level:
            self.logger.info(f"Max nodes in a level reached! Skipping further evaluation.  "
                             f"Sample: {targets[0]['index']}, Tree ID: {tree_id}")
            break_loop = True
        return break_loop

    def log_progress_level(self, curr_level, level):
        self.logger.info(f"Level: {level} \t | \t Nodes: {len(curr_level)} \t ")

    @staticmethod
    def get_node(node_id, tree):
        root_node_info = tree.nodes[node_id]
        root_node_info['id'] = node_id
        return root_node_info

    def generate_past_trajectory_pp_nx(self, tree, node_id):
        past_traj_label = []
        current_node = self.get_node(node_id, tree)
        parent = current_node
        root_position = np.asarray(current_node['position'])

        for _ in range(self.args.num_prev_pos):
            node_info_list = []
            node_position = (np.asarray(parent['position']) - root_position).astype(float)
            node_position /= (self.args.focus_vol_size // 2)
            node_info_list.append(node_position)
            node_radius = np.asarray(parent['radius']).reshape(1)
            node_radius /= (self.args.sub_vol_size // 2)
            node_info_list.append(node_radius)
            past_traj_label.append(np.concatenate(node_info_list))

            parent_id = list(tree.predecessors(parent['id']))
            if not len(parent_id):
                break
            parent = self.get_node(parent_id[0], tree)

        past_traj_label_np = np.asarray(past_traj_label)

        num_missing_points = self.args.num_prev_pos - past_traj_label_np.shape[0]
        if num_missing_points:
            last_point_info = past_traj_label_np[-1:]
            padding_info = np.tile(last_point_info, (num_missing_points, 1))
            past_traj_label_np = np.vstack((past_traj_label_np, padding_info))
        past_traj_label_t = torch.FloatTensor(past_traj_label_np).flatten()

        return past_traj_label_t

    def get_sub_vol_and_past_tr(self, node_batch, samples, pred_tree, samples_min, crop_pad):
        """
        Get the sub-volume and past trajectory for the nodes in the node_batch
        """
        sub_vol_batch = []
        past_traj_pos_batch = []
        to_remove = []
        for node in node_batch:
            image_size = torch.tensor(samples.squeeze().size())
            node_pos = torch.tensor(self.get_node(node, pred_tree)['position'])
            if torch.logical_and(torch.all(node_pos < (image_size - 1)), torch.all(node_pos >= 0)):
                sub_vol = crop_pad(samples.squeeze(dim=0), node_pos.tolist(), samples_min.item())
                sub_vol = sub_vol.unsqueeze(0)
            else:
                to_remove.append(node)
                continue

            # A quickfix
            if len(sub_vol.shape) == 6:
                sub_vol = sub_vol.squeeze(0)

            past_traj_pos = self.generate_past_trajectory_pp_nx(pred_tree, node)
            past_traj_pos = past_traj_pos.to(self.device).unsqueeze(0)
            sub_vol_batch.append(sub_vol)
            past_traj_pos_batch.append(past_traj_pos)

        # remove the nodes from the node_batch that are outside the image
        for node in to_remove:
            node_batch.remove(node)

        return sub_vol_batch, past_traj_pos_batch

    @staticmethod
    def update_perma_finished_branches(node_perma_finished_branches, perma_end_query, query_classes,
                                       perma_end_classes, selected_queries):
        """
        Update the permanently finished branches list
        """
        if perma_end_query is not None:
            node_perma_finished_branches.append(perma_end_query)
        if query_classes[:, selected_queries[0]] in perma_end_classes:
            node_perma_finished_branches.append(selected_queries[0])

    @staticmethod
    def update_perma_finished_branches_bifur(node_perma_finished_branches,
                                             perma_end_classes, query_classes, selected_queries):
        perma_end_mask = torch.isin(query_classes, torch.tensor(perma_end_classes).to(query_classes.device))
        perma_end_query_idxs = torch.nonzero(perma_end_mask, as_tuple=True)[1]
        perma_finished_queries = [query_idx for query_idx in selected_queries
                                  if query_idx in perma_end_query_idxs.tolist()]
        if len(perma_finished_queries):
            node_perma_finished_branches += perma_finished_queries

    @staticmethod
    def get_old_to_new_idx_dict(prev_step_info):
        """
        Get the mapping of old indices to new indices
        """
        old_indices = prev_step_info['old_indices']
        new_indices = prev_step_info['indices']
        if len(old_indices) == 0 or sum(len(lst) for lst in old_indices) == 0:
            return None
        else:
            same_indices = (torch.cat([s_idx[0] for s_idx in old_indices if len(s_idx)], dim=0) ==
                            torch.cat([s_idx[0] for s_idx in new_indices if len(s_idx)], dim=0)).all()
            if same_indices:
                return None
            else:
                index_mapping = []
                for old_sample_idxs, new_sample_idx in zip(old_indices, new_indices):
                    if not len(old_sample_idxs):
                        index_mapping.append(None)
                    else:
                        index_mapping.append({old_idx: new_idx for old_idx, new_idx in zip(old_sample_idxs[0].tolist(), new_sample_idx[0].tolist())})
                return index_mapping

    def map_old_to_new_indices(self, pred_tree, curr_step_batch, node_finished_branches_batch,
                               finished_branches_end_nodes_batch, node_perma_finished_branches_batch, prev_step_info):
        """
        Map new indices to old indices
        """
        index_mapping = self.get_old_to_new_idx_dict(prev_step_info)
        if index_mapping:
            for sample_id, sample_mapping in enumerate(index_mapping):
                if sample_mapping:
                    for node in curr_step_batch[sample_id]:
                        pred_tree.nodes[node]['query_index'] = sample_mapping[pred_tree.nodes[node]['query_index']]

                    if len(node_finished_branches_batch[sample_id]):
                        node_finished_branches_batch[sample_id] = [sample_mapping[node] if node != -1 else -1
                                                                   for node in node_finished_branches_batch[sample_id]]
                        for node in finished_branches_end_nodes_batch[sample_id]:
                            node_query_index = pred_tree.nodes[node]['query_index']
                            pred_tree.nodes[node]['query_index'] = sample_mapping[node_query_index] if node_query_index != -1 else -1

                    if len(node_perma_finished_branches_batch[sample_id]):
                        node_perma_finished_branches_batch[sample_id] = [sample_mapping[node]
                                                                         for node in node_perma_finished_branches_batch[sample_id]]

    def map_old_to_new_indices_sv(self, pred_tree_batch, curr_step_batch, node_finished_branches_batch,
                               finished_branches_end_nodes_batch, node_perma_finished_branches_batch, prev_step_info):
        """
        Map new indices to old indices for sub-volume evaluation
        """
        index_mapping = self.get_old_to_new_idx_dict(prev_step_info)
        if index_mapping:
            for sample_id, sample_mapping in enumerate(index_mapping):
                if sample_mapping:
                    pred_tree = pred_tree_batch[sample_id]
                    for node in curr_step_batch[sample_id]:
                        pred_tree.nodes[node]['query_index'] = sample_mapping[pred_tree.nodes[node]['query_index']]

                    if len(node_finished_branches_batch[sample_id]):
                        node_finished_branches_batch[sample_id] = [sample_mapping[node] if node != -1 else -1
                                                                   for node in node_finished_branches_batch[sample_id]]
                        for node in finished_branches_end_nodes_batch[sample_id]:
                            node_query_index = pred_tree.nodes[node]['query_index']
                            pred_tree.nodes[node]['query_index'] = sample_mapping[node_query_index] if node_query_index != -1 else -1

                    if len(node_perma_finished_branches_batch[sample_id]):
                        node_perma_finished_branches_batch[sample_id] = [sample_mapping[node]
                                                                         for node in node_perma_finished_branches_batch[sample_id]]

    def add_new_bifur_branches_points(self, bifur_dict, pred_tree, new_branches, curr_step,
                                      finished_branches_end_nodes, global_branch_id, query_positions,
                                      query_radii, curr_node_root_pos, level, step, next_step,
                                      node_perma_finished_branches, perma_end_classes, query_classes,
                                      selected_queries, point_hidden_state=None):
        """
        Add new bifurcation branches points
        """
        for bifur_id in bifur_dict:
            allocated_bifur_queries = bifur_dict[bifur_id][1]
            if len(allocated_bifur_queries):
                unused_buffer_queries = [idx for idx in allocated_bifur_queries if idx not in new_branches]
                used_bifur_queries = [idx for idx in allocated_bifur_queries if idx not in unused_buffer_queries]
                parent_query_index = bifur_dict[bifur_id][0]
                all_parents = curr_step + finished_branches_end_nodes
                parent_node = [step_node for step_node in all_parents if
                               self.get_node(step_node, pred_tree)['query_index'] == parent_query_index][0]
                new_branch_node_list, global_branch_id[0] = self.get_new_next_nodes_ar2_nx(used_bifur_queries,
                                                                                           parent_node, pred_tree, query_positions, query_radii,
                                                                                           query_classes, curr_node_root_pos, global_branch_id[0],
                                                                                           level, step, point_hidden_state)

                next_step += new_branch_node_list
                self.update_perma_finished_branches_bifur(node_perma_finished_branches, perma_end_classes,
                                                          query_classes, selected_queries)

    def add_new_bifur_branches_points_beg(self, global_branch_id, selected_queries, curr_step, pred_tree,
                                          query_positions, query_radii, curr_node_root_pos, level, step,
                                          next_step, node_finished_branches, finished_branches_end_nodes,
                                          node_perma_finished_branches, perma_end_classes, query_classes,
                                          point_hidden_state=None):
        """
        Add new bifurcation branches points for the first step
        """
        new_branch_node_list, global_branch_id[0] = self.get_new_next_nodes_ar2_nx(selected_queries,
                                                                                   curr_step[0], pred_tree, query_positions, query_radii, query_classes,
                                                                                   curr_node_root_pos, global_branch_id[0], level,
                                                                                   step, point_hidden_state)
        next_step += new_branch_node_list
        finished_branches = [-1]
        node_finished_branches += finished_branches
        finished_branches_end_nodes += [end_node for end_node in curr_step
                                        if self.get_node(end_node, pred_tree)['query_index'] in finished_branches]
        self.update_perma_finished_branches_bifur(node_perma_finished_branches, perma_end_classes,
                                                  query_classes, selected_queries)

    def add_cont_branches_points(self, continuing_branches, previous_selected_queries, curr_step,
                                 pred_tree, curr_node_root_pos, query_positions, query_radii, level, step,
                                 next_step, node_perma_finished_branches, query_classes,
                                 perma_end_classes, selected_queries, point_hidden_state=None):
        for query_index in continuing_branches:
            node_index = previous_selected_queries.index(query_index)
            step_node = curr_step[node_index]

            # get the node for the next step of this continuing branch
            next_step_node, perma_end_query = self.get_cont_next_node_ar2_nx(query_index, step_node,
                                                                             pred_tree, curr_node_root_pos, query_positions, query_radii,
                                                                             query_classes, level, step,
                                                                             point_hidden_state[query_index] if point_hidden_state is not None else None)

            # add the next step node's position to the global list of positions
            if next_step_node is not None:
                next_step.append(next_step_node)

            # update permanently finished branches
            if self.args.eval_perma_finish_end_bifur and node_perma_finished_branches is not None:
                self.update_perma_finished_branches(node_perma_finished_branches,
                                                    perma_end_query, query_classes, perma_end_classes,
                                                    selected_queries)

    def add_cont_branch_point_beg(self, selected_queries, curr_step, pred_tree, curr_node_root_pos, query_positions,
                                  query_radii, level, step, next_step, node_perma_finished_branches,
                                  query_classes, perma_end_classes, point_hidden_state=None):
        """
        Add continuing branches points for the first step
        """
        next_step_node, perma_end_query = self.get_cont_next_node_ar2_nx(selected_queries[0],
                                                                         curr_step[0], pred_tree, curr_node_root_pos, query_positions,
                                                                         query_radii, query_classes, level, step, point_hidden_state)
        if next_step_node is not None:
            next_step.append(next_step_node)
        self.update_perma_finished_branches(node_perma_finished_branches, perma_end_query,
                                            query_classes, perma_end_classes, selected_queries)

    def init_req_lists(self, indices, pred_tree, curr_step, selected_queries,
                       node_finished_branches, finished_branches_end_nodes):
        """
        Initialize the required lists
        """
        used_queries = indices[0][0].tolist() if len(indices[0]) else []
        previous_selected_queries = [self.get_node(node_i, pred_tree)['query_index'] for node_i in curr_step]
        new_branches = (list(set(selected_queries) - set(used_queries)))
        continuing_branches = [x for x in selected_queries if x in used_queries]
        finished_branches = list(set(previous_selected_queries) - set(selected_queries))
        node_finished_branches += finished_branches
        finished_branches_end_nodes += [end_node for end_node in curr_step
                                        if self.get_node(end_node, pred_tree)['query_index'] in finished_branches]

        return continuing_branches, new_branches

    @torch.no_grad()
    def evaluate_val(self, model, data_loader):
        model.eval()
        all_masks = []
        all_samples = []
        all_samples_ids = []
        all_preds = []
        all_targets = []
        elapsed_time = []
        crop_pad = CropAndPad(self.args.sub_vol_size, 'area')

        classes_to_filter = ['background', 'pad'] if self.args.pad_class else ['background']
        # classes that will permanently end tracking for a branch with point of these classes
        perma_end_classes = [self.args.class_dict['bifurcation'], self.args.class_dict['end']]

        # Only batch_size 1 supported for now
        for i, batch in enumerate(data_loader):
            samples, samples_min, targets, masks = (batch["image"], batch["image_min"], batch["label"], batch["mask"])
            if self.args.eval_only:
                self.logger.info(f'Sample: {targets[0]["index"]:d}')
            for tree_id in range(len(targets[0]['networkx'])):
                st = time.time()
                self.log_progress(tree_id)
                finished = False
                samples = samples.to(self.args.device)
                global_branch_id = [0]
                level = 0
                target_tree = targets[0]['networkx'][tree_id]
                pred_tree = nx.DiGraph()
                root_node = self.get_global_pred_root_node_nx(pred_tree, target_tree)
                curr_level = [root_node['id']]
                while not finished:
                    if self.check_finished(targets, tree_id, curr_level, level):
                        break
                    if self.args.eval_only:
                        self.log_progress_level(curr_level, level)

                    next_level = []
                    for level_node_i in range(0, len(curr_level), self.args.batch_size_per_sample):
                        node_batch = curr_level[level_node_i: level_node_i + self.args.batch_size_per_sample]
                        sub_vol_batch, past_traj_pos_batch = self.get_sub_vol_and_past_tr(node_batch, samples,
                                                                                          pred_tree, samples_min, crop_pad)
                        if not len(sub_vol_batch):
                            continue

                        # prev_step_info
                        prev_step_info = {"first_step": True,
                                          "out": None,
                                          "bifur_list": None,
                                          "starting_node_id": [int(node.split("-")[0]) for node in node_batch],
                                          "global_id": global_branch_id,
                                          "node_type": "selected_node",
                                          "hidden_state_list": []}

                        # get hidden states of prev_level_nodes
                        if node_batch[0] != root_node['id']:
                            prev_level_hs = torch.zeros([len(node_batch), 1, self.args.hidden_dim], device=self.device)
                            for curr_node_idx, curr_node in enumerate(node_batch):
                                curr_node_info = self.get_node(curr_node, pred_tree)
                                prev_level_hs[curr_node_idx, 0, :] = curr_node_info['hidden_state']
                            prev_step_info['hs_without_norm'] = prev_level_hs
                            prev_step_info['node_type'] = 'pair_node'
                            prev_step_info['labels'] = [curr_node_info['label'] for curr_node_info in
                                                        [self.get_node(curr_node, pred_tree) for curr_node in node_batch]]
                            query_indices_batch = [torch.tensor([0]) for _ in node_batch]
                            query_targets_batch = [torch.tensor([int(curr_node.split('-')[0])]) for curr_node in node_batch]
                            prev_step_info['indices'] = [list(t) for t in zip(query_indices_batch, query_targets_batch)]
                        else:
                            prev_step_info['indices'] = [[] for _ in range(len(node_batch))]

                        finish_flag = [False for i in range(len(node_batch))]
                        sub_vol = torch.cat(sub_vol_batch, dim=0)
                        past_traj_pos = torch.cat(past_traj_pos_batch, dim=0)
                        curr_step_batch = [[node] for node in node_batch]
                        curr_node_root_pos_batch = [torch.tensor(self.get_node(node, pred_tree)['position'],
                                                                 dtype=int).to(self.args.device) for node in node_batch]
                        node_finished_branches_batch = [[] for i in range(len(node_batch))]
                        finished_branches_end_nodes_batch = [[] for i in range(len(node_batch))]
                        node_perma_finished_branches_batch = [[] for i in range(len(node_batch))]

                        for step in range(1, self.args.seq_len):
                            num_all_nodes = sum([len(sublist) for sublist in curr_step_batch])
                            if not num_all_nodes:
                                break
                            next_step_batch = [[] for i in range(len(node_batch))]

                            norm_step = torch.tensor((step - 1) / (self.args.seq_len - 1)).unsqueeze(0).unsqueeze(0).to(self.args.device)
                            outputs, prev_step_info = model(sub_vol, past_traj_pos, prev_step_info, norm_step)
                            self.map_old_to_new_indices(pred_tree,
                                                        curr_step_batch,
                                                        node_finished_branches_batch,
                                                        finished_branches_end_nodes_batch,
                                                        node_perma_finished_branches_batch,
                                                        prev_step_info)

                            query_classes_batch, query_radii_batch, query_positions_batch = self.get_classes_radii_positions(outputs)
                            filter_class_ids = torch.tensor([self.args.class_dict[key]
                                                             for key in classes_to_filter]).to(query_classes_batch.device)
                            query_classes_batch_filtered = torch.logical_not(torch.isin(query_classes_batch, filter_class_ids))
                            selected_queries_batch = [query_classes_batch_filtered[i].nonzero().flatten().tolist()
                                                      for i in range(query_classes_batch_filtered.shape[0])]

                            allocated_queries = torch.nonzero(~outputs['query_mask'], as_tuple=True)
                            for sq_i, selected_queries in enumerate(selected_queries_batch):
                                sample_allocated_queries = allocated_queries[1][allocated_queries[0] == sq_i]
                                selected_queries_batch[sq_i] = [query for query in selected_queries
                                                                if query in sample_allocated_queries]
                            selected_queries_batch = [[elem for elem in sublist1 if elem not in sublist2]
                                                      for sublist1, sublist2 in zip(selected_queries_batch,
                                                                                    node_perma_finished_branches_batch)]
                            for sq_i, selected_queries in enumerate(selected_queries_batch):
                                if len(selected_queries) == 0 and not finish_flag[sq_i]:
                                    finish_flag[sq_i] = True
                                if finish_flag[sq_i]:
                                    curr_step_batch[sq_i] = []
                                    continue

                                indices = [prev_step_info['indices'][sq_i]]
                                next_step = next_step_batch[sq_i]
                                curr_step = curr_step_batch[sq_i]
                                curr_node_root_pos = curr_node_root_pos_batch[sq_i]
                                node_finished_branches = node_finished_branches_batch[sq_i]
                                finished_branches_end_nodes = finished_branches_end_nodes_batch[sq_i]
                                node_perma_finished_branches = node_perma_finished_branches_batch[sq_i]
                                query_radii = query_radii_batch[sq_i:sq_i + 1]
                                query_positions = query_positions_batch[sq_i:sq_i + 1]
                                query_classes = query_classes_batch[sq_i:sq_i + 1]
                                previous_selected_queries = [self.get_node(node_i, pred_tree)['query_index']
                                                             for node_i in curr_step]

                                if previous_selected_queries == [-1]:
                                    if len(selected_queries) == 1:
                                        point_hidden_state = prev_step_info['hs_without_norm'][sq_i, selected_queries[0], :]
                                        self.add_cont_branch_point_beg(selected_queries, curr_step, pred_tree, curr_node_root_pos,
                                                                       query_positions, query_radii, level, step, next_step,
                                                                       node_perma_finished_branches, query_classes, perma_end_classes,
                                                                       point_hidden_state)
                                    else:
                                        point_hidden_state = prev_step_info['hs_without_norm'][sq_i, :, :]
                                        self.add_new_bifur_branches_points_beg(global_branch_id, selected_queries, curr_step, pred_tree,
                                                                               query_positions, query_radii, curr_node_root_pos, level, step,
                                                                               next_step, node_finished_branches, finished_branches_end_nodes,
                                                                               node_perma_finished_branches, perma_end_classes, query_classes, point_hidden_state)

                                else:
                                    continuing_branches, new_branches = self.init_req_lists(indices, pred_tree,
                                                                                            curr_step, selected_queries, node_finished_branches,
                                                                                            finished_branches_end_nodes)
                                    point_hidden_state = prev_step_info['hs_without_norm'][sq_i, :, :]

                                    self.add_cont_branches_points(continuing_branches, previous_selected_queries, curr_step,
                                                                  pred_tree, curr_node_root_pos, query_positions, query_radii, level, step,
                                                                  next_step, node_perma_finished_branches, query_classes,
                                                                  perma_end_classes, selected_queries, point_hidden_state)

                                    if prev_step_info['bifur_list'] and len(prev_step_info['bifur_list'][sq_i]):
                                        bifur_dict = prev_step_info['bifur_list'][sq_i]
                                        self.add_new_bifur_branches_points(bifur_dict, pred_tree, new_branches, curr_step,
                                                                           finished_branches_end_nodes, global_branch_id, query_positions,
                                                                           query_radii, curr_node_root_pos, level, step,
                                                                           next_step, node_perma_finished_branches, perma_end_classes,
                                                                           query_classes, selected_queries, point_hidden_state)

                                curr_step = next_step
                                indices = self.get_updated_indices_nx(indices, curr_step, pred_tree)
                                prev_step_info['indices'][sq_i] = indices[0]
                                curr_step_batch[sq_i] = curr_step

                        next_step = [element for sublist in curr_step_batch for element in sublist]
                        next_level += next_step

                    for node in next_level:
                        node_info = self.get_node(node, pred_tree)
                        node_pos = node_info['position']
                        node_info['query_index'] = -1
                        node_info['rel_pos'] = np.array(node_pos) - np.array(node_pos).astype(int)

                    curr_level = next_level
                    if len(curr_level) == 0:
                        finished = True
                    level += 1

                all_preds.append(pred_tree)
                all_targets.append(targets[0]['networkx'][tree_id])
                et = time.time()
                elapsed_time.append(et - st)
                all_samples_ids.append(targets[0]['index'])
            if self.args.mask:
                all_masks.append(masks)
            all_samples.append(samples)

        return all_preds, all_targets, all_samples_ids, all_samples, all_masks, elapsed_time

    @torch.no_grad()
    def evaluate_sinsam(self, model, sample_id):
        model.eval()
        crop_pad = CropAndPad(self.args.sub_vol_size, self.args.zoom_levels, 'area')

        classes_to_filter = ['background', 'pad'] if self.args.pad_class else ['background']
        perma_end_classes = [self.args.class_dict['bifurcation'], self.args.class_dict['end']]

        annot_dir = os.path.join(self.args.data_dir, 'annots_test')
        mask_dir = os.path.join(self.args.data_dir, 'masks_test')
        img_dir = os.path.join(self.args.data_dir, 'images_test')
        img_path = os.path.join(img_dir, sample_id + '.nii.gz')
        mask_path = os.path.join(mask_dir, sample_id + '.nii.gz')
        annot_path = os.path.join(annot_dir, sample_id + '.pickle')

        # load, window and normalize the image
        image_reader = LoadImage(image_only=True)
        samples = image_reader(img_path)
        if self.args.window_input:
            window_transform = Compose([ThresholdIntensity(threshold=self.args.window_max,
                                                           above=False, cval=self.args.window_max),
                                        ThresholdIntensity(threshold=self.args.window_min,
                                                           above=True, cval=self.args.window_min)])
            samples = window_transform(samples)
        norm = NormalizeIntensity()
        samples = norm(samples)
        samples = samples.unsqueeze(0).unsqueeze(0).to(self.args.device)
        samples_min = torch.min(samples).to(self.args.device)

        # load the target and mask
        annot_reader = LoadAnnotPickle()
        targets = [annot_reader(annot_path)]
        targets[0]['index'] = sample_id
        masks = image_reader(mask_path).unsqueeze(0).unsqueeze(0)

        if self.args.eval_only:
            self.logger.info(f'Sample: {str(targets[0]["index"])}')

        for tree_id in range(len(targets[0]['networkx'])):
            st = time.time()
            self.log_progress(tree_id)
            finished = False
            samples = samples.to(self.args.device)
            global_branch_id = [0]
            level = 0
            target_tree = targets[0]['networkx'][tree_id]
            pred_tree = nx.DiGraph()
            root_node = self.get_global_pred_root_node_nx(pred_tree, target_tree)
            curr_level = [root_node['id']]
            while not finished:
                if self.check_finished(targets, tree_id, curr_level, level):
                    break
                if self.args.eval_only:
                    self.log_progress_level(curr_level, level)

                next_level = []
                for level_node_i in range(0, len(curr_level), self.args.batch_size_per_sample):
                    node_batch = curr_level[level_node_i: level_node_i + self.args.batch_size_per_sample]
                    sub_vol_batch, past_traj_pos_batch = self.get_sub_vol_and_past_tr(node_batch, samples,
                                                                                      pred_tree, samples_min, crop_pad)
                    if not len(sub_vol_batch):
                        continue

                    prev_step_info = {"first_step": True,
                                      "out": None,
                                      "bifur_list": None,
                                      "starting_node_id": [int(node.split("-")[0]) for node in node_batch],
                                      "global_id": global_branch_id,
                                      "node_type": "selected_node",
                                      "hidden_state_list": []}

                    if node_batch[0] != root_node['id']:
                        prev_level_hs = torch.zeros([len(node_batch), 1, self.args.hidden_dim], device=self.device)
                        for curr_node_idx, curr_node in enumerate(node_batch):
                            curr_node_info = self.get_node(curr_node, pred_tree)
                            prev_level_hs[curr_node_idx, 0, :] = curr_node_info['hidden_state']
                        prev_step_info['hs_without_norm'] = prev_level_hs
                        prev_step_info['node_type'] = 'pair_node'
                        prev_step_info['labels'] = [curr_node_info['label'] for curr_node_info in
                                                    [self.get_node(curr_node, pred_tree) for curr_node in node_batch]]
                        query_indices_batch = [torch.tensor([0]) for _ in node_batch]
                        query_targets_batch = [torch.tensor([int(curr_node.split('-')[0])]) for curr_node in node_batch]
                        prev_step_info['indices'] = [list(t) for t in zip(query_indices_batch, query_targets_batch)]
                    else:
                        prev_step_info['indices'] = [[] for _ in range(len(node_batch))]

                    finish_flag = [False for i in range(len(node_batch))]
                    sub_vol = torch.cat(sub_vol_batch, dim=0)
                    past_traj_pos = torch.cat(past_traj_pos_batch, dim=0)
                    curr_step_batch = [[node] for node in node_batch]
                    curr_node_root_pos_batch = [torch.tensor(self.get_node(node, pred_tree)['position'],
                                                             dtype=int).to(self.args.device) for node in node_batch]
                    node_finished_branches_batch = [[] for i in range(len(node_batch))]
                    finished_branches_end_nodes_batch = [[] for i in range(len(node_batch))]
                    node_perma_finished_branches_batch = [[] for i in range(len(node_batch))]

                    for step in range(1, self.args.seq_len):
                        num_all_nodes = sum([len(sublist) for sublist in curr_step_batch])
                        if not num_all_nodes:
                            break
                        next_step_batch = [[] for i in range(len(node_batch))]

                        norm_step = torch.tensor((step - 1) / (self.args.seq_len - 1)).unsqueeze(0).unsqueeze(0).to(self.args.device)
                        outputs, prev_step_info = model(sub_vol, past_traj_pos, prev_step_info, norm_step)
                        self.map_old_to_new_indices(pred_tree,
                                                    curr_step_batch,
                                                    node_finished_branches_batch,
                                                    finished_branches_end_nodes_batch,
                                                    node_perma_finished_branches_batch,
                                                    prev_step_info)

                        query_classes_batch, query_radii_batch, query_positions_batch = self.get_classes_radii_positions(outputs)
                        filter_class_ids = torch.tensor([self.args.class_dict[key]
                                                         for key in classes_to_filter]).to(query_classes_batch.device)
                        query_classes_batch_filtered = torch.logical_not(torch.isin(query_classes_batch, filter_class_ids))
                        selected_queries_batch = [query_classes_batch_filtered[i].nonzero().flatten().tolist()
                                                  for i in range(query_classes_batch_filtered.shape[0])]

                        allocated_queries = torch.nonzero(~outputs['query_mask'], as_tuple=True)
                        for sq_i, selected_queries in enumerate(selected_queries_batch):
                            sample_allocated_queries = allocated_queries[1][allocated_queries[0] == sq_i]
                            selected_queries_batch[sq_i] = [query for query in selected_queries
                                                            if query in sample_allocated_queries]
                        selected_queries_batch = [[elem for elem in sublist1 if elem not in sublist2]
                                                  for sublist1, sublist2 in zip(selected_queries_batch,
                                                                                node_perma_finished_branches_batch)]
                        for sq_i, selected_queries in enumerate(selected_queries_batch):
                            if len(selected_queries) == 0 and not finish_flag[sq_i]:
                                finish_flag[sq_i] = True
                            if finish_flag[sq_i]:
                                curr_step_batch[sq_i] = []
                                continue

                            indices = [prev_step_info['indices'][sq_i]]
                            next_step = next_step_batch[sq_i]
                            curr_step = curr_step_batch[sq_i]
                            curr_node_root_pos = curr_node_root_pos_batch[sq_i]
                            node_finished_branches = node_finished_branches_batch[sq_i]
                            finished_branches_end_nodes = finished_branches_end_nodes_batch[sq_i]
                            node_perma_finished_branches = node_perma_finished_branches_batch[sq_i]
                            query_radii = query_radii_batch[sq_i:sq_i + 1]
                            query_positions = query_positions_batch[sq_i:sq_i + 1]
                            query_classes = query_classes_batch[sq_i:sq_i + 1]
                            previous_selected_queries = [self.get_node(node_i, pred_tree)['query_index']
                                                         for node_i in curr_step]

                            if previous_selected_queries == [-1]:
                                if len(selected_queries) == 1:
                                    point_hidden_state = prev_step_info['hs_without_norm'][sq_i, selected_queries[0], :]
                                    self.add_cont_branch_point_beg(selected_queries, curr_step, pred_tree, curr_node_root_pos,
                                                                   query_positions, query_radii, level, step, next_step,
                                                                   node_perma_finished_branches, query_classes, perma_end_classes,
                                                                   point_hidden_state)
                                else:
                                    point_hidden_state = prev_step_info['hs_without_norm'][sq_i, :, :]
                                    self.add_new_bifur_branches_points_beg(global_branch_id, selected_queries, curr_step, pred_tree,
                                                                           query_positions, query_radii, curr_node_root_pos, level, step,
                                                                           next_step, node_finished_branches, finished_branches_end_nodes,
                                                                           node_perma_finished_branches, perma_end_classes, query_classes, point_hidden_state)
                            else:
                                continuing_branches, new_branches = self.init_req_lists(indices, pred_tree,
                                                                                        curr_step, selected_queries, node_finished_branches,
                                                                                        finished_branches_end_nodes)
                                point_hidden_state = prev_step_info['hs_without_norm'][sq_i, :, :]

                                self.add_cont_branches_points(continuing_branches, previous_selected_queries, curr_step,
                                                              pred_tree, curr_node_root_pos, query_positions, query_radii, level, step,
                                                              next_step, node_perma_finished_branches, query_classes, perma_end_classes,
                                                              selected_queries, point_hidden_state)

                                if prev_step_info['bifur_list'] and len(prev_step_info['bifur_list'][sq_i]):
                                    bifur_dict = prev_step_info['bifur_list'][sq_i]
                                    self.add_new_bifur_branches_points(bifur_dict, pred_tree, new_branches, curr_step,
                                                                       finished_branches_end_nodes, global_branch_id, query_positions,
                                                                       query_radii, curr_node_root_pos, level, step,
                                                                       next_step, node_perma_finished_branches, perma_end_classes,
                                                                       query_classes, point_hidden_state)

                            curr_step = next_step
                            indices = self.get_updated_indices_nx(indices, curr_step, pred_tree)
                            prev_step_info['indices'][sq_i] = indices[0]
                            curr_step_batch[sq_i] = curr_step

                    next_step = [element for sublist in curr_step_batch for element in sublist]
                    next_level += next_step

                for node in next_level:
                    node_info = self.get_node(node, pred_tree)
                    node_pos = node_info['position']
                    node_info['query_index'] = -1
                    node_info['rel_pos'] = np.array(node_pos) - np.array(node_pos).astype(int)

                curr_level = next_level
                if len(curr_level) == 0:
                    finished = True
                level += 1

        return [pred_tree], [targets[0]['networkx'][tree_id]], [targets[0]['index']], [samples], [masks], [time.time() - st]

    @torch.no_grad()
    def evaluate_sv(self, model, data_loader):
        model.eval()
        all_masks = []
        all_samples = []
        all_samples_ids = []
        all_preds = []
        all_targets = []
        elapsed_time = []
        classes_to_filter = ['background', 'pad'] if self.args.pad_class else ['background']
        level = 0
        perma_end_classes = [self.args.class_dict['bifurcation'], self.args.class_dict['end']]

        for i, batch in tqdm(enumerate(data_loader), desc="Batch", leave=False, total=len(data_loader)):
            start = time.time()
            sample_imgs, sample_past_trs, targets = (batch["image"], batch["past_tr"], batch["label"])
            if self.args.mask:
                masks = batch["mask"]
            if sample_imgs.device != self.args.device:
                sample_imgs = sample_imgs.to(self.args.device)
            if sample_past_trs is not None:
                sample_past_trs = sample_past_trs.to(self.args.device)

            pred_tree_batch = self.create_node_batch_nx_sv(targets)
            node_batch = ['0-0' for _ in range(len(pred_tree_batch))]
            global_branch_id = [0]
            prev_step_info = {"first_step": True,
                              "indices": [[] for _ in range(len(node_batch))],
                              "out": None,
                              "bifur_list": None,
                              "starting_node_id": [int(node.split("-")[0]) for node in node_batch],
                              "global_id": global_branch_id,
                              "node_type": "selected_node"}

            finish_flag = [False for i in range(len(node_batch))]
            sub_vol = sample_imgs
            past_traj_pos = sample_past_trs
            curr_step_batch = [[node] for node in node_batch]
            curr_node_root_pos_batch = [torch.tensor(self.get_node('0-0', pred_tree)['position'],
                                                     dtype=int).to(self.args.device) for pred_tree in pred_tree_batch]
            node_finished_branches_batch = [[] for i in range(len(node_batch))]
            finished_branches_end_nodes_batch = [[] for i in range(len(node_batch))]
            node_perma_finished_branches_batch = [[] for i in range(len(node_batch))]
            for s_i, step in enumerate(range(1, self.args.seq_len)):
                next_step_batch = [[] for i in range(len(node_batch))]
                norm_step = torch.tensor((step - 1) / (self.args.seq_len - 1)).unsqueeze(0).unsqueeze(0).to(self.args.device)
                outputs, prev_step_info = model(sub_vol, past_traj_pos, prev_step_info, norm_step)
                self.map_old_to_new_indices_sv(pred_tree_batch,
                                               curr_step_batch,
                                               node_finished_branches_batch,
                                               finished_branches_end_nodes_batch,
                                               node_perma_finished_branches_batch,
                                               prev_step_info)

                query_classes_batch, query_radii_batch, query_positions_batch = self.get_classes_radii_positions(outputs)
                filter_class_ids = torch.tensor([self.args.class_dict[key] for key in classes_to_filter]).to(query_classes_batch.device)
                query_classes_batch_filtered = torch.logical_not(torch.isin(query_classes_batch, filter_class_ids))
                selected_queries_batch = [query_classes_batch_filtered[i].nonzero().flatten().tolist()
                                          for i in range(query_classes_batch_filtered.shape[0])]

                allocated_queries = torch.nonzero(~outputs['query_mask'], as_tuple=True)
                for sq_i, selected_queries in enumerate(selected_queries_batch):
                    sample_allocated_queries = allocated_queries[1][allocated_queries[0] == sq_i]
                    selected_queries_batch[sq_i] = [query for query in selected_queries
                                                    if query in sample_allocated_queries]
                selected_queries_batch = [[elem for elem in sublist1 if elem not in sublist2]
                                          for sublist1, sublist2 in zip(selected_queries_batch,
                                                                        node_perma_finished_branches_batch)]

                for sq_i, selected_queries in enumerate(selected_queries_batch):
                    if len(selected_queries) == 0 and not finish_flag[sq_i]:
                        finish_flag[sq_i] = True
                    if finish_flag[sq_i]:
                        curr_step_batch[sq_i] = []
                        continue
                    pred_tree = pred_tree_batch[sq_i]
                    indices = [prev_step_info['indices'][sq_i]]
                    next_step = next_step_batch[sq_i]
                    curr_step = curr_step_batch[sq_i]
                    curr_node_root_pos = curr_node_root_pos_batch[sq_i]
                    node_finished_branches = node_finished_branches_batch[sq_i]
                    finished_branches_end_nodes = finished_branches_end_nodes_batch[sq_i]
                    node_perma_finished_branches = node_perma_finished_branches_batch[sq_i]
                    query_radii = query_radii_batch[sq_i:sq_i + 1]
                    query_positions = query_positions_batch[sq_i:sq_i + 1]
                    query_classes = query_classes_batch[sq_i:sq_i + 1]
                    previous_selected_queries = [self.get_node(node_i, pred_tree)['query_index']
                                                 for node_i in curr_step]

                    if previous_selected_queries == [-1]:
                        if len(selected_queries) == 1:
                            self.add_cont_branch_point_beg(selected_queries, curr_step, pred_tree, curr_node_root_pos,
                                                           query_positions, query_radii, level, step, next_step,
                                                           node_perma_finished_branches, query_classes, perma_end_classes)
                        else:
                            self.add_new_bifur_branches_points_beg(global_branch_id, selected_queries, curr_step, pred_tree,
                                                                   query_positions, query_radii, curr_node_root_pos, level, step,
                                                                   next_step, node_finished_branches, finished_branches_end_nodes,
                                                                   node_perma_finished_branches, perma_end_classes, query_classes)
                    else:
                        continuing_branches, new_branches = self.init_req_lists(indices, pred_tree,
                                                                                curr_step, selected_queries, node_finished_branches,
                                                                                finished_branches_end_nodes)
                        self.add_cont_branches_points(continuing_branches, previous_selected_queries, curr_step,
                                                      pred_tree, curr_node_root_pos, query_positions, query_radii, level, step,
                                                      next_step, node_perma_finished_branches, query_classes, perma_end_classes,
                                                      selected_queries)

                        if prev_step_info['bifur_list'] and len(prev_step_info['bifur_list'][sq_i]):
                            bifur_dict = prev_step_info['bifur_list'][sq_i]
                            self.add_new_bifur_branches_points(bifur_dict, pred_tree, new_branches, curr_step,
                                                               finished_branches_end_nodes, global_branch_id, query_positions,
                                                               query_radii, curr_node_root_pos, level, step,
                                                               None, next_step, node_perma_finished_branches,
                                                               perma_end_classes, query_classes, selected_queries)

                    curr_step = next_step
                    indices = self.get_updated_indices_nx(indices, curr_step, pred_tree)
                    prev_step_info['indices'][sq_i] = indices[0]
                    curr_step_batch[sq_i] = curr_step

            end = time.time()
            elp_time = (end - start)/len(targets)
            elapsed_time += [elp_time for _ in range(len(targets))]
            all_preds += pred_tree_batch
            for sam_i in range(len(targets)):
                all_targets.append(targets[sam_i]['selected_node'])
                all_samples.append(sample_imgs[sam_i].cpu().detach())
                all_samples_ids.append(targets[sam_i]['index'])
                if self.args.mask:
                    all_masks.append(masks[sam_i].cpu().detach())
        return all_preds, all_targets, all_samples_ids, all_samples, all_masks, elapsed_time