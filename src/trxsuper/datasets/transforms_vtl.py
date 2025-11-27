import pickle
import numpy as np
import torch
import re
import copy

from scipy.stats import laplace
from scipy.interpolate import CubicSpline, interp1d
import os
os.environ["BIGTREE_CONF_ASSERTIONS"] = ""

from bigtree import find_name, levelordergroup_iter, preorder_iter, find_full_path, Node
from monai.utils import convert_to_tensor, pytorch_after
from monai.transforms import (
    LoadImage,
    NormalizeIntensity,
    ThresholdIntensity,
    Compose,
    Transform,
    MapTransform,
    Randomizable,
    Crop,
    Pad)


class LoadAnnotPickle(Transform):
    def __init__(self):
        pass

    def __call__(self, input):
        with open(input, 'rb') as handle:
            data = pickle.load(handle)
        data['index'] = [int(s) for s in re.findall(r'\d+', input)][-1]
        return data


class LoadAnnotPickled(MapTransform):
    def __init__(self, keys,
                 allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.transform = LoadAnnotPickle()

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform(d[key])
        return d


class CropAndPad(Transform):
    def __init__(self, sub_vol_size):
        self.crop = Crop()
        self.pad = Pad()
        self.sub_vol_size = sub_vol_size

    @staticmethod
    def compute_slices(roi_center, roi_size, img_size):
        """
        Compute the crop slices based on specified `center & size` or `start & end` or `slices`.

        Args:
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI
            img_size: size of the whole image

        Outputs:
            slices: list of slices for each of the spatial dimensions.
            padding: list of padding for each of the spatial dimensions.
        """

        assert roi_center is not None and roi_size is not None and img_size is not None
        img_size_t = convert_to_tensor(data=img_size, dtype=torch.int16, wrap_sequence=True, device="cuda")
        roi_center_t = convert_to_tensor(data=roi_center, dtype=torch.int16, wrap_sequence=True, device="cuda")
        roi_size_t = convert_to_tensor(data=roi_size, dtype=torch.int16, wrap_sequence=True, device="cuda")
        _zeros = torch.zeros_like(roi_center_t)
        half = (
            torch.divide(roi_size_t, 2, rounding_mode="floor")
            if pytorch_after(1, 8)
            else torch.floor_divide(roi_size_t, 2)
        )
        roi_start_t = roi_center_t - half
        roi_end_t = roi_start_t + roi_size_t
        roi_start_clipped_t = torch.maximum(roi_start_t, _zeros)
        roi_end_clipped_t = torch.minimum(roi_end_t, img_size_t)
        roi_start_pad_t = torch.abs(torch.minimum(roi_start_t, _zeros))
        roi_end_pad_t = torch.maximum(roi_end_t - img_size_t, _zeros)

        slices = [slice(int(s), int(e)) for s, e in zip(roi_start_clipped_t.tolist(), roi_end_clipped_t.tolist())]
        padding = [(0, 0)] + [(int(s), int(e)) for s, e in zip(roi_start_pad_t.tolist(), roi_end_pad_t.tolist())]

        return slices, padding

    def extract_image_crop(self, image_data, node_position, image_min):
        crop_slices, crop_padding = self.compute_slices(node_position,
                                                        self.sub_vol_size,
                                                        image_data.shape[-3:])
        cropped_image = self.crop(image_data, crop_slices)
        cropped_image = self.pad(cropped_image, crop_padding, value=image_min)

        return cropped_image

    def __call__(self, data, position, image_min):
        # extract image sub volume around root point
        data = self.extract_image_crop(data, position, image_min)

        return data


class CropAndPadd(MapTransform):
    def __init__(self, keys, sub_vol_size):
        super().__init__(keys)
        self.transform = CropAndPad(sub_vol_size)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            images = []
            for target in d['label']:
                if key == 'image':
                    images.append(self.transform(d[key], target['root_position'], d['image_min'].item()))
                elif key == 'mask':
                    images.append(self.transform(d[key], target['root_position'], 0.0))
                else:
                    raise NotImplementedError
            d[key] = images

        return d


class ComputeImageRanged(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d['image_min'] = torch.min(d[key])
            d['image_max'] = torch.max(d[key])
        return d


class ExtRandomSubTree(Randomizable, Transform):
    def __init__(self, root_prob, bifur_prob, end_prob, seq_len, num_prev_pos,
                 traj_train_len):
        self.root_prob = root_prob
        self.bifur_prob = bifur_prob
        self.end_prob = end_prob
        self.seq_len = seq_len
        self.num_prev_pos = num_prev_pos
        self.traj_train_len = traj_train_len
        self.traj_train_num_pts = (traj_train_len * (seq_len - 1)) + 1

    def randomize(self, data):
        """
        Pick a random point using curated sampling
        """
        num_trees = len(data['branches'])
        tree_id = self.R.randint(low=0, high=num_trees, size=1)[0]
        full_tree = data['branches'][tree_id]

        num_traj = len(data['trajectories'][tree_id])
        traj_id = self.R.randint(low=0, high=num_traj, size=1)[0]
        selected_trajectory = data['trajectories'][tree_id][traj_id]
        path = selected_trajectory['path']
        bifur_ids = selected_trajectory['bifur_ids']
        endpt_id = selected_trajectory['endpt_id']
        endpt_index = path.index(endpt_id)
        prob = self.R.random()

        if prob < self.bifur_prob:
            bifur_indices = [path.index(bifur) for bifur in bifur_ids]
            candidates = set()
            for bifur_index in bifur_indices:
                start = max(0, bifur_index - self.traj_train_num_pts + 1)
                end = bifur_index + 1
                candidates.update(range(start, end))
            max_valid_index = endpt_index - ((self.traj_train_len - 1) * (self.seq_len - 1))
            valid_candidates = [idx for idx in candidates if idx <= max_valid_index]
            if len(valid_candidates) == 0:
                return self.randomize(data)
            point_id = path[self.R.choice(valid_candidates)]
            point_type = 'bifurcation'

        elif prob < (self.bifur_prob + self.end_prob):
            max_valid_index = endpt_index - ((self.traj_train_len - 1) * (self.seq_len - 1))
            min_valid_index = max(0, endpt_index - self.traj_train_num_pts + 1)
            valid_candidates = list(range(min_valid_index, max_valid_index + 1))
            if len(valid_candidates) == 0:
                return self.randomize(data)
            point_id = path[self.R.choice(valid_candidates)]
            point_type = 'end'

        elif prob < (self.bifur_prob + self.end_prob + self.root_prob):
            min_valid_index = 0
            max_valid_index = min(self.seq_len - 1, endpt_index - ((self.traj_train_len - 1) * (self.seq_len - 1)))
            valid_candidates = list(range(min_valid_index, max_valid_index + 1))
            if len(valid_candidates) == 0:
                return self.randomize(data)
            point_id = path[self.R.choice(valid_candidates)]
            point_type = 'root'

        else:
            endpt_index = path.index(endpt_id)
            min_valid_index = 0
            max_valid_index = endpt_index - ((self.traj_train_len - 1) * (self.seq_len - 1))
            valid_candidates = list(range(min_valid_index, max_valid_index + 1))
            if len(valid_candidates) == 0:
                return self.randomize(data)
            point_id = path[self.R.choice(valid_candidates)]
            point_type = 'random'

        assert point_id is not None, "Selected node not found!"
        selected_node = find_name(full_tree, point_id)

        output = {'tree_id': tree_id,
                  'selected_path': path,
                  'selected_node': selected_node,
                  'point_id': point_id,
                  'point_type': point_type}

        return output

    @staticmethod
    def select_nearby_point(full_tree, point_id, dist):
        selected_node = find_name(full_tree, point_id)
        for _ in range(dist):
            parent = selected_node.parent
            if parent is not None:
                selected_node = parent
            else:
                continue

        return selected_node

    @staticmethod
    def get_past_traj_start(selected_node, num_prev_pos):
        past_traj_head_node = selected_node
        for past_nodes_num in range(num_prev_pos):
            if past_traj_head_node.parent is not None:
                past_traj_head_node = past_traj_head_node.parent
            else:
                break

        return past_traj_head_node, past_nodes_num

    @staticmethod
    def create_subtree_with_past_tr_val(selected_node, past_traj_num, seq_len):
        root_node = Node(selected_node.name,
                         position=selected_node.position,
                         radius=selected_node.radius,
                         label=selected_node.label)

        for i, level in enumerate(levelordergroup_iter(selected_node)):
            if i == 0:
                continue
            elif i <= seq_len:
                for node in level:
                    parent_node = find_name(root_node, node.parent.name)
                    Node(node.name,
                         position=node.position,
                         radius=node.radius,
                         parent=parent_node,
                         label=node.label)
            else:
                break

        if selected_node.parent is not None:
            current_node = selected_node.parent
            past_tr_end_node = Node(current_node.name,
                                    position=current_node.position,
                                    radius=current_node.radius,
                                    label=current_node.label)

            past_tr_head_node = past_tr_end_node
            for _ in range(1, past_traj_num):
                current_node = current_node.parent
                past_tr_head_node.parent = Node(current_node.name,
                                                position=current_node.position,
                                                radius=current_node.radius,
                                                label=current_node.label)
                past_tr_head_node = past_tr_head_node.parent
            root_node.parent = past_tr_end_node
        else:
            past_tr_head_node = root_node

        return root_node, past_tr_head_node

    def create_subtree_with_past_tr(self, selected_node, selected_path, max_root_buffer_nodes):
        selected_node_index = selected_path.index(selected_node.name)
        past_traj_start_index = max(selected_node_index - (self.num_prev_pos - 1), 0)
        num_prev_pos = selected_node_index - past_traj_start_index + 1
        buffer_start_index = max(past_traj_start_index - max_root_buffer_nodes, 0)
        num_root_buffer_nodes = past_traj_start_index - buffer_start_index
        buffer_start_node = find_name(selected_node.root, selected_path[buffer_start_index])

        root_node = Node(buffer_start_node.name,
                         position=buffer_start_node.position,
                         radius=buffer_start_node.radius,
                         label=buffer_start_node.label)

        new_level = None
        prev_level = None
        path_end_level = None
        end_nodes = []
        max_levels = self.traj_train_num_pts + num_prev_pos + num_root_buffer_nodes
        for i, level in enumerate(levelordergroup_iter(buffer_start_node)):
            if i == 0:
                continue

            if i == num_root_buffer_nodes + 1:
                if new_level:
                    past_tr_head_node = [node for node in new_level if node.name in selected_path][0]
                else:
                    past_tr_head_node = root_node

            if i == num_root_buffer_nodes + num_prev_pos:
                if new_level:
                    selected_node = [node for node in new_level if node.name in selected_path][0]
                else:
                    selected_node = root_node

            if i == max_levels:
                break

            if i == max_levels - 1:
                path_end_level = prev_level

            new_level = []
            for node in level:
                parent_node = find_name(root_node, node.parent.name)
                new_level.append(Node(node.name,
                     position=node.position,
                     radius=node.radius,
                     parent=parent_node,
                     label=node.label))

                if len(node.children) == 0:
                    end_nodes.append(node)

            prev_level = list(level)

        if path_end_level is None:
            path_end_level = prev_level

        selected_node_index = selected_path.index(selected_node.name)
        path_end_node_name = [node.name for node in path_end_level + end_nodes
                              if node.name in selected_path][0]
        path_end_node_index = selected_path.index(path_end_node_name)
        selected_path = selected_path[selected_node_index:path_end_node_index + 1]

        return selected_node, past_tr_head_node, selected_path, num_prev_pos

    def __call__(self, data):
        output = self.randomize(data)
        max_root_buffer_nodes = 10
        (selected_node, past_traj_head_node,
         selected_path, num_prev_pos) = self.create_subtree_with_past_tr(
            output['selected_node'], output['selected_path'], max_root_buffer_nodes)

        data = {'index': data['index'],
                'tree_id': output['tree_id'],
                'point_id': output['point_id'],
                'point_type': output['point_type'],
                'selected_path': selected_path,
                'selected_node': selected_node,
                'past_traj_head_node': past_traj_head_node,
                'num_prev_pos': num_prev_pos,
                }

        return data


class ExtRandomSubTreed(Randomizable, MapTransform):
    def __init__(self, keys, root_prob, bifur_prob, end_prob, seq_len, num_prev_pos,
                 traj_train_len):
        super(Randomizable, self).__init__(keys)
        self.transform = ExtRandomSubTree(root_prob, bifur_prob, end_prob,
                                          seq_len, num_prev_pos, traj_train_len)

    def set_random_state(self, seed=None, state=None):
        self.transform.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform(d[key])

        return d


class DivideSubTree(Transform):
    def __init__(self, seq_len, num_prev_pos, traj_train_len):
        self.seq_len = seq_len
        self.num_prev_pos = num_prev_pos
        self.traj_train_len = traj_train_len
        self.traj_train_num_pts = (traj_train_len * (seq_len - 1)) + 1

    def get_starting_points(self, data):
        selected_node_list = [data['selected_node']]
        point_index = data['selected_path'].index(data['selected_node'].node_name)

        for i in range(1, self.traj_train_len):
            next_point_id = data['selected_path'][point_index + (self.seq_len - 1) * i]
            next_node = find_name(data['selected_node'], next_point_id)
            assert next_node is not None, "Selected node not found!"
            selected_node_list.append(next_node)

        return selected_node_list

    @staticmethod
    def get_past_traj_start(selected_node, num_prev_pos):
        past_traj_head_node = selected_node
        for past_nodes_num in range(num_prev_pos):
            if past_traj_head_node.parent is not None:
                past_traj_head_node = past_traj_head_node.parent
            else:
                break

        return past_traj_head_node, past_nodes_num

    @staticmethod
    def create_subtree_with_past_tr(selected_node, past_traj_num, seq_len):
        root_node = Node(selected_node.name,
                         position=selected_node.position,
                         radius=selected_node.radius,
                         label=selected_node.label)

        for i, level in enumerate(levelordergroup_iter(selected_node)):
            if i == 0:
                continue
            elif i <= seq_len:
                for node in level:
                    parent_node = find_name(root_node, node.parent.name)
                    Node(node.name,
                         position=node.position,
                         radius=node.radius,
                         parent=parent_node,
                         label=node.label)
            else:
                break

        if selected_node.parent is not None:
            current_node = selected_node.parent
            past_tr_end_node = Node(current_node.name,
                                    position=current_node.position,
                                    radius=current_node.radius,
                                    label=current_node.label)
            past_tr_head_node = past_tr_end_node

            for i in range(1, past_traj_num):
                current_node = current_node.parent
                past_tr_head_node.parent = Node(current_node.name,
                                                position=current_node.position,
                                                radius=current_node.radius,
                                                label=current_node.label)
                past_tr_head_node = past_tr_head_node.parent

            root_node.parent = past_tr_end_node
        else:
            past_tr_head_node = root_node

        return root_node, past_tr_head_node

    def __call__(self, data):
        selected_node_list = self.get_starting_points(data)
        selected_nodes, past_traj_head_nodes = [], []
        for selected_node_orig in selected_node_list:
            _, past_traj_num = self.get_past_traj_start(selected_node_orig, self.num_prev_pos)
            selected_node, past_traj_head_node = self.create_subtree_with_past_tr(selected_node_orig,
                                                                                  past_traj_num, self.seq_len)
            selected_nodes.append(selected_node)
            past_traj_head_nodes.append(past_traj_head_node)

        data['selected_node'] = selected_nodes
        data['past_traj_head_node'] = past_traj_head_nodes

        return data


class DivideSubTreed(MapTransform):
    def __init__(self, keys, seq_len, num_prev_pos, traj_train_len):
        super().__init__(keys)
        self.transform = DivideSubTree(seq_len, num_prev_pos, traj_train_len)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform(d[key])

        return d


class AddRandMicroTrajNoise(Randomizable, Transform):
    def __init__(self):
        pass

    def randomize(self, num_nodes):
        offsets = self.R.normal(loc=0, scale=0.025, size=(num_nodes, 3))

        return offsets

    @staticmethod
    def update_tree_positions(data, all_node_paths, offsets):
        for i, path in enumerate(all_node_paths):
            if sum(offsets[i]):
                path_node = find_full_path(data['past_traj_head_node'].root, path)
                new_position = np.array(path_node.position) + offsets[i]
                path_node.position = new_position.tolist()

    def __call__(self, data):
        all_node_paths = [node.path_name for node in preorder_iter(data['past_traj_head_node'].root)]
        offsets = self.randomize(len(all_node_paths))
        self.update_tree_positions(data, all_node_paths, offsets)

        return data


class AddRandMicroTrajNoised(Randomizable, MapTransform):
    def __init__(self, keys):
        super(Randomizable, self).__init__(keys)
        self.transform = AddRandMicroTrajNoise()

    def set_random_state(self, seed=None, state=None):
        self.transform.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform(d[key])

        return d


class AddRandBifurTrajNoise(Randomizable, Transform):
    def __init__(self, num_prev_pos, seq_len, traj_train_len):
        self.noise_prob = 0.5
        self.num_prev_pos = num_prev_pos
        self.min_num_pts = ((traj_train_len - 1) * (seq_len - 1)) + 1
        self.std_res = 0.25
        self.std_devs = np.linspace(0.5, 4, int(round(20 / 0.25)))

    def randomize(self, bif_node, data):
        path = data['selected_path']
        selected_node = data['selected_node']
        prob = self.R.random()
        if prob < self.noise_prob:
            radius = bif_node.radius
            index = int(radius / self.std_res)
            if index >= len(self.std_devs):
                index = len(self.std_devs) - 1
            corr_std = self.std_devs[index]

            bifur_node_index = path.index(bif_node.name)
            lower_bound = 0
            upper_bound = len(path) - 1

            start_node = find_name(bif_node.root, path[0])
            for i, level in enumerate(levelordergroup_iter(start_node)):
                if i == len(path):
                    break
                if i == bifur_node_index:
                    continue
                if len(level) > 1:
                    level = [node for node in level if node.name in path]
                node = level[0]
                if len(node.children) > 1 or node.name == selected_node.name:
                    if i < bifur_node_index:
                        lower_bound = i
                    else:
                        upper_bound = i
                        break

            lower_bound = lower_bound - bifur_node_index
            upper_bound = upper_bound - bifur_node_index
            lower_bound = max(lower_bound, -radius)
            upper_bound = min(upper_bound, radius)

            offsets = self.truncated_mixture_laplace(loc=0, min_scale=0.5,
                                                     max_scale=corr_std, components=10, a=lower_bound,
                                                     b=upper_bound, size=1, random_state=self.R)
            offsets = np.round(offsets).astype(np.int32)
        else:
            offsets = [0]

        return offsets[0]

    @staticmethod
    def truncated_mixture_laplace(loc=0, min_scale=0.1, max_scale=4, components=10,
                                  a=-10, b=10, size=1, random_state=None):
        """Generate samples from a truncated mixture of Laplace distributions."""
        scales = np.linspace(min_scale, max_scale, components)
        weights = np.ones(components) / components
        samples = []

        for _ in range(size):
            component_idx = random_state.choice(len(scales), p=weights)
            scale = scales[component_idx]

            cdf_a = laplace.cdf(a, loc=loc, scale=scale)
            cdf_b = laplace.cdf(b, loc=loc, scale=scale)

            uniform_sample = random_state.uniform(cdf_a, cdf_b)
            truncated_sample = laplace.ppf(uniform_sample, loc=loc, scale=scale)

            samples.append(truncated_sample)

        return np.array(samples)

    @staticmethod
    def linear_interpolate(points, radii, start_idx, end_idx, step):
        """
        Linearly interpolates 3D points and radii using a fixed step distance.

        Parameters:
            points (list or np.array): List of 3D points [[x1, y1, z1], [x2, y2, z2], ...].
            radii (list or np.array): Corresponding radii for each point.
            step (float): Desired distance between interpolated points.

        Returns:
            interp_points (np.array): Interpolated 3D points.
            interp_radii (np.array): Interpolated radii.
        """
        points = np.array(points)
        radii = np.array(radii)
        distances = np.zeros(len(points))
        distances[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))

        interp_x = interp1d(distances, points[:, 0], kind='linear')
        interp_y = interp1d(distances, points[:, 1], kind='linear')
        interp_z = interp1d(distances, points[:, 2], kind='linear')
        interp_r = interp1d(distances, radii, kind='linear')

        start_dist = distances[start_idx]
        end_dist = distances[end_idx]
        total_length = end_dist - start_dist

        if step is not None:
            n_points = max(round(total_length / step) + 1, 3)
        assert n_points is not None, "Either n_points or points_dist should be provided"
        new_distances = np.linspace(start_dist, end_dist, n_points)
        new_points = np.vstack((interp_x(new_distances), interp_y(new_distances), interp_z(new_distances))).T
        new_radii = interp_r(new_distances)

        return new_points, new_radii

    def smooth_interpolate(self, points, radii, start_idx, end_idx, step=1.0, max_dist=1.5):
        points = np.array(points)
        radii = np.array(radii)
        n_points = len(points)

        distances = np.zeros(n_points)
        distances[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))

        spline_x = CubicSpline(distances, points[:, 0])
        spline_y = CubicSpline(distances, points[:, 1])
        spline_z = CubicSpline(distances, points[:, 2])
        spline_r = CubicSpline(distances, radii)

        start_dist = distances[start_idx]
        end_dist = distances[end_idx]
        total_length = end_dist - start_dist

        if step is not None:
            n_points = max(round(total_length / step) + 1, 3)
        assert n_points is not None, "Either n_points or points_dist should be provided"
        new_distances = np.linspace(start_dist, end_dist, n_points)

        in_between_points = np.vstack((spline_x(new_distances), spline_y(new_distances), spline_z(new_distances))).T
        in_between_radii = spline_r(new_distances)
        num_points_diff = len(in_between_points) - 2
        dists = np.linalg.norm(np.diff(in_between_points, axis=0), axis=1)
        if any(dists > max_dist) or any(dists < 0.5):
            new_points = np.vstack((points[:start_idx], in_between_points, points[end_idx + 1:]))
            new_radii = np.hstack((radii[:start_idx], in_between_radii, radii[end_idx + 1:]))
            in_between_points, in_between_radii = self.linear_interpolate(new_points,
                                                                          new_radii, start_idx, end_idx + num_points_diff, 1.0)

        return in_between_points, in_between_radii

    def compute_inbetween_points(self, source_node, target_node, positions_before,
                                 positions_after, radii_before, radii_after, max_dist):
        positions_before = positions_before[source_node.name][::-1]
        radii_before = radii_before[source_node.name][::-1]
        positions_after = positions_after[target_node.name]
        radii_after = radii_after[target_node.name]
        positions = positions_before + positions_after
        radii = radii_before + radii_after
        start_idx = len(positions_before) - 1
        end_idx = len(positions) - len(positions_after)

        if len(positions_before) < 2 or len(positions_after) < 2:
            in_between_positions, in_between_radii = self.linear_interpolate(positions, radii, start_idx, end_idx, 1.0)
        else:
            in_between_positions, in_between_radii = self.smooth_interpolate(positions, radii, start_idx, end_idx,
                                                                             1.0, max_dist)
        in_between_positions, in_between_radii = in_between_positions[1:-1], in_between_radii[1:-1]

        return in_between_positions, in_between_radii

    @staticmethod
    def get_positions_and_radii_before(node, num_neighbors=3):
        positions_before = []
        radii_before = []
        parent = node
        for i in range(num_neighbors):
            positions_before.append(parent.position)
            radii_before.append(parent.radius)
            if parent.parent is None:
                break
            parent = parent.parent
        return positions_before, radii_before

    @staticmethod
    def get_positions_and_radii_after(node, num_neighbors=3):
        positions_after = []
        radii_after = []
        child = node
        for i in range(num_neighbors):
            positions_after.append(child.position)
            radii_after.append(child.radius)
            if not len(child.children) or len(child.children) > 1:
                break
            child = child.children[0]
        return positions_after, radii_after

    def offset_bifur_node(self, data, bif_node, offset):
        for i, level in enumerate(levelordergroup_iter(bif_node)):
            if i == 1:
                break

        path = data['selected_path']
        nodes_to_move = [node for node in level if node.name not in path]
        main_traj_node = [node for node in level if node.name in path][0]

        new_bif_node_name = path[path.index(bif_node.name) + offset]
        new_bif_node = find_name(bif_node.root, new_bif_node_name)

        positions_before, positions_after, radii_before, radii_after = {}, {}, {}, {}
        for mn_i, move_node in enumerate(nodes_to_move):
            if abs(offset) > 1:
                updated_move_node = None
                for i, level in enumerate(levelordergroup_iter(move_node)):
                    if i < abs(offset) and len(level) == 1:
                        updated_move_node = level[0]
                    else:
                        break

                updated_move_node.parent = None
                if new_bif_node.name not in positions_before:
                    (positions_before[new_bif_node.name],
                     radii_before[new_bif_node.name]) = self.get_positions_and_radii_before(new_bif_node)
                (positions_after[updated_move_node.name],
                 radii_after[updated_move_node.name]) = self.get_positions_and_radii_after(updated_move_node)
                move_node.parent = None
                del move_node

                if mn_i == len(nodes_to_move) - 1:
                    if offset > 0:
                        main_traj_move_node = bif_node
                        prev_main_traj_move_node = None
                        for j in range(offset):
                            if (main_traj_move_node.parent and
                                    main_traj_move_node.name != data['selected_node'].name):
                                prev_main_traj_move_node = main_traj_move_node
                                main_traj_move_node = main_traj_move_node.parent
                            else:
                                break

                            if len(main_traj_move_node.children) > 1:
                                break

                        if prev_main_traj_move_node:
                            prev_main_traj_move_node.parent = None
                        new_bif_node.parent = None

                        (positions_before[main_traj_move_node.name],
                         radii_before[main_traj_move_node.name]) = self.get_positions_and_radii_before(main_traj_move_node)
                        (positions_after[new_bif_node.name],
                         radii_after[new_bif_node.name]) = self.get_positions_and_radii_after(new_bif_node)
                    else:
                        main_traj_move_node = main_traj_node
                        for j, level in enumerate(levelordergroup_iter(main_traj_move_node)):
                            if j < abs(offset) - 1 and len(level) == 1 and level[0] in path:
                                main_traj_move_node = level[0]
                            else:
                                break

                        new_bif_node_main_child = [node for node in new_bif_node.children
                                                   if node.name in path][0]
                        new_bif_node_main_child.parent = None
                        main_traj_move_node.parent = None

                        if new_bif_node.name not in positions_before:
                            (positions_before[new_bif_node.name],
                             radii_before[new_bif_node.name]) = self.get_positions_and_radii_before(new_bif_node)
                        (positions_after[main_traj_move_node.name],
                         radii_after[main_traj_move_node.name]) = self.get_positions_and_radii_after(main_traj_move_node)
            else:
                updated_move_node = move_node
                if new_bif_node.name not in positions_before:
                    (positions_before[new_bif_node.name],
                     radii_before[new_bif_node.name]) = self.get_positions_and_radii_before(new_bif_node)
                (positions_after[updated_move_node.name],
                 radii_after[updated_move_node.name]) = self.get_positions_and_radii_after(updated_move_node)
                if mn_i == len(nodes_to_move) - 1:
                    if offset > 0:
                        main_traj_move_node = bif_node
                        if (main_traj_move_node.parent and
                                main_traj_move_node.name != data['selected_node'].name):
                            prev_main_traj_move_node = main_traj_move_node
                            main_traj_move_node = main_traj_move_node.parent
                            prev_main_traj_move_node.parent = None
                        (positions_before[main_traj_move_node.name],
                         radii_before[main_traj_move_node.name]) = self.get_positions_and_radii_before(main_traj_move_node)
                        (positions_after[new_bif_node.name],
                         radii_after[new_bif_node.name]) = self.get_positions_and_radii_after(new_bif_node)
                        new_bif_node.parent = None
                    else:
                        main_traj_move_node = main_traj_node
                        if new_bif_node.name not in positions_before:
                            (positions_before[new_bif_node.name],
                             radii_before[new_bif_node.name]) = self.get_positions_and_radii_before(new_bif_node)
                        (positions_after[main_traj_move_node.name],
                         radii_after[main_traj_move_node.name]) = self.get_positions_and_radii_after(main_traj_move_node)

                        main_traj_move_node.parent = None
                        bif_node.parent = None

            max_dist = 1.2
            source_nodes = [new_bif_node]
            target_nodes = [updated_move_node]
            if mn_i == len(nodes_to_move) - 1:
                if offset > 0:
                    source_nodes.append(main_traj_move_node)
                    target_nodes.append(new_bif_node)
                else:
                    source_nodes.append(new_bif_node)
                    target_nodes.append(main_traj_move_node)

            for source_node, target_node in zip(source_nodes, target_nodes):
                bifur_dist = np.linalg.norm(np.array(source_node.position) - np.array(target_node.position))
                if bifur_dist > max_dist:
                    in_between_positions, in_between_radii = self.compute_inbetween_points(source_node,
                                                                                           target_node, positions_before, positions_after,
                                                                                           radii_before, radii_after, max_dist)

                    new_br_name = target_node.node_name.split("-")[0]
                    point_id = 0.0
                    parent = source_node
                    for i, (position, radius) in enumerate(zip(in_between_positions, in_between_radii)):
                        new_node = Node(new_br_name + "-" + str(point_id),
                                        position=position.tolist(),
                                        radius=radius,
                                        parent=parent,
                                        label=1)
                        parent = new_node
                        point_id += 0.1
                else:
                    parent = source_node

                target_node.parent = parent

            bif_node.label = 1
            new_bif_node.label = 2
            full_path = find_name(data['selected_node'].root, path[-1]).path_name
            data['selected_path'] = full_path[1:].split("/")

    def rename_node_names(self, data, selected_path_len):
        root_node = data['selected_node'].root
        selected_path_len_decreased = False
        if find_name(root_node, data['selected_path'][-1]).children == 0:
            selected_path_len -= 1
            selected_path_len_decreased = True

        st_br_id = int(root_node.node_name.split("-")[0])
        node_name = str(st_br_id) + "-" + str(0)
        new_root_node = Node(node_name,
                             position=root_node.position,
                             radius=root_node.radius,
                             label=root_node.label)
        old_node_id_to_new = {root_node.name: node_name}
        new_node_id_to_old = {node_name: root_node.name}
        new_level = None
        end_node = find_name(root_node, data['selected_path'][-1])
        end_node_depth = end_node.depth
        selected_node_level = end_node_depth - (selected_path_len + 1) + 1
        if selected_node_level < 1:
            selected_path_len = selected_path_len + selected_node_level - 1
            selected_node_level = 1
        past_traj_head_node_level = selected_node_level - (data['num_prev_pos'] - 1)
        if past_traj_head_node_level < 1:
            past_traj_head_node_level = 1
            data['num_prev_pos'] = selected_node_level - 1

        for i, level in enumerate(levelordergroup_iter(root_node)):
            if i == 0:
                continue

            if i == past_traj_head_node_level:
                if new_level:
                    data['past_traj_head_node'] = [node for node in new_level
                                                   if new_node_id_to_old[node.name] in data['selected_path']][0]
                else:
                    data['past_traj_head_node'] = new_root_node

            if i == selected_node_level:
                if new_level:
                    data['selected_node'] = [node for node in new_level
                                                   if new_node_id_to_old[node.name] in data['selected_path']][0]
                else:
                    data['selected_node'] = new_root_node

            new_level = []
            for node in level:
                parent = node.parent
                parent_name = old_node_id_to_new[parent.name]
                parent_node = find_name(new_root_node, parent_name)
                if len(parent.children) > 1:
                    st_br_id += 1
                    node_name = str(st_br_id) + "-" + str(0)
                else:
                    parent_br = int(parent_name.split("-")[0])
                    parent_pt = int(parent_name.split("-")[1])
                    node_name = str(parent_br) + "-" + str(parent_pt + 1)
                new_level.append(Node(node_name,
                     parent=parent_node,
                     position=node.position,
                     radius=node.radius,
                     label=node.label))
                old_node_id_to_new[node.name] = node_name
                new_node_id_to_old[node_name] = node.name

        if selected_path_len_decreased:
            selected_path_len += 1

        old_selected_node_index = data['selected_path'].index(new_node_id_to_old[data['selected_node'].name])
        old_end_node_index = data['selected_path'].index(end_node.name)
        cropped_old_path = data['selected_path'][old_selected_node_index: old_end_node_index + 1]
        data['selected_path'] = [old_node_id_to_new[node_name] for node_name in cropped_old_path]
        data['point_id'] = data['selected_node'].name

    def __call__(self, data, image_size):
        backup_path = copy.deepcopy(data['selected_path'])
        backup_selected_node = copy.deepcopy(data['selected_node'])

        if data['past_traj_head_node'] is None:
            print("Sample index: ", data['index'])
            print("Selected node: ", data['selected_node'].name)
            print("Selected path: ", data['selected_path'])
            print("Selected Node Tree:")
            for i, level in enumerate(levelordergroup_iter(data['selected_node'].root)):
                print(f"Level {i}: {[node.name for node in level]}")

        backup_past_traj_head_node = find_full_path(backup_selected_node,
                                                                                      data['past_traj_head_node'].path_name)
        bif_nodes = [bif_node for bif_node in preorder_iter(data['selected_node'])
                     if len(bif_node.children) > 1 and bif_node.name in data['selected_path']]
        selected_path_len = len(data['selected_path'])

        nodes_shifted = False
        for bif_node in bif_nodes:
            if bif_node.name == data['selected_path'][-1]:
                continue

            offset = self.randomize(bif_node, data)

            if abs(offset) > 0:
                # offset the bifurcation node
                self.offset_bifur_node(data, bif_node, offset)
                nodes_shifted = True

        if len(data['selected_path']) < self.min_num_pts:
            data['selected_node'] = backup_selected_node
            data['selected_path'] = backup_path
            data['past_traj_head_node'] = backup_past_traj_head_node
            nodes_shifted = False

        if nodes_shifted:
            self.rename_node_names(data, selected_path_len)

        return data


class AddRandBifurTrajNoised(Randomizable, MapTransform):
    def __init__(self, keys, num_prev_pos, seq_len, traj_train_len):
        super(Randomizable, self).__init__(keys)
        self.transform = AddRandBifurTrajNoise(num_prev_pos, seq_len, traj_train_len)

    def set_random_state(self, seed=None, state=None):
        self.transform.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform(d[key], data['image'].shape[-3:])

        return d


class ConvertTreeToTargets(Transform):
    def __init__(self, seq_len, num_prev_pos, sub_vol_size, class_dict):
        self.seq_len = seq_len
        self.num_prev_pos = num_prev_pos
        self.sub_vol_size = sub_vol_size
        self.class_dict = class_dict

    def extract_sub_tree(self, selected_node):
        root_position = np.asarray(selected_node.position).astype(int)
        grouped_node_seq = self.init_sub_tree(root_position)
        grouped_node_seq, level = self.extract_ordered_sub_tree(grouped_node_seq,
                                                                selected_node, root_position)
        grouped_node_seq = self.pad_sub_tree(grouped_node_seq, level)

        return grouped_node_seq

    @staticmethod
    def init_sub_tree(root_position):
        grouped_node_seq = {'root_position': root_position,
                            'node_ids': [], 'node_parents': [], 'parent_idxs': [], 'rel_positions': [],
                            'radii': [], 'labels': []}

        return grouped_node_seq

    @staticmethod
    def find_element_index(element, my_list):
        try:
            index = my_list.index(element)
            return index
        except ValueError:
            return -1

    def extract_ordered_sub_tree(self, grouped_node_seq, selected_node, root_position):
        prev_level_branches = []
        ordered_branches = []
        for level, curr_level_nodes in enumerate(levelordergroup_iter(selected_node)):
            if level < self.seq_len:
                new_branches, curr_level_branches = self.find_branches(curr_level_nodes, prev_level_branches)
                if len(new_branches):
                    ordered_branches += new_branches

                grouped_node_seq = self.init_prop_lists(grouped_node_seq)
                for index_br, branch in enumerate(ordered_branches):
                    curr_level_br_index = self.find_element_index(branch, curr_level_branches)
                    if curr_level_br_index == -1:
                        grouped_node_seq = self.pad_finished_branch(grouped_node_seq, branch, index_br, level)
                    else:
                        grouped_node_seq = self.append_node_info(grouped_node_seq, curr_level_nodes[curr_level_br_index],
                                                                 root_position, level)
                prev_level_branches = [name.split("-")[0] for name in grouped_node_seq['node_ids'][level]]
            else:
                level -= 1
                break

        return grouped_node_seq, level

    @staticmethod
    def find_branches(curr_level_nodes, prev_level_branches):
        curr_level_branches = []
        for node in curr_level_nodes:
            curr_level_branches.append(node.node_name.split("-")[0])
        new_branches = sorted(list(set(curr_level_branches) - set(prev_level_branches)))

        return new_branches, curr_level_branches

    @staticmethod
    def init_prop_lists(grouped_node_seq):
        grouped_node_seq['node_ids'].append([])
        grouped_node_seq['node_parents'].append([])
        grouped_node_seq['parent_idxs'].append([])
        grouped_node_seq['rel_positions'].append([])
        grouped_node_seq['radii'].append([])
        grouped_node_seq['labels'].append([])

        return grouped_node_seq

    def append_node_info(self, grouped_node_seq, node, root_position, level):
        rel_position = np.array(node.position - root_position, dtype=float)
        node_radius = node.radius
        num_children = len(node.children)
        if num_children < 2:
            label_id = num_children
        else:
            label_id = self.class_dict['bifurcation']
        rel_position /= (self.sub_vol_size // 2)
        node_radius /= (self.sub_vol_size // 2)

        if level:
            parent_id = node.parent.node_name if node.parent is not None else -1
            parent_idxs = self.find_element_index(parent_id, grouped_node_seq['node_ids'][level - 1])
        else:
            parent_id = -1
            parent_idxs = -1

        grouped_node_seq['node_ids'][level].append(node.node_name)
        grouped_node_seq['node_parents'][level].append(parent_id)
        grouped_node_seq['parent_idxs'][level].append(parent_idxs)
        grouped_node_seq['rel_positions'][level].append(rel_position.tolist())
        grouped_node_seq['radii'][level].append(node_radius)
        grouped_node_seq['labels'][level].append(label_id)

        return grouped_node_seq

    def pad_finished_branch(self, grouped_node_seq, branch_nm, index_br, level):
        parent_nm = grouped_node_seq['node_ids'][level - 1][index_br]
        point_nm = parent_nm.split("-")[1]
        new_point_nm = str(int(point_nm) + 1)
        node_id = branch_nm + "-" + new_point_nm

        node_position = grouped_node_seq['rel_positions'][level - 1][index_br]
        node_label = self.class_dict['background']
        grouped_node_seq['node_ids'][level].append(node_id)
        grouped_node_seq['node_parents'][level].append(parent_nm)
        grouped_node_seq['parent_idxs'][level].append(index_br)
        grouped_node_seq['rel_positions'][level].append(node_position)
        grouped_node_seq['radii'][level].append(0.0)
        grouped_node_seq['labels'][level].append(node_label)

        return grouped_node_seq

    def pad_sub_tree(self, grouped_node_seq, level):
        if level < self.seq_len - 1:
            prev_level_names = grouped_node_seq['node_ids'][-1]
            for j in range(level + 1, self.seq_len):
                level_names = []
                for name in prev_level_names:
                    br, nm = name.split("-")
                    new_nm = str(int(nm) + 1)
                    level_names.append(br + "-" + new_nm)

                level_positions = grouped_node_seq['rel_positions'][-1]
                node_label = self.class_dict['background']
                labels = (np.zeros(np.array(grouped_node_seq['labels'][-1]).shape, dtype=int) + node_label).tolist()
                level_radii = np.zeros(np.array(grouped_node_seq['radii'][-1]).shape).tolist()

                grouped_node_seq['node_ids'].append(level_names)
                grouped_node_seq['node_parents'].append(prev_level_names)
                grouped_node_seq['parent_idxs'].append(grouped_node_seq['parent_idxs'][-1])
                grouped_node_seq['rel_positions'].append(level_positions)
                grouped_node_seq['radii'].append(level_radii)
                grouped_node_seq['labels'].append(labels)
                prev_level_names = level_names

        return grouped_node_seq

    def generate_past_trajectory_pp(self, root_node, root_position):
        past_traj_label = []
        parent = root_node

        for i in range(self.num_prev_pos):
            node_info_list = []
            node_position = np.asarray(parent.position)
            rel_position = (node_position - root_position).astype(float)
            rel_position /= (self.sub_vol_size // 2)
            node_info_list.append(rel_position)
            node_radius = np.asarray(parent.radius).reshape(1)
            node_radius /= (self.sub_vol_size // 2)
            node_info_list.append(node_radius)
            past_traj_label.append(np.concatenate(node_info_list))
            parent = parent.parent
            if parent is None:
                break

        past_traj_label_np = np.asarray(past_traj_label)
        num_missing_points = self.num_prev_pos - past_traj_label_np.shape[0]
        if num_missing_points:
            last_point_info = past_traj_label_np[-1:]
            padding_info = np.tile(last_point_info, (num_missing_points, 1))
            past_traj_label_np = np.vstack((past_traj_label_np, padding_info))

        past_traj_label_t = torch.FloatTensor(past_traj_label_np).flatten()

        return past_traj_label_t

    def __call__(self, data):
        data = [data.copy() for _ in range(len(data['selected_node']))]
        for idx, label in enumerate(data):
            label['selected_node'] = label['selected_node'][idx]
            label['past_traj_head_node'] = label['past_traj_head_node'][idx]
            label.update(self.extract_sub_tree(label['selected_node']))
            label['past_tr'] = self.generate_past_trajectory_pp(label['selected_node'],
                                                                label['root_position'])
            label.pop('selected_node')
            label.pop('past_traj_head_node')

        return data


class ConvertTreeToTargetsd(MapTransform):
    def __init__(self, keys, seq_len, num_prev_pos, sub_vol_size, class_dict):
        super().__init__(keys)
        self.transform = ConvertTreeToTargets(seq_len, num_prev_pos, sub_vol_size, class_dict)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform(d[key])

        return d


class LoadImageCropsAndTrees(Transform):
    def __init__(self, seq_len, num_prev_pos, sub_vol_size, class_dict, images, annots, image_mins, masks):
        self.images = images
        self.annots = annots
        self.image_mins = image_mins
        self.masks = masks
        self.sub_vol_size = sub_vol_size
        self.crop_and_pad = CropAndPad(sub_vol_size)
        self.convert_tree_to_targets = ConvertTreeToTargets(seq_len, num_prev_pos, sub_vol_size, class_dict)

    def get_annot_dict(self, selected_node, input_dict):
        _, past_traj_num = ExtRandomSubTree.get_past_traj_start(selected_node, self.get_sub_tree.num_prev_pos)
        selected_node, past_traj_head_node = ExtRandomSubTree.create_subtree_with_past_tr_val(selected_node,
                                                                                           past_traj_num, self.get_sub_tree.seq_len)
        data = self.convert_tree_to_targets.extract_sub_tree(selected_node)
        data['past_tr'] = self.convert_tree_to_targets.generate_past_trajectory_pp(selected_node, data['root_position'])
        data.update(input_dict)

        return data

    def get_image_crops(self, input, selected_node):
        image = self.crop_and_pad(self.images[input['sample_id']], selected_node.position,
                                  self.image_mins[input['sample_id']])
        if self.masks is not None:
            mask = self.crop_and_pad(self.masks[input['sample_id']], selected_node.position,
                                     0.0)
        else:
            mask = None

        return image, mask

    def __call__(self, input):
        annot_tree = self.annots[input['sample_id']]['branches'][input['tree_id']]
        selected_node = find_name(annot_tree, input['node_id'])
        input_dict = {'index': input['sample_id'],
                      'tree_id': input['tree_id'],
                      'point_id': input['node_id'],
                      'point_type': input['point_type'],
                      'dist': input['distance']}

        data = self.get_annot_dict(selected_node, input_dict)
        image, mask = self.get_image_crops(input, selected_node)

        return data, image, mask


class LoadImageCropsAndTreesd(MapTransform):
    def __init__(self, keys, seq_len, num_prev_pos, sub_vol_size, class_dict, mask, paths,
                 window_input, window_min, window_max, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        image_mins = {}
        images = {}
        annots = {}
        if mask:
            masks = {}
        else:
            masks = None

        image_reader = LoadImage(image_only=True)
        annot_reader = LoadAnnotPickle()
        self.norm = NormalizeIntensity()

        if window_input:
            window = Compose(
                [ThresholdIntensity(threshold=window_max, above=False, cval=window_max),
                 ThresholdIntensity(threshold=window_min, above=True, cval=window_min)])

        for path in paths:
            index = [int(s) for s in re.findall(r'\d+', path[0])][-1]
            if window_input:
                image = window(image_reader(path[0]).unsqueeze(0))
            else:
                image = image_reader(path[0]).unsqueeze(0)
            image = self.norm(image)
            image_min = torch.min(image)
            images[index] = image
            image_mins[index] = image_min
            annots[index] = annot_reader(path[1])
            if mask:
                masks[index] = image_reader(path[2]).unsqueeze(0)

        self.mask = mask
        self.transform = LoadImageCropsAndTrees(seq_len, num_prev_pos, sub_vol_size, class_dict,
                                                images, annots, image_mins, masks)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key], image, mask = self.transform(d[key])
            d['image'] = self.norm(image) if self.norm_crop else image
            d['mask'] = mask
        return d