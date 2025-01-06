import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

import my_utils

device = "cuda"


def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation="tanh"):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == "tanh":
        net.append(nn.Tanh())
    elif activation == "sig":
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net


class RewardModel:
    def __init__(
        self,
        ds,
        da,
        ensemble_size=3,
        lr=3e-4,
        mb_size=128,
        size_segment=1,
        max_size=100,
        activation="tanh",
        capacity=5e5,  # length of total labeled queries
        large_batch=1,
        env=None,
        max_inputs_size=1e5,  # length of total queries saved to be asked to user
        mirror=False,
    ):
        # train data is trajectories, must process to sa and s..
        self.ds = ds
        self.da = da
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment

        self.capacity = capacity * 3
        if mirror:
            self.capacity = 4 * self.capacity
        self.buffer_seg1 = np.empty(
            (self.capacity, size_segment, self.ds + self.da), dtype=np.float32
        )
        self.buffer_seg2 = np.empty(
            (self.capacity, size_segment, self.ds + self.da), dtype=np.float32
        )
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False

        self.construct_ensemble()
        self.inputs = []
        self.body_states = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 128
        self.CEloss = nn.CrossEntropyLoss()
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.large_batch = large_batch

        self.env = env

        self.max_inputs_size = max_inputs_size
        self.inputs_size = 0

        self.mirror = mirror

        self.best_query_so_far = None

    def construct_ensemble(self):
        for i in range(self.de):
            model = (
                nn.Sequential(
                    *gen_net(
                        in_size=self.ds + self.da,
                        out_size=1,
                        H=256,
                        n_layers=3,
                        activation=self.activation,
                    )
                )
                .float()
                .to(device)
            )
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())

        self.opt = torch.optim.Adam(self.paramlst, lr=self.lr)

    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size * new_frac)

    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)

    def add_data(self, obs, act, body_state):
        if len(obs) < self.size_segment:
            return
        sa = np.concatenate([obs, act], axis=-1)
        self.inputs.append(sa)
        self.body_states.append(body_state)

        self.inputs_size += len(obs)
        while self.inputs_size > self.max_inputs_size:
            self.inputs_size -= len(self.inputs[0])
            self.inputs.pop(0)
            self.body_states.pop(0)

    def get_rank_probability(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)

        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def get_entropy(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def p_hat_member(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

        # taking 0 index for probability x_1 > x_2
        return F.softmax(r_hat, dim=-1)[:, 0]

    def p_hat_entropy(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent

    def r_hat_member(self, x, member=-1):
        # the network parameterizes r hat in eqn 1 from the paper
        return self.ensemble[member](torch.from_numpy(x).float().to(device))

    def r_hat(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)

    def r_hat_batch(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        return np.mean(r_hats, axis=0)

    def r_hat_tensor(self, x):
        # r_hat that gets batch of tensor and return batch of tensor
        r_hats = torch.stack([self.ensemble[member](x) for member in range(self.de)])
        return torch.mean(r_hats, dim=0)

    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(),
                "%s/reward_model_%s_%s.pt" % (model_dir, step, member),
            )

    def load(self, model_dir, step):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load("%s/reward_model_%s_%s.pt" % (model_dir, step, member))
            )

    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len / batch_size))

        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch + 1) * batch_size
            if (epoch + 1) * batch_size > max_len:
                last_index = max_len

            sa_t_1 = self.buffer_seg1[epoch * batch_size : last_index]
            sa_t_2 = self.buffer_seg2[epoch * batch_size : last_index]
            labels = self.buffer_label[epoch * batch_size : last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(device)
            total += labels.size(0)
            for member in range(self.de):
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)

    def get_queries(self, mb_size=20):
        max_len = len(self.inputs)

        # Select batch indices proportional to the length of each trajectory
        len_traj_list = [len(traj) for traj in self.inputs]
        total_length = sum(len_traj_list)
        probabilities = np.array(len_traj_list) / total_length

        # Sample batch indices with probability proportional to trajectory length
        batch_index_1 = np.random.choice(
            max_len, size=mb_size, replace=True, p=probabilities
        )
        batch_index_2 = np.random.choice(
            max_len, size=mb_size, replace=True, p=probabilities
        )

        sa1 = [self.inputs[i] for i in batch_index_1]
        bs1 = [self.body_states[i] for i in batch_index_1]
        sa2 = [self.inputs[i] for i in batch_index_2]
        bs2 = [self.body_states[i] for i in batch_index_2]

        starts1 = [
            np.random.randint(0, len(arr) - self.size_segment + 1) for arr in sa1
        ]
        sa1 = np.array(
            [arr[start : start + self.size_segment] for arr, start in zip(sa1, starts1)]
        )
        bs1 = np.array(
            [arr[start : start + self.size_segment] for arr, start in zip(bs1, starts1)]
        )

        starts2 = [
            np.random.randint(0, len(arr) - self.size_segment + 1) for arr in sa2
        ]
        sa2 = np.array(
            [arr[start : start + self.size_segment] for arr, start in zip(sa2, starts2)]
        )
        bs2 = np.array(
            [arr[start : start + self.size_segment] for arr, start in zip(bs2, starts2)]
        )

        return sa1, sa2, bs1, bs2

    def put_queries(self, sa1, sa2, labels):
        if self.mirror:
            sa1_mirror = sa1.copy()
            sa2_mirror = sa2.copy()
            for i in range(len(sa1)):
                sa1_mirror[i] = self.env.unwrapped.get_mirrored_state_action_query(
                    sa1[i]
                )
                sa2_mirror[i] = self.env.unwrapped.get_mirrored_state_action_query(
                    sa2[i]
                )

            sa1 = np.concatenate((sa1, sa1, sa1_mirror, sa1_mirror), axis=0)
            sa2 = np.concatenate((sa2, sa2_mirror, sa2, sa2_mirror), axis=0)
            labels = np.concatenate((labels, labels, labels, labels), axis=0)

        total_sample = sa1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(
                self.buffer_seg1[self.buffer_index : self.capacity],
                sa1[:maximum_index],
            )
            np.copyto(
                self.buffer_seg2[self.buffer_index : self.capacity],
                sa2[:maximum_index],
            )
            np.copyto(
                self.buffer_label[self.buffer_index : self.capacity],
                labels[:maximum_index],
            )

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index : next_index], sa1)
            np.copyto(self.buffer_seg2[self.buffer_index : next_index], sa2)
            np.copyto(self.buffer_label[self.buffer_index : next_index], labels)
            self.buffer_index = next_index

    def uniform_sampling(self, size=0):
        labels = []
        sa1s = []
        sa2s = []
        size = size if size else self.mb_size
        cnt = 0
        while cnt < size:
            sa1, sa2, bs1, bs2 = self.get_queries(mb_size=1)

            sa1, sa2, bs1, bs2 = sa1[0], sa2[0], bs1[0], bs2[0]

            frames1 = self.env.unwrapped.obs_query_to_scene_input(
                sa1[:, : self.ds], bs1
            )
            frames2 = self.env.unwrapped.obs_query_to_scene_input(
                sa2[:, : self.ds], bs2
            )
            label, best = my_utils.label_preference(
                frames1, frames2, self.env.unwrapped.step_dt
            )
            if label is not None:
                cnt += 1
                labels.append([label])
                sa1s.append(sa1)
                sa2s.append(sa2)
                if self.best_query_so_far is not None and best is None:
                    labels.append([0])
                    sa1s.append(self.best_query_so_far)
                    sa2s.append(sa1)
                    labels.append([0])
                    sa1s.append(self.best_query_so_far)
                    sa2s.append(sa2)
                if best is not None:
                    if best == 0:
                        self.best_query_so_far = sa1
                    else:
                        self.best_query_so_far = sa2

        self.put_queries(np.array(sa1s), np.array(sa2s), np.array(labels))

        return len(labels)

    def disagreement_sampling(self, size=0):
        labels = []
        sa1s = []
        sa2s = []
        size = size if size else self.mb_size
        cnt = 0
        while cnt < size:
            sa1, sa2, bs1, bs2 = self.get_queries(mb_size=self.large_batch)

            _, disagree = self.get_rank_probability(sa1, sa2)
            top_index = (-disagree).argsort()[0]
            sa1, sa2, bs1, bs2 = (
                sa1[top_index],
                sa2[top_index],
                bs1[top_index],
                bs2[top_index],
            )

            frames1 = self.env.unwrapped.obs_query_to_scene_input(
                sa1[:, : self.ds], bs1
            )
            frames2 = self.env.unwrapped.obs_query_to_scene_input(
                sa2[:, : self.ds], bs2
            )
            label, best = my_utils.label_preference(
                frames1, frames2, self.env.unwrapped.step_dt
            )
            if label is not None:
                cnt += 1
                labels.append([label])
                sa1s.append(sa1)
                sa2s.append(sa2)
                if self.best_query_so_far is not None and best is None:
                    labels.append([0])
                    sa1s.append(self.best_query_so_far)
                    sa2s.append(sa1)
                    labels.append([0])
                    sa1s.append(self.best_query_so_far)
                    sa2s.append(sa2)
                if best is not None:
                    if best == 0:
                        self.best_query_so_far = sa1
                    else:
                        self.best_query_so_far = sa2

        self.put_queries(np.array(sa1s), np.array(sa2s), np.array(labels))

        return len(labels)

    def high_reward_sampling(self, size=0):
        labels = []
        sa1s = []
        sa2s = []
        size = size if size else self.mb_size
        cnt = 0
        while cnt < size:
            sa1, sa2, bs1, bs2 = self.get_queries(mb_size=self.large_batch)
            sa = np.concatenate((sa1, sa2), axis=0)
            bs = np.concatenate((bs1, bs2), axis=0)

            r = np.array([self.r_hat_batch(sa) for sa in sa]).sum(axis=1).squeeze()

            top_index1 = (-r).argsort()[0]
            top_index2 = (-r).argsort()[1]
            sa1, sa2, bs1, bs2 = (
                sa[top_index1],
                sa[top_index2],
                bs[top_index1],
                bs[top_index2],
            )

            frames1 = self.env.unwrapped.obs_query_to_scene_input(
                sa1[:, : self.ds], bs1
            )
            frames2 = self.env.unwrapped.obs_query_to_scene_input(
                sa2[:, : self.ds], bs2
            )
            label, best = my_utils.label_preference(
                frames1, frames2, self.env.unwrapped.step_dt
            )
            if label is not None:
                cnt += 1
                labels.append([label])
                sa1s.append(sa1)
                sa2s.append(sa2)
                if self.best_query_so_far is not None and best is None:
                    labels.append([0])
                    sa1s.append(self.best_query_so_far)
                    sa2s.append(sa1)
                    labels.append([0])
                    sa1s.append(self.best_query_so_far)
                    sa2s.append(sa2)
                if best is not None:
                    if best == 0:
                        self.best_query_so_far = sa1
                    else:
                        self.best_query_so_far = sa2

        self.put_queries(np.array(sa1s), np.array(sa2s), np.array(labels))

        return len(labels)

    def high_reward_and_disagreement_sampling(self):
        return self.high_reward_sampling(
            self.mb_size // 2
        ) + self.disagreement_sampling((self.mb_size + 1) // 2)

    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0

        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0

            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            for member in range(self.de):

                # get random batch
                idxs = total_batch_index[member][
                    epoch * self.train_batch_size : last_index
                ]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)

                if member == 0:
                    total += labels.size(0)

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total

        return ensemble_acc

    def train_soft_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0

        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0

            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            for member in range(self.de):

                # get random batch
                idxs = total_batch_index[member][
                    epoch * self.train_batch_size : last_index
                ]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)

                if member == 0:
                    total += labels.size(0)

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros_like(r_hat).scatter(
                    1, labels.unsqueeze(1), 1  # dim, idx, value to write
                )
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total

        return ensemble_acc

    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]
