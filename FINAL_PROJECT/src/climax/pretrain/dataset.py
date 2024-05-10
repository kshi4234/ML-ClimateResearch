# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import os
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset


class NpyReader(IterableDataset):
    def __init__(
        self,
        file_list,
        start_idx,
        end_idx,
        variables,
        out_variables,
        shuffle: bool = False,
        multi_dataset_training=False,
    ) -> None:
        super().__init__()
        start_idx = int(start_idx * len(file_list))
        end_idx = int(end_idx * len(file_list))
        file_list = file_list[start_idx:end_idx]
        self.file_list = [f for f in file_list if "climatology" not in f]
        self.variables = variables
        self.out_variables = out_variables if out_variables is not None else variables
        self.shuffle = shuffle
        self.multi_dataset_training = multi_dataset_training

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.file_list)
        # print("------------------------ WORKER INFO -----------------------")
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.file_list)
        else:
            if not torch.distributed.is_initialized():
                # print("NO DISTRIBUTED")
                rank = 0
                world_size = 1
            else:
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            num_workers_per_ddp = worker_info.num_workers
            if self.multi_dataset_training:
                # print("MULTIDATASET TRAINING!!!")
                num_nodes = int(os.environ.get("SLURM_NNODES", None))
                num_gpus_per_node = int(world_size / num_nodes)
                # print("WORLD_SIZE: %d, GPUS_PER_NODE: %d", world_size, num_gpus_per_node)
                # print("RANK:", rank)
                num_shards = num_workers_per_ddp * num_gpus_per_node
                rank = rank % num_gpus_per_node
            else:
                num_shards = num_workers_per_ddp * world_size
            per_worker = int(math.floor(len(self.file_list) / float(num_shards)))
            worker_id = rank * num_workers_per_ddp + worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker
        # print("OUT OF WORKER STUFF!")
        # print("RANK:", rank, ", FILE LIST:", self.file_list)
        for idx in range(iter_start, iter_end):
            path = self.file_list[idx]
            # print("RANK:", rank, ", PATH:", path)
            print("PATH:", path)
            data = np.load(path)
            print("DATA:", data)
            # print("RANK:", rank, ", DATA:", data)
            # print("VARIABLES:", self.variables)
            yield {k: data[k] for k in self.variables}, self.variables, self.out_variables
            # print("NPREADER IDX:", idx)


class Forecast(IterableDataset):
    def __init__(
        self, dataset: NpyReader, max_predict_range: int = 6, random_lead_time: bool = False, hrs_each_step: int = 1
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.max_predict_range = max_predict_range
        self.random_lead_time = random_lead_time
        self.hrs_each_step = hrs_each_step

    def __iter__(self):
        # print("FORECAST -------------")
        # i = 1
        for data, variables, out_variables in self.dataset:
            x = np.concatenate([data[k].astype(np.float32) for k in data.keys()], axis=1)
            x = torch.from_numpy(x)
            y = np.concatenate([data[k].astype(np.float32) for k in out_variables], axis=1)
            y = torch.from_numpy(y)

            inputs = x[: -self.max_predict_range]  # N, C, H, W

            if self.random_lead_time:
                predict_ranges = torch.randint(low=1, high=self.max_predict_range, size=(inputs.shape[0],))
            else:
                predict_ranges = torch.ones(inputs.shape[0]).to(torch.long) * self.max_predict_range
            lead_times = self.hrs_each_step * predict_ranges / 100
            lead_times = lead_times.to(inputs.dtype)
            output_ids = torch.arange(inputs.shape[0]) + predict_ranges
            outputs = y[output_ids]

            # print(i)
            # i+=1
            yield inputs, outputs, lead_times, variables, out_variables


class IndividualForecastDataIter(IterableDataset):
    def __init__(self, dataset, transforms: torch.nn.Module, output_transforms: torch.nn.Module, region_info = None):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.output_transforms = output_transforms
        self.region_info = region_info

    def __iter__(self):
        # print("Individual Data ------------------")
        # j = 1
        for (inp, out, lead_times, variables, out_variables) in self.dataset:
            assert inp.shape[0] == out.shape[0]
            for i in range(inp.shape[0]):
                if self.region_info is not None:
                    yield self.transforms(inp[i]), self.output_transforms(out[i]), lead_times[i], variables, out_variables, self.region_info
                else:
                    yield self.transforms(inp[i]), self.output_transforms(out[i]), lead_times[i], variables, out_variables
            # print("INDIVIDUAL FORECAST IDX:", j)
            # j+=1


class ShuffleIterableDataset(IterableDataset):
    def __init__(self, dataset, buffer_size: int) -> None:
        super().__init__()
        assert buffer_size > 0
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buf = []
        # print("Shuffle-------------")
        for x in self.dataset:
            # print("X:", x[0].size(), x[1].size(), x[2], x[3])
            if len(buf) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        # print("SHUFFLING!!!")
        random.shuffle(buf)
        # print("SHUFFLING SUCCESS!!!")
        while buf:
            yield buf.pop()
