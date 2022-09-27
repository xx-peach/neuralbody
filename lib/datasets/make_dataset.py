from .transforms import make_transforms
from . import samplers
import torch
import torch.utils.data
import imp
import os
from .collate_batch import make_collator
import numpy as np
import time
from lib.config.config import cfg


def _dataset_factory(is_train):
    if is_train:
        module = cfg.train_dataset_module
        path = cfg.train_dataset_path
        args = cfg.train_dataset
    else:
        module = cfg.test_dataset_module
        path = cfg.test_dataset_path
        args = cfg.test_dataset
    dataset = imp.load_source(module, path).Dataset(**args)
    return dataset


def make_dataset(cfg, dataset_name, transforms, is_train=True):
    dataset = _dataset_factory(is_train)
    return dataset


def make_data_sampler(dataset, shuffle, is_distributed, is_train):
    if not is_train and cfg.test.sampler == 'FrameSampler':
        sampler = samplers.FrameSampler(dataset)
        return sampler
    if is_distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter,
                            is_train):
    if is_train:
        batch_sampler = cfg.train.batch_sampler
        sampler_meta = cfg.train.sampler_meta
    else:
        batch_sampler = cfg.test.batch_sampler
        sampler_meta = cfg.test.sampler_meta

    if batch_sampler == 'default':
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last)
    elif batch_sampler == 'image_size':
        batch_sampler = samplers.ImageSizeBatchSampler(sampler, batch_size,
                                                       drop_last, sampler_meta)

    if max_iter != -1:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, max_iter)
    return batch_sampler


def worker_init_fn(worker_id):
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))


def make_data_loader(cfg, is_train=True, is_distributed=False, max_iter=-1):
    """ Create Dataset and Dataloader
    Args:
        cfg            - all the configurations for this experiment
        is_train       - bool, True if training, false if testing
        is_distributed - bool, True if we're using distributed traing/testing
        max_iter       - int, maximum number of iterations for one epoch
    Returns:
        data_loader - torch.utils.data.DataLoader(dataset, batch_sampler, num_workers, collator, worker_init_fn)
    """
    # get `batch_size`, `shuffle`, and `drop_last` configs according to is_train
    if is_train:
        batch_size = cfg.train.batch_size
        # shuffle = True
        shuffle = cfg.train.shuffle
        drop_last = False
    else:
        batch_size = cfg.test.batch_size
        shuffle = True if is_distributed else False
        drop_last = False

    # specify the dataset name for this experiment
    dataset_name = cfg.train.dataset if is_train else cfg.test.dataset
    #? create data transform, namely ToTensor() + Normalize(), 没用到
    transforms = make_transforms(cfg, is_train)
    # create whole dataset for this experiment
    dataset = make_dataset(cfg, dataset_name, transforms, is_train)

    # create data sampler and batch data sampler
    sampler = make_data_sampler(dataset, shuffle, is_distributed, is_train)
    batch_sampler = make_batch_data_sampler(cfg, sampler, batch_size,
                                            drop_last, max_iter, is_train)

    # create final data loader with dataset, batch_sampler, num_workers, collator(default '') and worker_intit_fn
    num_workers = cfg.train.num_workers
    collator = make_collator(cfg, is_train)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_sampler=batch_sampler,
                                              num_workers=num_workers,
                                              collate_fn=collator,
                                              worker_init_fn=worker_init_fn)

    return data_loader
