#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_num_workers = 16
        self.max_epoch = 300        
        self.input_size = (256, 832)  # (height, width)
        self.test_size = (256, 832)
        self.num_classes = 7
        
        self.train_ann = "train.json"
        # name of annotation file for evaluation
        self.val_ann = "val.json"
        # name of annotation file for testing
        self.test_ann = "test.json"