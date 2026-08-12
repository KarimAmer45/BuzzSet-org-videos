#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

# The optimized COCO evaluator is an optional C++ extension. On Windows it
# requires MSVC's cl.exe and blocks this project before training/eval can run.
# Leave it unexported so callers fall back to pycocotools.COCOeval.
