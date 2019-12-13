# -*- coding: utf-8 -*-
# @Time    : 2019/12/13 14:36
# @Author  : zhoujun
import os
import tarfile
DATA_ROOT = 'data/'
fname = os.path.join(DATA_ROOT, "aclImdb_v1.tar.gz")
if not os.path.exists(os.path.join(DATA_ROOT, "aclImdb")):
    print("从压缩包解压...")
    with tarfile.open(fname, 'r') as f:
        f.extractall(DATA_ROOT)