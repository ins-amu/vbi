#!/usr/bin/env python
# coding: utf-8

import vbi
from vbi import report_cfg
from vbi import extract_features_df, extract_features_list
from vbi import get_features_by_domain, get_features_by_given_names

D = vbi.LoadSample(nn=88)
ts = D.get_bold()
print(ts.shape)
import numpy as np
np.savetxt("ts.csv", ts[:10, :].T, delimiter=",")

cfg = get_features_by_domain(domain="connectivity")
cfg = get_features_by_given_names(cfg, ['fc_stat'])
cfg = vbi.update_cfg(cfg, 'fc_stat', {"features":["sum"], "fc_function":"corrcoef"})
report_cfg(cfg)

data = extract_features_list([ts], 0.5, cfg)

