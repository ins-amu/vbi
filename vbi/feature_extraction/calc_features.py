import vbi
import tqdm
import numpy as np
from multiprocessing import Pool
from vbi.feature_extraction.features import *
from vbi.feature_extraction.features_utils import *
from vbi.feature_extraction.features_settings import *


def calc_features(ts, dict_features, fs,
                  preprocess=None,          # preprocess function
                  preprocess_args=None,     # arguments for preprocess function
                  verbose=False):
    '''
    Extract features from time series data

    Parameters
    ----------
    dict_features : dictionary
        Dictionary of features to extract
    ts : nd-array
        Input from which the features are extracted
    fs : int, float
        Sampling frequency, set to 1 if not used

    Returns
    -------
    labels: list
        List of labels of the features
    features: list
        List of features extracted
    '''

    def length(x):
        return (len(x)) if (len(x) > 0) else 0

    labels = []
    features = []
    info = {}

    domain = list(dict_features.keys())

    for _type in domain:
        domain_feats = dict_features[_type]
        for feat in domain_feats:
            if dict_features[_type][feat]['use'] == 'yes':
                c = length(features)
                func_name = feat
                func = dict_features[_type][feat]['function']
                params = dict_features[_type][feat]['parameters']
                if isinstance(params, str):
                    params = {}
                # check if fs is in the parameters, set it to fs from input
                if 'fs' in params:
                    params['fs'] = fs

                # apply preprocessing if any
                if preprocess is not None:
                    ts = preprocess(ts, **preprocess_args)

                res = eval(func)(ts, **params)

                if isinstance(res, (np.ndarray, list)):
                    for i in range(len(res)):
                        labels.append(func_name + '_' + str(i))
                    features.extend(res)
                else:
                    labels.append(func_name)
                    features.append(res)
                info[func_name] = {'index': [c, length(features)]}

    return features, labels, info


def feature_extractor(ts, fs, dict_features,
                      preprocess=None,          # preprocess function
                      preprocess_args=None,     # arguments for preprocess function
                      verbose=False, **kwargs):
    '''
    Extract features from time series data

    Parameters
    ----------
    ts : nd-array [n_ensembles x n_regions x n_samples]
        Input from which the features are extracted
    fs : int, float
        Sampling frequency
    json_path : string
        Path to json file
    verbose : boolean
        If True, print the features extracted

    **kwargs
    --------
    n_workers : int
        Number of workers for parallelization, default is 1
        Parallelization is done by ensembles (first dimension of ts)


    Returns
    -------
    labels: list
        List of labels of the features
    features: list of lists
        List of features extracted, each list is features extracted from one ensemble
    '''

    features = []
    labels = []
    n_workers = kwargs.get('n_workers', 1)

    def update_bar(_):
        pbar.update()

    if isinstance(ts, (list, tuple)):
        ts = np.array(ts)

    if ts.ndim == 2:
        ts = ts.reshape(1, ts.shape[0], ts.shape[1])
    elif ts.ndim == 1:
        ts = ts.reshape(1, 1, ts.shape[0])
    else:
        pass
    ns, nr, nt = ts.shape

    if ns > 1:
        for i in range(ns):
            ts_ = ts[i, :, :]
            _, _, info = calc_features(ts_, dict_features, fs,
                                       preprocess=preprocess,
                                       preprocess_args=preprocess_args,
                                       verbose=verbose)
            if info:
                break

    if ns == 1:
        ts = ts.reshape(nr, nt)
        features, labels, info = calc_features(
            ts, dict_features, fs, verbose=verbose)
        return features, labels, info
    else:
        with Pool(processes=n_workers) as pool:
            with tqdm.tqdm(total=ns) as pbar:
                async_res = [pool.apply_async(
                    calc_features, (ts[i, :, :], dict_features, fs),
                    callback=update_bar)
                    for i in range(ns)]
                features = [res.get()[0] for res in async_res]

    return features, labels, info
