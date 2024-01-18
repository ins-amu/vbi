import os
import vbi
import sys
import tqdm
import importlib
import numpy as np
import pandas as pd
from multiprocessing import Pool
from vbi.feature_extraction.features import *
from vbi.feature_extraction.features_utils import *
from vbi.feature_extraction.features_settings import *
from vbi.feature_extraction.utility import *


def calc_features(ts,
                  fs,
                  fea_dict,
                  preprocess=None,          # preprocess function
                  preprocess_args=None,     # arguments for preprocess function
                  **kwargs):
    # window_size=None,         # window size for preprocessing #!TODO
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

    features_path = fea_dict['features_path'] if (
        "features_path" in fea_dict.keys()) else None

    if features_path:
        module_name = features_path.split(os.sep)[-1][:-3]
        sys.path.append(
            features_path[:-len(features_path.split(os.sep)[-1])-1])
        exec("import " + module_name)
        importlib.reload(sys.modules[features_path.split(os.sep)[-1][:-3]])
        exec("from " + module_name + " import *")

    # module = sys.modules[module_name]
    # print(module.calc_mean)
    # print(module.calc_mean([1,2,3], 1, 2))

    def length(x):
        return (len(x)) if (len(x) > 0) else 0

    labels = []
    features = []
    info = {}

    domain = list(fea_dict.keys())
    # remove features_path from domain if exists
    if 'features_path' in domain:
        domain.remove('features_path')

    for _type in domain:
        domain_feats = fea_dict[_type]
        for fe in domain_feats:
            if fea_dict[_type][fe]['use'] == 'yes':
                c = length(features)
                func_name = fe
                func = fea_dict[_type][fe]['function']
                params = fea_dict[_type][fe]['parameters']

                if params is None:
                    params = {}

                if 'fs' in params.keys():
                    params['fs'] = fs

                if preprocess is not None:
                    ts = preprocess(ts, **preprocess_args)

                val, lab = eval(func)(ts, **params)

                if isinstance(val, (np.ndarray, list)):
                    labels.extend(lab)
                    features.extend(val)
                else:
                    labels.append(func_name)
                    features.append(val)
                info[func_name] = {'index': [c, length(features)]}

    return features, labels, info


def extract_features(ts,
                     fs,
                     fea_dict,
                     output_format='list',
                     **kwargs):
    #  window_size=None,         # window size for preprocessing #!TODO
    '''
    Extract features from time series data

    Parameters
    ----------
    ts : list of np.ndarray [[n_regions x n_samples]]
        Input from which the features are extracted
    fs : int, float
        Sampling frequency
    cfg : dictionary
        Dictionary of features to extract
    output_format : string
        Output format, either 
        'list' (list of numpy arrays)
        'dataframe' (pandas dataframe)
        (default is 'list')

    **kwargs
    --------
    n_workers : int
        Number of workers for parallelization, default is 1
        Parallelization is done by ensembles (first dimension of ts)
    dtype : type
        Data type of the features extracted, default is np.float32
    verbose : boolean
        If True, print the some information
    preprocess : function
        Function for preprocessing the time series
    preprocess_args : dictionary
        Arguments for preprocessing function

    Returns
    -------
    Data: object
        Object with the following attributes:
        values: list of numpy arrays or pandas dataframe
            extracted features
        labels: list of strings
            List of labels of the features
        info: dictionary
            Dictionary with the information of the features extracted
    '''

    labels = []
    features = []

    n_workers = kwargs.get('n_workers', 1)
    dtype = kwargs.get('dtype', np.float32)
    verbose = kwargs.get('verbose', True)
    preprocess = kwargs.get('preprocess', None)
    preprocess_args = kwargs.get('preprocess_args', None)

    def update_bar(verbose):
        if verbose:
            pbar.update()
        else:
            pass

    ts = prepare_input(ts)
    n_trial, n_region, n_sample = ts.shape

    if n_workers == 1:
        features = []

        for i in tqdm.tqdm(range(n_trial), disable=not verbose):
            fea, labels, info = calc_features(
                ts[i, :, :], fs, fea_dict, **kwargs)
            features.append(np.array(fea).astype(dtype))
    else:

        for i in range(n_trial):
            _, labels, info = calc_features(ts[i], fs, fea_dict,
                                            preprocess=preprocess,
                                            preprocess_args=preprocess_args,
                                            **kwargs)
            if info:
                break
        with Pool(processes=n_workers) as pool:
            with tqdm.tqdm(total=n_trial, disable=not verbose) as pbar:
                async_res = [pool.apply_async(calc_features,
                                              args=(ts[i],
                                                    fs,
                                                    fea_dict
                                                    ),
                                              kwds=dict(kwargs),
                                              callback=update_bar)
                             for i in range(n_trial)]
                features = [np.array(res.get()[0]).astype(dtype)
                            for res in async_res]

    if output_format == 'dataframe':
        features = pd.DataFrame(features)
        features.columns = labels
        
    data = Data_F(values=features, labels=labels, info=info)

    return data


def dataframe_feature_extractor(ts,
                                fs,
                                fea_dict,
                                **kwargs):
    '''
    extract features from time series data and return a pandas dataframe

    Parameters
    ----------
    ts : list of np.ndarray [[n_regions x n_samples]]
        Input from which the features are extracted
    fs : int, float
        Sampling frequency
    cfg : dictionary
        Dictionary of features to extract
    **kwargs
    --------
    n_workers : int
        Number of workers for parallelization, default is 1
        Parallelization is done by ensembles (first dimension of ts)
    dtype : type
        Data type of the features extracted, default is np.float32
    verbose : boolean
        If True, print the some information
    preprocess : function
        Function for preprocessing the time series
    preprocess_args : dictionary
        Arguments for preprocessing function

    Returns
    -------
    Data: object
        Object with the following attributes:
        values: pandas dataframe
            extracted features
        labels: list of strings
            List of labels of the features
        info: dictionary
            Dictionary with the information of the features extracted

    '''
    output_format = 'dataframe'
    return extract_features(ts, fs, fea_dict, output_format, **kwargs)


def list_feature_extractor(ts,
                           fs,
                           fea_dict,
                           **kwargs):
    '''
    extract features from time series data and return a pandas dataframe

    Parameters
    ----------
    ts : list of np.ndarray [[n_regions x n_samples]]
        Input from which the features are extracted
    fs : int, float
        Sampling frequency
    cfg : dictionary
        Dictionary of features to extract
    **kwargs
    --------
    n_workers : int
        Number of workers for parallelization, default is 1
        Parallelization is done by ensembles (first dimension of ts)
    dtype : type
        Data type of the features extracted, default is np.float32
    verbose : boolean
        If True, print the some information
    preprocess : function
        Function for preprocessing the time series
    preprocess_args : dictionary
        Arguments for preprocessing function

    Returns
    -------
    Data: object
        Object with the following attributes:
        values: list of numpy arrays
            extracted features
        labels: list of strings
            List of labels of the features
        info: dictionary
            Dictionary with the information of the features extracted

    '''
    output_format = 'list'
    return extract_features(ts, fs, fea_dict, output_format, **kwargs)
