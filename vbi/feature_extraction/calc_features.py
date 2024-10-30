import os
import vbi
import sys
import tqdm
import importlib
import numpy as np
import pandas as pd
from multiprocessing import Pool
import vbi.feature_extraction.features
from vbi.feature_extraction.features_settings import Data_F


def calc_features(
    ts: np.ndarray,
    fs: float,
    cfg: dict,
    preprocess=None,
    preprocess_args: dict = {},
    verbose: bool = False,
):
    """
    Extract features from time series data.

    Parameters
    ----------
    ts : np.ndarray
        Time series data
    fs : int, float
        Sampling frequency
    cfg : dict
        Dictionary of features configurations
    preprocess : function
        Function for preprocessing the time series
    preprocess_args : dictionary
        Arguments for preprocessing function

    Returns
    -------
    features : list of numpy arrays

    """

    features_path = cfg["features_path"] if ("features_path" in cfg.keys()) else None

    if features_path:
        module_name = features_path.split(os.sep)[-1][:-3]
        sys.path.append(features_path[: -len(features_path.split(os.sep)[-1]) - 1])
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

    domain = list(cfg.keys())
    # remove features_path from domain if exists
    if "features_path" in domain:
        domain.remove("features_path")

    for _type in domain:
        domain_feats = cfg[_type]
        for fe in domain_feats:
            if cfg[_type][fe]["use"] == "yes":
                c = length(features)
                func_name = fe
                func = cfg[_type][fe]["function"]
                params = cfg[_type][fe]["parameters"]
                
                if "verbose" in params.keys():
                    params["verbose"] = verbose

                if params is None:
                    params = {}

                if "fs" in params.keys():
                    params["fs"] = fs

                if preprocess is not None:
                    ts = preprocess(ts, **preprocess_args)
                val, lab = eval(func)(ts, **params)

                if isinstance(val, (np.ndarray, list)):
                    labels.extend(lab)
                    features.extend(val)
                else:
                    labels.append(func_name)
                    features.append(val)
                info[func_name] = {"index": [c, length(features)]}

    return features, labels, info


def extract_features(
    ts: np.ndarray, fs: float, cfg: dict, output_type=Data_F, **kwargs
):
    """
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
    - n_workers : int
        Number of workers for parallelization, default is 1
        Parallelization is done by ensembles (first dimension of ts)
    - dtype : type
        Data type of the features extracted, default is np.float32
    - verbose : boolean
        If True, print the some information
    - preprocess : function
        Function for preprocessing the time series
    - preprocess_args : dictionary
        Arguments for preprocessing function
    - n_trial: int
        Number of trials

    Returns
    -------
    Data Object with the following attributes:
    values: list of numpy arrays or pandas dataframe
        extracted features
    labels: list of strings
        List of labels of the features
    info: dictionary
        Dictionary with the information of the features extracted

    """

    labels = []
    features = []

    n_workers = kwargs.get("n_workers", 1)
    dtype = kwargs.get("dtype", np.float32)
    verbose = kwargs.get("verbose", True)
    preprocess = kwargs.get("preprocess", None)
    preprocess_args = kwargs.get("preprocess_args", {})
    n_trial = kwargs.get("n_trial", len(ts))

    def update_bar(verbose):
        if verbose:
            pbar.update()
        else:
            pass

    # ts, n_trial = prepare_input(ts)
    labels = None
    info = None

    if n_workers == 1:
        features = []
        for i in tqdm.tqdm(range(n_trial), disable=not verbose):
            values, _labels, _info = calc_features(
                ts[i], fs, cfg, preprocess, preprocess_args
            )
            features.append(np.array(values).astype(dtype))
            if (labels is None) and (not np.isnan(values).any()):
                labels = _labels
                info = _info
    else:
        for i in range(n_trial):
            values, _labels, _info = calc_features(
                ts[i],
                fs,
                cfg,
                preprocess=preprocess,
                preprocess_args=preprocess_args
            )
            if (labels is None) and (not np.isnan(values).any()):
                labels = _labels
                info = _info
                break
        with Pool(processes=n_workers) as pool:
            with tqdm.tqdm(total=n_trial, disable=not verbose) as pbar:
                async_res = [
                    pool.apply_async(
                        calc_features,
                        args=(ts[i], fs, cfg, preprocess, preprocess_args),
                        # kwds=dict(kwargs),
                        callback=update_bar,
                    )
                    for i in range(n_trial)
                ]
                features = [np.array(res.get()[0]).astype(dtype) for res in async_res]

    if output_type == "dataframe":
        df = pd.DataFrame(features)
        if labels is not None:
            df.columns = labels
        return df
    elif output_type == "list":
        return features, labels

    data = Data_F(values=features, labels=labels, info=info)

    return data


def extract_features_df(ts: np.ndarray, fs: float, cfg: dict, **kwargs):
    """
    Extract features from time series data and return a pandas dataframe

    Parameters
    ----------
    ts : list of np.ndarray [[n_regions x n_samples]]
        Input from which the features are extracted
    fs : int, float
        Sampling frequency
    cfg : dictionary
        Dictionary of features to extract

    **kwargs
    - n_workers : int
        Number of workers for parallelization, default is 1
        Parallelization is done by ensembles (first dimension of ts)
    - dtype : type
        Data type of the features extracted, default is np.float32
    - verbose : boolean
        If True, print the some information
    - preprocess : function
        Function for preprocessing the time series
    - preprocess_args : dictionary
        Arguments for preprocessing function

    Returns
    -------

    Data Object with the following attributes:
    - values: pandas dataframe
        extracted features
    - labels: list of strings
        List of labels of the features
    - info: dictionary
        Dictionary with the information of the features extracted
    """
    return extract_features(ts, fs, cfg, "dataframe", **kwargs)


def extract_features_list(ts, fs, cfg, **kwargs):
    """
    extract features from time series data and return a list of features and labels

    Parameters
    ----------
    ts : list of np.ndarray [[n_regions x n_samples]]
        Input from which the features are extracted
    fs : int, float
        Sampling frequency
    cfg : dictionary
        Dictionary of features to extract

    **kwargs
    - n_workers : int
        Number of workers for parallelization, default is 1
        Parallelization is done by ensembles (first dimension of ts)
    - dtype : type
        Data type of the features extracted, default is np.float32
    - verbose : boolean
        If True, print the some information
    - preprocess : function
        Function for preprocessing the time series
    - preprocess_args : dictionary
        Arguments for preprocessing function

    Returns
    -------
    Data Object with the following attributes:
    - values: list of numpy arrays
        extracted features
    - labels: list of strings
        List of labels of the features
    - info: dictionary
        Dictionary with the information of the features extracted
    """
    return extract_features(ts, fs, cfg, "list", **kwargs)
