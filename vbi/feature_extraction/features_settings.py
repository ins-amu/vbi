import json
import vbi
import numpy as np
from copy import deepcopy


def load_json(path):
    '''
    Load json file

    Parameters
    ----------
    path : string
        Path to json file

    Returns
    -------
    json_data : dictionary
        Dictionary with the json data
    '''
    with open(path) as json_file:
        json_data = json.load(json_file)
    return json_data


def get_features_by_domain(domain=None, json_path=None):
    '''
    Create a dictionary of features in given domain(s).

    Parameters
    ----------
    domain : string or list of strings
        Domain of features to extract, including 'temporal', 'statistical', 'connectivity' and 'spectral'
    path : string (optional)
        Path to json file, if None, the default json file is used
    '''
    _domains = [
        "spectral",
        "connectivity",
        "temporal",
        "statistical",
        "information_theory"
        ]

    
    if json_path is None:
        json_path = vbi.__path__[0] + '/feature_extraction/features.json'

    if not isinstance(domain, (list, tuple)):
        domain = [domain]

    domain = list(set(domain))
    domain = [d.lower() for d in domain if d is not None] # lower case
    
    # for d in domain:
    #     if d not in valid_domains:
    #         raise SystemExit(
    #             f'Domain not valid. Please choose between: {" ".join(valid_domains)}')

    dict_features = load_json(json_path)
    if len(domain) == 0:
        return dict_features

    for d in _domains:
        if d not in domain:
            dict_features.pop(d)
    return dict_features


def get_features_by_given_names(cfg, names=None):
    '''
    filter features by given names from cfg (a dictionary of features)
    '''

    cfg = deepcopy(cfg)

    if names is None:
        return cfg

    if not isinstance(names, (list, tuple)):
        names = [names]
    
    names = [n.lower() for n in names] # lower case

    # check if names are valid
    avail_names = []
    for d in cfg:
        avail_names += list(cfg[d].keys())

    for n in names:
        if n not in avail_names:
            print(f'Warning: {n} is not a valid in provided feature names.')

    # filter cfg
    for d in cfg:
        for f in list(cfg[d].keys()):
            if f not in names:
                cfg[d].pop(f)

    return cfg


def get_features_by_tag(tag=None, json_path=None): #! TODO: not tested
    '''
    Create a dictionary of features in given tag.

    Parameters
    ----------
    tag : string
        Tag of features to extract
    path : string
        Path to json file
    '''

    available_tags = ['fmri', 'audio', 'eeg', 'ecg', None]

    if path is None:
        path = vbi.__path__[0] + '/feature_extraction/features.json'

        if tag not in ['audio', 'eeg', 'ecg', None]:
            raise SystemExit(
                'Tag not valid. Please choose between: audio, eeg, ecg or None')
    features_tag = {}
    dict_features = load_json(json_path)
    if tag is None:
        return dict_features
    else:
        for domain in dict_features:
            features_tag[domain] = {}
            for feat in dict_features[domain]:
                if dict_features[domain][feat]["use"] == "no":
                    continue
                # Check if tag is defined
                try:
                    js_tag = dict_features[domain][feat]["tag"]
                    if isinstance(js_tag, list):
                        if any([tag in js_t for js_t in js_tag]):
                            features_tag[domain].update(
                                {feat: dict_features[domain][feat]})
                    elif js_tag == tag:
                        features_tag[domain].update(
                            {feat: dict_features[domain][feat]})
                except KeyError:
                    continue
        # To remove empty dicts
        return dict([[d, features_tag[d]] for d in list(features_tag.keys()) if bool(features_tag[d])])


def add_feature(cfg,
                domain,
                name,
                features_path,
                parameters={},
                tag=None,
                description=""
                ):
    '''
    Add a feature to the cfg dictionary

    Parameters
    ----------
    cfg : dictionary
        Dictionary of features
    domain : string
        Domain of the feature
    name : string
        Name of the feature
    function : function
        Function to compute the feature
    parameters : dictionary
        Parameters of the feature
    tag : string
        Tag of the feature
    description : string
        Description of the feature

    Returns
    -------
    cfg : dictionary
        Updated dictionary of features
    '''
    if isinstance(features_path, str):
        features_path = __import__(features_path)
    _path = features_path.__file__

    # _path = getattr(feature_path, name)

    if domain not in cfg:
        cfg[domain] = {}

    cfg[domain][name] = {}
    cfg[domain][name]["parameters"] = parameters
    cfg[domain][name]["tag"] = tag
    cfg[domain][name]["description"] = description
    cfg[domain][name]["use"] = "yes"
    cfg[domain][name]["function"] = name.lower()
    cfg["features_path"] = _path
    # function.__module__ + "." + function.__name__

    return cfg


def add_features_from_json(json_path, features_path, fea_dict={}):
    '''
    add features from json file to cfg

    Parameters
    ----------
    cfg : dictionary
        Dictionary of features, if empty, a new dictionary is created
    json_path : string
        Path to json file
    '''

    #! TODO: if fea_dict is not empty, check if the feature is already in the dict
    #! check also for conflicts in the parameters, and function

    if json_path is None:
        json_path = vbi.__path__[0] + '/feature_extraction/features.json'

    dict_features = load_json(json_path)

    for domain in dict_features:
        for feat in dict_features[domain]:
            if dict_features[domain][feat]["use"] == "no":
                continue
            fea_dict = add_feature(fea_dict,
                              domain=domain,
                              name=feat,
                              features_path=features_path,
                              parameters=dict_features[domain][feat]["parameters"],
                              tag=dict_features[domain][feat]["tag"],
                              description=dict_features[domain][feat]["description"]
                              )

    return fea_dict
