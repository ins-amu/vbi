import json
import vbi 
import numpy as np

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
    Create a dictionary of features in given domain.

    Parameters
    ----------
    domain : string
        Domain of features to extract
    path : string
        Path to json file
    '''

    if json_path is None:
        json_path = vbi.__path__[0] + '/feature_extraction/features.json'

        if domain not in ['temporal', 'spectral', 'statistical', 'connectivity', None]:
            raise SystemExit(
                'Domain not valid. Please choose between: temporal, spectral, statistical, connectivity or None')
        dict_features = load_json(json_path)
        if domain is None:
            return dict_features
        else:
            return {domain: dict_features[domain]}

def get_features_by_tag(tag=None, json_path=None):
    '''
    Create a dictionary of features in given tag.

    Parameters
    ----------
    tag : string
        Tag of features to extract
    path : string
        Path to json file
    '''

    if path is None:
        path = vbi.__path__[0] + '/feature_extraction/features.json'

        if tag not in ['audio', 'eeg', 'ecg', None]:
            raise SystemExit('Tag not valid. Please choose between: audio, eeg, ecg or None')
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
                            features_tag[domain].update({feat: dict_features[domain][feat]})
                    elif js_tag == tag:
                        features_tag[domain].update({feat: dict_features[domain][feat]})
                except KeyError:
                    continue
        # To remove empty dicts
        return dict([[d, features_tag[d]] for d in list(features_tag.keys()) if bool(features_tag[d])])
