"""
Utility Python Functions
By ALlen Diao (axdiao)

General purpose utility functions useful for multiple drivers and files
"""

import os

def config(attr):
    """
    Retrieves the appropriate attribute from the config.json file
    :return
        value of attribute in config.json
    """
    if not hasattr(config, 'config'):
        with open('config.json') as f:
            config.config = eval(f.read())
    node = config.config
    for part in attr.split('.'):
        node = node[part]
    return node

