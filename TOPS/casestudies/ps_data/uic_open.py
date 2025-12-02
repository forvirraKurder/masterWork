import numpy as np

def load():
    return {
        'base_mva': 1,
        'f': 50,
        'slack_bus': 'B1',

        'buses': [
            ['name',    'V_n'],
            ['B1',         1],
            ['B2',         1],
            ['B3',         1],
        ],

        'lines': [
            ['name',    'from_bus', 'to_bus',   'length',   'S_n',  'V_n',     'unit',     'R',    'X',      'B'],
            ['L1-2',          'B1',              'B2',          1,            1,       1,        'pu',   0.0,   0.1,       0],
            ['L1-3',          'B1',              'B3',          1,            1,       1,        'pu',   0.0,   0.1,       0],
        ],

        #'loads': [
        #    ['name', 'bus', 'P',  'Q', 'model'],
        #    ['L1',       'B3', 0.2,  0.2,       'Z'],
        #],

        'generators': {
            'VoltageSource': [
                ['name',    'bus',  'S_n',    'V_n',     'P',     'V_ref',      'X'],
                ['IB',         'B1',       1,             1,       1,          1,      1e-6],
            ],
        },

        'vsc': {
            'UIC_open': [
                ['name', 'bus', 'S_n', 'V_n', 'p_ref',    'q_ref',      'Ki',    'xf',  'v_ref'],
                ['UIC1',  'B2',     1,         1,      0.1,         0.0,     0.01,    0.1,          1],
                ['UIC2', 'B3',     1,         1,      0.1,         0.0,     0.01,    0.1,          1],
            ],
        },
    }