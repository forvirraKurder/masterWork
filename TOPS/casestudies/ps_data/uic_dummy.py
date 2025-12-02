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
            #['B3',         1],
        ],

        'lines': [
            ['name',  'from_bus', 'to_bus',   'length',   'S_n',  'V_n',  'unit',        'R',     'X',   'B'],
            ['L1-2',              'B1',        'B2',             1,         1,       1,    'pu',          0,      0.1,   0],
            #['L2-3',             'B2',        'B3',             1,         1,       1,    'pu',          0,      0.1,   0],
        ],

       #'loads': [
       #    ['name', 'bus', 'P', 'Q', 'model'],
       #    ['L1',    'B2',       1.2,   0.2, 'Z'],
       #     ],

        #'generators': {
        #    'GEN': [
        #        ['name', 'bus', 'S_n', 'V_n', 'P', 'V', 'H', 'D', 'X_d', 'X_q', 'X_d_t', 'X_q_t', 'X_d_st', 'X_q_st',
        #         'T_d0_t', 'T_q0_t', 'T_d0_st', 'T_q0_st'],
        #        ['IB', 'B2', 10000000, 20, -600, 1, 999999., 0, 1.8, 1.7, 0.3, 0.3, 0.2, 0.2, 8.0, 0.6, 0.05, 0.05],
        #    ],
        #},

        'vsc': {
            'UIC': [
                ['name', 'bus', 'S_n', 'V_n', 'p_ref',  'q_ref',      'Ki',   'xf',  'k_p', 'v_ref'],
                ['UIC1',    'B1',       1,       1,         0,         0,    0.05,   0.01,      1,          1],
                ['UIC2',   'B2',       1,       1,      0.6,         0,    0.01,      0.1,      1,          1],
            ],
        },
}
