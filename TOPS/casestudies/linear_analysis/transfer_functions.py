import src.dynamic as dps
import src.linear_ps as dps_mdl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import control


def main():
    import casestudies.ps_data.ma_4bus as model_data
    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)

    inputs = [
        ['GEN', 'P_m', 0],
        ['GEN', 'E_f', 0]
    ]

    # Define output functions externally
    outputs = [
        ['GEN', 'angle', 0],
        ['GEN', 'v_t_abs', 0]
    ]

    ps_lin.linearize(inputs=inputs, outputs=outputs)
    ss = control.ss(ps_lin.a, ps_lin.b, ps_lin.c, ps_lin.d)
    tf = control.ss2tf(ss)

    control.bode(tf, np.logspace(-3, 3, 500))
    plt.show()

if __name__ == '__main__':
    main()