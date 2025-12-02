from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import src.dynamic as dps
import src.solvers as dps_sol
import importlib

from dyn_models.gen import VoltageSource

importlib.reload(dps)


if __name__ == '__main__':

    # region Model loading and initialisation stage
    import casestudies.ps_data.uic_ib as model_data
    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)  # Load into a PowerSystemModel object

    ps.power_flow()  # Power flow calculation
    # Print load flow solution
    for bus, v in zip(ps.buses['name'], ps.v_0):
        print(f'{bus}: {np.abs(v):.2f} /_ {np.angle(v):.2f}')

    ps.init_dyn_sim()  # Initialise dynamic variables
    x0 = ps.x0.copy()  # Initial states

    # List of machine parameters for easy access
    gen_pars =ps.vsc['UIC'].par # Access like this: S_n_gen = genpars['S_n']

    t = 0
    t_end = 20  # Simulation time

    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x0, t_end, max_step=5e-3)  # solver
    # endregion

    # region Runtime variables
    result_dict = defaultdict(list)
    # Additional plot variables
    P_m_stored = []
    P_e_stored = []
    E_f_stored = []
    v_bus = []
    I_stored = []
    t_stored = []
    modal_stored = []

    event_flag = True
    # endregion

    # Simulation loop starts here!
    while t < t_end:
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        if 1 < t < 1.04:
            ps.y_bus_red_mod[1,1] = 1e1

        else:
            ps.y_bus_red_mod[1,1] = 0

        if t > 2 and event_flag:
            event_flag = False
            ps.gen['VoltageSource']._input_values['angle'][0] += 0.01*100*np.pi*5e-3
        # region Store variables
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]
        # Store additional variables

        P_e_stored.append(ps.vsc['UIC'].p_e(x, v).copy())

        I_gen = ps.y_bus_red_full[0, 1] * (v[0] - v[1])
        I_stored.append(np.abs(I_gen))
        t_stored.append(t)
        # endregion

    # Convert dict to pandas dataframe
    result = pd.DataFrame(result_dict, columns=pd.MultiIndex.from_tuples(result_dict))
    # region Plotting of pe s
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    time_axis = np.arange(len(P_e_stored))
    vi_uic = result[('UIC1', 'vix')]+1j*result[('UIC1', 'viy')]
    axs[0].plot(t_stored, P_e_stored)
    axs[0].set_ylabel('VSC Electrical Power Output P_e (p.u.)')
    axs[0].set_title('VSC Electrical Power Output P_e')
    axs[1].plot(t_stored, I_stored)
    axs[1].set_ylabel('VSC Current Magnitude (p.u.)')
    axs[1].set_title('VSC Current Magnitude')
    axs[1].set_xlabel('Time step')
    axs[2].plot(t_stored, np.angle(vi_uic), label='vi')
    plt.tight_layout()
    plt.show()

    # endregion
