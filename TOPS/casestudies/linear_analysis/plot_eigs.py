import src.dynamic as dps
import src.modal_analysis as dps_mdl
import src.plotting as dps_plt
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    import casestudies.ps_data.uic_dummy as model_data
    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    # Perform system linearization
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()

    # Plot eigenvalues
    dps_plt.plot_eigs(ps_lin.eigs)

    # Get mode shape for electromechanical modes
    mode_idx = ps_lin.get_mode_idx(['all', 'non_conj'], damp_threshold=10000)
    print(f"mode_idx: {mode_idx}")  # Debug print
    rev = ps_lin.rev
    labels = ps.vsc['UIC'].par['name']
    mode_shape = rev[np.ix_(ps.vsc['UIC'].state_idx_global['vix'], mode_idx)]

    # Plot mode shape only if there are modes to plot
    n_modes = mode_shape.shape[1]
    if n_modes == 0:
        print("No modes found for the given criteria. Skipping mode shape plot.")
    else:
        fig, ax = plt.subplots(1, n_modes, subplot_kw={'projection': 'polar'})
        # make ax iterable even when n_modes == 1
        axes = ax if isinstance(ax, np.ndarray) else [ax]

        for ax_, ms in zip(axes, mode_shape.T):
            dps_plt.plot_mode_shape(ms, ax=ax_, normalize=True)
            mode_shape_max = ms[np.argmax(np.abs(ms))]
            print(abs(ms) / np.abs(mode_shape_max))

    # Colourmap is [red, blue, green, purple, ...]
    plt.show(block=True)

    print("eigenvalues:")
    for eig in ps_lin.eigs:
        print(eig)
