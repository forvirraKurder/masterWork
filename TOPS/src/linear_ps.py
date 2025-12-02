import numpy as np
import src.utility_functions as utils


class PowerSystemModelLinearization:
    def __init__(self, ps):
        self.ps = ps

        self.n = self.ps.n_states
        self.eps = 1e-12
        self.linearization_ready = False
        self.eigenvalues_ready = False
        self.output_functions = []  # List to store output functions

        self.a = None  # n x n
        self.b = None  # n x m
        self.c = None  # p x n
        self.d = None  # p x m

        self.lev = np.empty((self.n,)*2, dtype=complex)
        self.rev = np.empty((self.n,)*2, dtype=complex)
        self.eigs = np.empty(self.n, dtype=complex)
        self.freq = np.empty(self.n)
        self.damping = np.empty(self.n)

    def linearize(self, t0=0, x0=np.array([]), inputs=np.array([]),
                  outputs=np.array([])):
        self.x0 = x0 if len(x0) > 0 else self.ps.x0
        self.a = utils.jacobian_num(lambda x: self.ps.ode_fun(t0, x), self.x0, eps=self.eps)

        # Prep output functions
        for col_idx, output in enumerate(outputs):
            model_type, output_string, index = output
            for model in self.ps.dyn_mdls:
                if model.__class__.__name__ == model_type and hasattr(model, output_string):
                    func = getattr(model, output_string)
                    self.output_functions.append(lambda x, v, f=func, idx=index: f(x, v)[idx])
                    break

        self.c = utils.jacobian_num(self.outputs, self.x0, eps=self.eps)

        if len(inputs) > 0:
            self.perturb_inputs(inputs)
        else:
            self.b = np.zeros((self.n, 0))

        self.linearization_ready = True

    def outputs(self, x):
        """
        Returns a vector valued function y(x,v) representing the outputs of the system.
        Should first compute the algebraic variables for a given x.
        :return:
        """
        vred = self.ps.solve_algebraic(0, x)
        y = np.zeros(len(self.output_functions))
        for i, func in enumerate(self.output_functions):
            y[i] = func(x, vred)
        return y

    def residues(self, mode_idx):
        if not self.eigenvalues_ready:
            self.eigenvalue_decomposition()
        return self.lev.dot(self.b)[[mode_idx], :]*self.c.dot(self.rev)[:, [mode_idx]]

    def eigenvalue_decomposition(self):
        if not self.linearization_ready:
            self.linearize()

        self.eigs, evs = np.linalg.eig(self.a)

        # Right/left rigenvectors (rev/lev)
        self.rev = evs
        self.lev = np.linalg.inv(self.rev)
        # self.damping = -self.eigs.real / abs(self.eigs)
        self.damping = np.divide(
            -self.eigs.real, abs(self.eigs),
            out=np.zeros_like(self.eigs.real)*np.nan,
            where=self.eigs.real != 0,
        )
        self.freq = self.eigs.imag / (2 * np.pi)

        self.eigenvalues_ready = True

    def perturb_inputs(self, inputs):
        ps = self.ps
        eps = self.eps
        b = np.zeros((ps.n_states, len(inputs)))
        d = np.zeros((len(self.output_functions), len(inputs)))

        for col_idx, inp in enumerate(inputs):
            model_type, input_string, index = inp
            b_column = np.zeros(ps.n_states)
            d_column = np.zeros(len(self.output_functions))
            for mdl in ps.dyn_mdls:
                if mdl.__class__.__name__ == model_type and hasattr(mdl, input_string):
                    input_values = mdl._input_values[input_string]

                    # Perturb the input positively
                    input_values[index] += eps
                    f_plus = ps.ode_fun(0, ps.x0)
                    y_plus = self.outputs(ps.x0)

                    # Perturb the input negatively
                    input_values[index] -= 2 * eps
                    f_minus = ps.ode_fun(0, ps.x0)
                    y_minus = self.outputs(ps.x0)

                    # Restore the original input value
                    input_values[index] += eps

                    # Compute the column of B for this input
                    b_column += (f_plus - f_minus) / (2 * eps)
                    d_column += (y_plus - y_minus) / (2 * eps)
            b[:, col_idx] = b_column  # Assign the computed column to B
            d[:, col_idx] = d_column  # Assign the computed column to D
        self.b = b
        self.d = d
        return b

    def perturb_states(self, outputs=None):
        ps = self.ps
        t = 0
        x = ps.x0.copy()
        v = ps.v0.copy()

        # Initialize the A and C matrices
        n_states = len(x)
        a = np.zeros((n_states, n_states))
        c = np.zeros((len(outputs), n_states)) if outputs else None

        # Loop through the state perturbations
        for j in range(n_states):
            dx = np.zeros_like(x)  # Reset dx for each state
            dx[j] = 1e-6  # Small perturbation

            # Solve algebraic variables for perturbed states
            v_plus = ps.solve_algebraic(t, x + dx)
            v_minus = ps.solve_algebraic(t, x - dx)

            # Compute A matrix (Jacobian of algebraic variables w.r.t. states)
            a[:, j] = (v_plus - v_minus) / (2 * dx[j])

            # Compute C matrix (Jacobian of outputs w.r.t. states) if outputs are provided
            if outputs:
                for i, (mdl_type, func_name, idx) in enumerate(outputs):
                    func = getattr(getattr(self.ps, mdl_type), func_name)
                    f_plus = func(x + dx, v_plus)[idx]
                    f_minus = func(x - dx, v_minus)[idx]
                    c[i, j] = (f_plus - f_minus) / (2 * dx[j])

        # Store the computed matrices
        self.a = a
        self.c = c
        return a, c

    def get_mode_idx(self, mode_type=['em', 'non_conj'], damp_threshold=1, freq_range=[0.1, 3], sorted=True):
        # Get indices of modes from specified criteria.
        eigs = self.eigs
        idx = np.ones(len(eigs), dtype=bool)
        if not isinstance(mode_type, list):
            mode_type = [mode_type]

        for mt in mode_type:
            if mt == 'em':
                idx *= (abs(eigs.imag) / (2 * np.pi) > freq_range[0]) & (abs(eigs.imag) / (2 * np.pi) < freq_range[1])
            if mt == 'non_conj':
                idx *= eigs.imag >= 0

        idx *= self.damping < damp_threshold

        idx = np.where(idx)[0]
        if sorted:
            idx = idx[np.argsort(self.damping[idx])]
        return idx

    def get_dominant_mode(self):
        em_idx = (0.1 < self.freq) & (self.freq < 2)
        return np.argmin(self.damping)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import src.plotting as dps_plt
    import src.dynamic as dps
    import casestudies.ps_data.k2a as model_data

    import importlib
    importlib.reload(dps)

    ps = dps.PowerSystemModel(model_data.load())
    ps.setup()
    ps.build_y_bus('lf')
    ps.power_flow()
    ps.init_dyn_sim()

    # Perform system linearization
    ps_lin = PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()


    # Plot eigenvalues
    dps_plt.plot_eigs(ps_lin.eigs)

    # Get mode shape for electromechanical modes
    mode_idx = ps_lin.get_mode_idx(['em'], damp_threshold=0.3)
    rev = ps_lin.rev
    mode_shape = rev[np.ix_(ps.gen['GEN'].state_idx_global['speed'], mode_idx)]

    # Plot mode shape
    fig, ax = plt.subplots(1, mode_shape.shape[1], subplot_kw={'projection': 'polar'})
    for ax_, ms in zip(ax, mode_shape.T):
        dps_plt.plot_mode_shape(ms, ax=ax_, normalize=True)

    plt.show()
