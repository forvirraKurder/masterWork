from sympy import roots
import math
import numpy as np
import sympy as sp
import control
from sympy.physics.units import pebibytes

import src.utility_functions as uf
import src.dynamic
import src.linear_ps as dps_mdl


class EigFromY:
    def __init__(self, ps, model):
        self.ps = ps
        self.model = model

        # --- Buses: build a consistent index mapping ---
        bus_table = self.model['buses']
        bus_header = bus_table[0]
        name_col = bus_header.index('name')
        self.bus_names = [row[name_col] for row in bus_table[1:]]
        self.nbus = len(self.bus_names)
        self.bus_idx = {name: i for i, name in enumerate(self.bus_names)}

        # --- VSC (UIC) indexing: support multiple UIC devices ---
        self.vsc_devices = []  # list of dicts: {'name': ..., 'bus': ...}
        self.vsc_bus = None  # kept for backward compatibility (first device)
        self.vsc_name = None

        if 'vsc' in self.model and 'UIC_open' in self.model['vsc']:
            uic_table = self.model['vsc']['UIC_open']
            if len(uic_table) > 1:
                uic_header = uic_table[0]
                bus_col = uic_header.index('bus')
                name_col_uic = uic_header.index('name')

                # collect *all* UIC devices
                for row in uic_table[1:]:
                    name = row[name_col_uic]
                    bus = row[bus_col]
                    self.vsc_devices.append({'name': name, 'bus': bus})

                # for backward-compatibility keep the first one
                if self.vsc_devices:
                    self.vsc_name = self.vsc_devices[0]['name']
                    self.vsc_bus = self.vsc_devices[0]['bus']

        # --- Build static network Y (complex and dq) including lines + Z-loads ---
        self.Y_bus_complex, _ = self.build_Ybus_lines()
        self.Y_bus_lines_dq = self.build_Ybus_dq_lines()

        # Numerical transfer functions from inputs to outputs
        s = sp.symbols('s')
        self.inputs = [['UIC_open', 'v_t_d', 0], ['UIC_open', 'v_t_q', 0]]
        self.outputs = [['UIC_open',  'i_x', 0], ['UIC_open',      'i_y', 0]]
        self.numerical_tf = self.numeric_transfer_functions()
        self.analytical_tf = self.analytical_transfer_function(s)

        # Build symbolic Y_nodal_dq matrices
        self.Y_nodal_dq = self.build_sympy_Y_nodal_dq(self.numerical_tf)
        self.analytical_Y_nodal_dq = self.build_sympy_Y_nodal_dq(self.analytical_tf)

        # --- determinant-based eigenvalues (zeros of det(Y_nodal(s))) ---
        self.roots, self.det_expr, self.det_coeffs = self.roots_when_det_zero(
            s, self.Y_nodal_dq
        )
        (
            self.analytical_roots,
            self.analytical_det_expr,
            self.analytical_det_coeffs,
        ) = self.roots_when_det_zero(s, self.analytical_Y_nodal_dq)


    # -------------------------------------------------------------------------
    # Network admittance construction (complex + dq) with lines + loads
    # -------------------------------------------------------------------------

    def build_Ybus_lines(self):
        """
        Build complex Y_bus (nbus x nbus) from line data and Z-type loads.
        Lines are modeled with a π-equivalent, and Z-loads are added as shunts
        on the diagonal of Y.
        """
        data = self.model
        nbus = self.nbus
        Y = np.zeros((nbus, nbus), dtype=complex)

        # --- Line contributions ---
        line_table = data['lines']
        if len(line_table) > 1:
            line_header = line_table[0]
            name_col = line_header.index('name')
            fb_col = line_header.index('from_bus')
            tb_col = line_header.index('to_bus')
            length_col = line_header.index('length')
            Sn_col = line_header.index('S_n')
            Vn_col = line_header.index('V_n')
            unit_col = line_header.index('unit')
            R_col = line_header.index('R')
            X_col = line_header.index('X')
            B_col = line_header.index('B')

            for row in line_table[1:]:
                name = row[name_col]
                fb = row[fb_col]
                tb = row[tb_col]
                # length = row[length_col]
                # Sn     = row[Sn_col]
                # Vn     = row[Vn_col]
                unit = row[unit_col]
                R = row[R_col]
                X = row[X_col]
                B = row[B_col]

                if unit.lower() == 'pu':
                    r = R
                    x = X
                    b_total = B
                else:
                    # Implement conversion from Ohm/km to pu if needed
                    raise NotImplementedError("Non-pu line data conversion not implemented.")

                z = r + 1j * x
                if z == 0:
                    raise ZeroDivisionError(f"Line {name} has zero impedance.")

                y_series = 1 / z
                y_shunt = 1j * b_total / 2.0

                i = self.bus_idx[fb]
                j = self.bus_idx[tb]

                # Self-admittances
                Y[i, i] += y_series + y_shunt
                Y[j, j] += y_series + y_shunt

                # Mutual admittances
                Y[i, j] -= y_series
                Y[j, i] -= y_series

         #--- Load contributions (Z-type constant-impedance loads) ---
        loads: ['name', 'bus', 'P', 'Q', 'model']
        if 'loads' in data and len(data['loads']) > 1:
            load_table = data['loads']
            load_header = load_table[0]
            name_col = load_header.index('name')
            bus_col = load_header.index('bus')
            P_col = load_header.index('P')
            Q_col = load_header.index('Q')
            model_col = load_header.index('model')

            for row in load_table[1:]:
                name = row[name_col]
                bus = row[bus_col]
                P = row[P_col]
                Q = row[Q_col]
                model = row[model_col]

                if model.upper() == 'Z':
                    # Complex power S = P + jQ (load consumes P,Q > 0)
                    # For V = 1 pu, Y = S* (complex conjugate)
                    S = P + 1j * Q
                    Y_load = np.conjugate(S)  # = P - jQ

                    if bus not in self.bus_idx:
                        raise KeyError(f"Load {name} refers to unknown bus '{bus}'.")

                    k = self.bus_idx[bus]
                    Y[k, k] += Y_load

        # add voltage source large admittance when present
        if 'generators' in data and 'VoltageSource' in data['generators']:
            vs_table = data['generators']['VoltageSource']
            if len(vs_table) > 1:
                vs_header = vs_table[0]
                name_col = vs_header.index('name')
                bus_col = vs_header.index('bus')
                X_col = vs_header.index('X')

                for row in vs_table[1:]:
                    name = row[name_col]
                    bus = row[bus_col]
                    Xs = row[X_col]

                    if Xs == 0:
                        raise ZeroDivisionError(f"VoltageSource {name} has zero reactance X.")

                    # Impedance Z = jXs  ->  Y = 1/Z = -j / Xs
                    y_vs = -1j / Xs

                    if bus not in self.bus_idx:
                        raise KeyError(f"VoltageSource {name} refers to unknown bus '{bus}'.")

                    k = self.bus_idx[bus]
                    Y[k, k] += y_vs

        Y = np.round(Y, 3)  # rounds real & imaginary parts to 3 decimals

        return Y, self.bus_names

    def complex_to_dq_block(self, y):
        """
        Convert a complex admittance y = G + jB into a 2x2 dq block:
            [ G  -B ]
            [ B   G ]
        such that [i_d; i_q] = Y_dq * [v_d; v_q].
        """
        G = y.real
        B = y.imag
        return np.array([[G, -B],
                         [B, G]])

    def build_Ybus_dq_lines(self):
        """
        Build dq-frame Y_bus from lines + loads.
        For n buses, returns a (2n x 2n) real matrix with 2x2 blocks.
        """
        Y_complex = self.Y_bus_complex
        nbus = Y_complex.shape[0]

        Y_dq = np.zeros((2 * nbus, 2 * nbus))
        for i in range(nbus):
            for j in range(nbus):
                block = self.complex_to_dq_block(Y_complex[i, j])
                Y_dq[2 * i:2 * i + 2, 2 * j:2 * j + 2] = block

        return Y_dq

    # -------------------------------------------------------------------------
    # Transfer functions
    # -------------------------------------------------------------------------

    def numeric_transfer_functions(self):
        """
        Build transfer function matrix from inputs to outputs.
        Inputs and outputs are defined in self.inputs / self.outputs.
        Returns a control.TransferFunction MIMO object.
        """
        ps_lin = dps_mdl.PowerSystemModelLinearization(self.ps)
        inputs = self.inputs
        outputs = self.outputs
        ps_lin.linearize(inputs=inputs, outputs=outputs)
        self.ps_lin = ps_lin  # keep for later inspection if needed

        ss = control.ss(ps_lin.a, ps_lin.b, ps_lin.c, ps_lin.d)
        tf = control.ss2tf(ss)



        # Change sign of the TF from input 1 to output 0
        tf.num[0][1] *= -1  # or: tf.num[0][1] = -tf.num[0][1]
        tf.num[1][0] *= -1  # or: tf.num[1][0] = -tf.num[1][0]

        return tf

    def tf_to_sympy(self, tf_ij, s):
        """
        Convert a python-control TransferFunction element (SISO) to
        a SymPy expression in s.
        """
        # tf_ij.num and tf_ij.den are 2D lists (for MIMO),
        # each entry is a 1D list of coefficients (highest power first).
        num_coeffs = np.array(tf_ij.num[0][0], dtype=float)
        den_coeffs = np.array(tf_ij.den[0][0], dtype=float)

        # Build numerator polynomial
        num_poly = 0
        n = len(num_coeffs)
        for k in range(n):
            num_poly += num_coeffs[k] * s ** (n - k - 1)

        # Build denominator polynomial
        den_poly = 0
        m = len(den_coeffs)
        for k in range(m):
            den_poly += den_coeffs[k] * s ** (m - k - 1)

        return sp.simplify(num_poly / den_poly)

    # -------------------------------------------------------------------------
    # Build symbolic Y_nodal_dq(s) with TF block at the correct bus
    # -------------------------------------------------------------------------
    def build_sympy_Y_nodal_dq(self, tf):

        s = sp.symbols('s')

        # Convert numeric dq Y_bus to a SymPy matrix
        Y_num = self.Y_bus_lines_dq

        n = Y_num.shape[0]
        Y_sym = sp.Matrix(n, n, lambda i, j: sp.nsimplify(Y_num[i, j]))

        # If there are no UIC devices or no TF info, just return the static Y
        if not getattr(self, "vsc_devices", None) or tf is None:
            return Y_sym

        # Helper: ensure each tf[i,j] becomes a SymPy expression
        def tf_entry_to_sympy(tf_ij):
            # If it's already a SymPy expression, just return it
            if isinstance(tf_ij, sp.Expr):
                return tf_ij
            # Otherwise assume it is a python-control TransferFunction
            return self.tf_to_sympy(tf_ij, s)

        # Helper: pick the correct 2x2 TF block for a given device
        def get_tf_block(dev):
            # case 1: we got a dict of per-device TFs
            if isinstance(tf, dict):
                name = dev['name']
                if name not in tf:
                    raise KeyError(
                        f"No transfer function block provided for UIC '{name}'. "
                        "Either add it to the tf dict or pass a single 2x2 block "
                        "to be reused for all UICs."
                    )
                return tf[name]

            # case 2: a single 2x2 block is shared among all UIC devices
            return tf

        # Add a 2x2 dynamic admittance block at the bus of each UIC
        for dev in self.vsc_devices:
            bus_name = dev['bus']
            if bus_name not in self.bus_idx:
                raise KeyError(f"VSC bus '{bus_name}' not found in bus list.")

            k = self.bus_idx[bus_name]
            row0 = 2 * k
            col0 = 2 * k

            tf_block = get_tf_block(dev)

            for i in range(2):
                for j in range(2):
                    expr_ij = tf_entry_to_sympy(tf_block[i, j])
                    Y_sym[row0 + i, col0 + j] += expr_ij

        return Y_sym


    # -------------------------------------------------------------------------
    # Determinant roots and eigenvalue analysis
    # -------------------------------------------------------------------------

    def roots_when_det_zero(self, s, Y_nodal_obj):
        """
        Compute the eigenvalues as the zeros of det(Y_nodal(s)),
        following det(Y_nodal(s)) = 0.

        Implementation:
            - compute det(Y_nodal(s)) as a rational function,
            - write it as num(s) / den(s),
            - solve num(s) = 0 (denominator cleared).
        """
        # Ensure SymPy matrix

        Y = sp.Matrix(Y_nodal_obj.tolist())


        # Simplify each entry
        Y = Y.applyfunc(lambda e: sp.cancel(sp.together(e)))



        # Determinant as rational function
        det_expr = sp.cancel(sp.together(Y.det(method='berkowitz')))
        det_expr = sp.cancel(det_expr)

        # Split into numerator / denominator
        num, den = sp.fraction(det_expr)
        # Factor numerator to expose repeated factors, etc.
        num = sp.factor(num)

        # Convert numerator to polynomial in s
        poly = sp.Poly(num, s, domain='CC')
        coeffs = [complex(c.evalf()) for c in poly.all_coeffs()]

        # Strip any tiny leading numerical dust
        tiny = 1e-100
        while coeffs and abs(coeffs[0]) < tiny:
            print("coeffs before pop:")
            print(coeffs)
            coeffs.pop(0)

        roots_arr = np.roots(coeffs) if len(coeffs) > 1 else np.array([])

        if den != 1:
            poly_den = sp.Poly(den, s, domain='CC')
            coeffs_den = [complex(c.evalf()) for c in poly_den.all_coeffs()]

            while coeffs_den and abs(coeffs_den[0]) < tiny:
                coeffs_den.pop(0)

            roots_den = np.roots(coeffs_den) if len(coeffs_den) > 1 else np.array([])
        else:
            coeffs_den = []
            roots_den = np.array([])


        return roots_arr, det_expr, coeffs

    def analyze_roots(self, roots=None):
        """Compute damping ratio and frequency for each root s."""
        info = []
        for eig in roots:
            c = complex(eig)
            real_part = c.real
            imag_part = c.imag
            omega = math.hypot(real_part, imag_part)  # sqrt(Re^2 + Im^2)

            if omega == 0:
                zeta = float('inf')
            else:
                zeta = -real_part / omega

            freq_hz = imag_part / (2 * math.pi)
            info.append((eig, zeta, freq_hz))
        return info

    def pretty_print_eigs(self):
        root_info_test = self.analyze_roots(self.roots)
        for i, (eig, zeta, f) in enumerate(root_info_test):
            print(f"{i+1}: {eig:+.3e}  |  ζ={zeta:.3f}  |  f={f:.3f} Hz")

    def pretty_print_analytical_eigs(self):
        root_info_analytical = self.analyze_roots(self.analytical_roots)
        for i, (eig, zeta, f) in enumerate(root_info_analytical):
            print(f"{i+1}: {eig:+.3e}  |  ζ={zeta:.3f}  |  f={f:.3f} Hz")

    # -------------------------------------------------------------------------
    # Analytical transfer function
    # -------------------------------------------------------------------------

    def make_vsc_tf_block(self, s, Ki, xf, P, Q):
        """
        Build a 2x2 control.TransferFunction block for one VSC:
            [[Re(s),  Im(s)],
             [-Im(s), Re(s)]]
        for given parameters Ki, xf, P, Q.
        """
        w_0 = 2 * 3.14 * 50

        D = P ** 2 * Ki ** 2 * w_0 ** 2 * xf ** 2 + (Ki * w_0 + s * xf - Q * Ki * w_0 * xf) ** 2
        N1 = P * (Ki * w_0) ** 2
        N2 = P ** 2 * (Ki * w_0) ** 2 * xf + (Q * Ki * w_0 - s) * (Q * Ki * w_0 * xf - Ki * w_0 - s * xf)

        Re = N1 / D
        Im = N2 / D


        def symexpr_to_tf(expr, s_sym):
            """Convert a SymPy rational expression in s_sym to control.TransferFunction."""
            num_sym, den_sym = sp.fraction(sp.simplify(expr))
            num_sym = sp.expand(num_sym)
            den_sym = sp.expand(den_sym)

            num_poly = sp.Poly(num_sym, s_sym)
            den_poly = sp.Poly(den_sym, s_sym)

            # Convert coefficients to floats for control.tf
            num_coeffs = [float(c.evalf()) for c in num_poly.all_coeffs()]
            den_coeffs = [float(c.evalf()) for c in den_poly.all_coeffs()]

            return control.tf(num_coeffs, den_coeffs)

        tf_re = symexpr_to_tf(Re, s)
        tf_im = symexpr_to_tf(Im, s)

        return np.array([[tf_re,  tf_im],
                         [-tf_im, tf_re]], dtype=object)

    def analytical_transfer_function(self, s):
        """
        Return a dict mapping each VSC name -> 2x2 TF block:
            {
                'VSC1': [[Re1, Im1], [-Im1, Re1]],
                'VSC2': [[Re2, Im2], [-Im2, Re2]],
                ...
            }

        build_sympy_Y_nodal_dq already knows how to handle this dict
        and will put each block at the correct bus.
        """

        # 1) Define parameters for each VSC (manual, quick way).
        #    Make sure these keys match self.vsc_devices[i]['name'].
        vsc_params = {
            'UIC1': dict(Ki=0.01, xf=0.10),
            'UIC2': dict(Ki=0.01, xf=0.10),
        }

        tf_dict = {}
        index = 0

        for dev in self.vsc_devices:
            name = dev['name']

            # Get Ki, xf for this device; fall back to some default if missing
            Ki = vsc_params.get(name, {}).get('Ki')
            xf = vsc_params.get(name, {}).get('xf')

            # Get P, Q for this device.
            # Assumes your ps object has vsc[name] entries with _input_values['p_ref'], ['q_ref']
            P = self.ps.vsc['UIC_open']._input_values['p_ref'][index]
            Q = self.ps.vsc['UIC_open']._input_values['q_ref'][index]
            print(f"Building analytical TF for VSC '{name}': Ki={Ki}, xf={xf}, P={P}, Q={Q}")

            tf_block = self.make_vsc_tf_block(s, Ki, xf, P, Q)
            tf_dict[name] = tf_block
            index += 1

        return tf_dict
