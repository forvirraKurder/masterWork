# --- top of file (module scope) ---
import numpy as np
import sympy as sp
import src.dynamic as dps
import src.linear_ps as dps_mdl
import control
import src.eigFromY as eigY

s = sp.symbols('s')  # global symbol

def build_Ybus_lines(data):
    """
    Build complex Y_bus (nbus x nbus) from line data only.
    """
    # --- bus indexing ---
    bus_rows = data['buses'][1:]  # skip header
    bus_names = [row[0] for row in bus_rows]
    nbus = len(bus_names)
    bus_idx = {name: i for i, name in enumerate(bus_names)}

    Y = np.zeros((nbus, nbus), dtype=complex)

    # --- line contributions ---
    for row in data['lines'][1:]:  # skip header
        name, fb, tb, length, S_n, V_n, unit, R, X, B = row

        # If unit is per-unit, R, X, B are already in pu.
        # Otherwise you could convert from Ohm/km here (not needed for your example).
        if unit.lower() == 'pu':
            r = R
            x = X
            b_total = B
        else:
            # Example conversion from Ohm/km to pu (if ever needed)
            # Z_base = (V_n**2) / S_n   # V_n in kV, S_n in MVA
            # r = (R * length) / Z_base
            # x = (X * length) / Z_base
            # b_total = (B * length) * Z_base  # depending on B units
            raise NotImplementedError("Non-pu line data conversion not implemented.")

        z = r + 1j * x
        if z == 0:
            raise ZeroDivisionError(f"Line {name} has zero impedance.")

        y_series = 1 / z                 # series admittance
        y_shunt = 1j * b_total / 2.0     # half shunt at each end (π-model)

        i = bus_idx[fb]
        j = bus_idx[tb]

        # Self-admittances
        Y[i, i] += y_series + y_shunt
        Y[j, j] += y_series + y_shunt

        # Mutual admittances
        Y[i, j] -= y_series
        Y[j, i] -= y_series

    return Y, bus_names


def complex_to_dq_block(y):
    """
    Convert a complex admittance y = G + jB into a 2x2 dq block:
        [ G  -B ]
        [ B   G ]
    such that [i_d; i_q] = Y_dq * [v_d; v_q].
    """
    G = y.real
    B = y.imag
    return np.array([[G, -B],
                     [B,  G]])


def build_Ybus_dq_lines(data):
    """
    Build dq-frame Y_bus from lines only.
    For n buses, returns a (2n x 2n) real matrix with 2x2 blocks.
    """
    Y_complex, bus_names = build_Ybus_lines(data)
    nbus = Y_complex.shape[0]

    Y_dq = np.zeros((2 * nbus, 2 * nbus))

    for i in range(nbus):
        for j in range(nbus):
            block = complex_to_dq_block(Y_complex[i, j])
            Y_dq[2*i:2*i+2, 2*j:2*j+2] = block

    return Y_dq, bus_names

def tf_to_sympy(tf_ij, s):
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
        num_poly += num_coeffs[k] * s**(n - k - 1)

    # Build denominator polynomial
    den_poly = 0
    m = len(den_coeffs)
    for k in range(m):
        den_poly += den_coeffs[k] * s**(m - k - 1)

    return sp.simplify(num_poly / den_poly)

def roots_when_det_zero(Y_nodal_obj, s):
    Y = sp.Matrix(Y_nodal_obj.tolist())

    # Make every entry a simple rational in s
    Y = Y.applyfunc(lambda e: sp.cancel(sp.together(e)))

    # Fraction-free determinant tends to preserve exact cancellations better
    det_expr = sp.cancel(sp.together(Y.det(method='berkowitz')))

    # Cancel common factors in/out *again* (helps if det() introduced fractions)
    det_expr = sp.cancel(det_expr)

    # Extract numerator only (zeros of det)
    num, den = sp.fraction(det_expr)
    num = sp.factor(num)  # expose repeated factors

    # Convert to polynomial in s
    poly = sp.Poly(num, s, domain='CC')
    coeffs = [complex(c.evalf()) for c in poly.all_coeffs()]

    # Guard: strip tiny numerical dust
    tiny = 1e-12
    while coeffs and abs(coeffs[0]) < tiny:
        coeffs.pop(0)

    roots = np.roots(coeffs) if len(coeffs) > 1 else np.array([])
    return roots, det_expr, coeffs

def analyze_roots(roots):
    """Optional: compute damping ratio and frequency for each root s."""
    info = []
    for eig in roots:
        real_part = eig.real
        imag_part = eig.imag
        omega = np.hypot(real_part, imag_part)  # sqrt(Re^2 + Im^2)
        zeta = (-real_part / omega) if omega != 0 else float('inf')
        freq_hz = imag_part / (2*np.pi)
        info.append((eig, zeta, freq_hz))
    return info

def main():
    import casestudies.ps_data.uic_dummy as model_data
    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)

    inputs = [
        ['UIC', 'v_t_d', 0], ['UIC', 'v_t_q', 0]
    ]
    outputs = [
        ['UIC', 'i_x', 0], ['UIC', 'i_y', 0]
    ]

    ps_lin.linearize(inputs=inputs, outputs=outputs)
    ss = control.ss(ps_lin.a, ps_lin.b, ps_lin.c, ps_lin.d)
    tf = control.ss2tf(ss)

    print(ps_lin.d)

    # Print the transfer function matrix
    print("Converter transfer function matrix (dq frame):")
    for i in range(tf.noutputs):
        row_str = " | ".join(f"{tf[i,j]}" for j in range(tf.ninputs))
        print(row_str)

    # Convert the 2x2 TF matrix to SymPy admittance elements
    Ydd = tf_to_sympy(tf[0,0], s)  # i_d / v_d
    Ydq = tf_to_sympy(tf[0,1], s)  # i_d / v_q
    Yqd = tf_to_sympy(tf[1,0], s)  # i_q / v_d
    Yqq = tf_to_sympy(tf[1,1], s)  # i_q / v_q

    Y_nodal, bus_names = build_Ybus_dq_lines(model)
    Y_nodal = Y_nodal.astype(object)   # <-- important

    print(Y_nodal)

    # Add converter Norton admittance at bus B1 (index 0)
    # block structure for bus 1 dq is:
    # [0,0] -> (d,d), [0,1] -> (d,q)
    # [1,0] -> (q,d), [1,1] -> (q,q)

    Y_nodal[0, 0] += Ydd
    Y_nodal[0, 1] += Ydq
    Y_nodal[1, 0] += Yqd
    Y_nodal[1, 1] += Yqq


    print("\nY_nodal symbolic matrix:")
    for row in Y_nodal:
        row_str = " | ".join(f"{elem}" for elem in row)
        print(row_str)

    s_roots, det_Y, det_coeffs = roots_when_det_zero(Y_nodal, s)

    # Optional: print summary with damping & frequency
    root_info = analyze_roots(s_roots)
    print("\nSystem eigenvalues (det(Y_nodal)=0):")
    for i, (eig, zeta, f) in enumerate(root_info):
        print(f"{i}: {eig:+.6e}  |  ζ={zeta:.4f}  |  f={f:.6f} Hz")


    #manual matrix creation for testing

    Y_test = np.array([[1/s + 1, -1],
                       [-1, 1/s + 1]], dtype=object)
    s_roots_test, det_Y_test, det_coeffs_test = roots_when_det_zero(Y_test, s)
    print("\nTest matrix eigenvalues (det(Y_test)=0):")
    root_info_test = analyze_roots(s_roots_test)
    for i, (eig, zeta, f) in enumerate(root_info_test):
        print(f"{i}: {eig:+.6e}  |  ζ={zeta:.4f}  |  f={f:.6f} Hz")

    P = 0.0
    Q = 0.0
    Ki = 0.01
    xf = 0.1
    w_0 = 2*np.pi*50

    D = P**2 *Ki**2 *w_0**2*xf**2 + (Ki*w_0 + s*xf - Q*Ki*w_0*xf)**2
    N1 = P*Ki**2*w_0**2
    N2 = P**2 *Ki**2 *w_0**2*xf + (Q*Ki*w_0*xf -s)*(-Ki*w_0 - s*xf + Q*Ki*w_0*xf)
    Re = 0
    Im = N2 / D

    Y_nodal[0, 0] += Re
    Y_nodal[0, 1] += Im
    Y_nodal[1, 0] -= Im
    Y_nodal[1, 1] += Re

    Y_nodal[2, 2] += 0.5
    Y_nodal[2, 3] += 0.5
    Y_nodal[3, 2] -= 0.5
    Y_nodal[3, 3] += 0.5

    s_roots_test, det_Y_test, det_coeffs_test = roots_when_det_zero(Y_nodal, s)
    print("\nTest matrix eigenvalues (det(Y_test)=0):")
    root_info_test = analyze_roots(s_roots_test)
    for i, (eig, zeta, f) in enumerate(root_info_test):
        print(f"{i}: {eig:+.6e}  |  ζ={zeta:.4f}  |  f={f:.6f} Hz")

    ki = 0.01
    xf = 0.1
    w0 = 2 * np.pi * 50

    P = 0.4
    Q = 0.2

    # inner-loop parameters
    a = ki * w0 * (Q - 1 / xf)
    b = ki * w0 * P
    c = ki * w0 / xf

    # G(s) and B(s)

    D = (s - a) ** 2 + b ** 2
    B = (1 / xf) - (c * (s - a)) / (xf * D)
    G = (b * c) / (xf * D)

    Y_nodal, bus_names = build_Ybus_dq_lines(model)
    Y_nodal = Y_nodal.astype(object)  # <-- important

    Y_nodal[0, 0] += G
    Y_nodal[0, 1] += B
    Y_nodal[1, 0] -= B
    Y_nodal[1, 1] += G

    Y_nodal[2, 2] += 0.5
    Y_nodal[2, 3] += 0.5
    Y_nodal[3, 2] -= 0.5
    Y_nodal[3, 3] += 0.5


    P = 0.6
    Q = 0.6

    # inner-loop parameters
    a = ki * w0 * (Q - 1 / xf)
    b = ki * w0 * P
    c = ki * w0 / xf

    # G(s) and B(s)
    D = (s - a) ** 2 + b ** 2
    B = (1 / xf) - (c * (s - a)) / (xf * D)
    G = (b * c) / (xf * D)

    Y_nodal[4, 4] += G
    Y_nodal[4, 5] += B
    Y_nodal[5, 4] -= B
    Y_nodal[5, 5] += G

    s_roots_test, det_Y_test, det_coeffs_test = roots_when_det_zero(Y_nodal, s)
    print("\nTest matrix eigenvalues (det(Y_test)=0):")
    root_info_test = analyze_roots(s_roots_test)
    for i, (eig, zeta, f) in enumerate(root_info_test):
        print(f"{i}: {eig:+.6e}  |  ζ={zeta:.4f}  |  f={f:.6f} Hz")



if __name__ == '__main__':
    main()

