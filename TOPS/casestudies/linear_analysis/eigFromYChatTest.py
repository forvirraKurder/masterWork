import src.dynamic as dps
from TOPS.casestudies.ps_data import uic_ib
from TOPS.casestudies.ps_data import uic_open
import src.modal_analysis as dps_mdl
from eigFromYChat import EigFromY
import sympy as sp

s = sp.symbols('s')  # global symbol

def main():
    model = uic_ib.load()
    model_open = uic_open.load()
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    ps_open = dps.PowerSystemModel(model=model_open)
    ps_open.init_dyn_sim()
    eigFromY = EigFromY(ps_open, model_open)

    # Perform system linearization for comparison
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()

    Y_nodal = eigFromY.Y_nodal_dq
    print("\nY_nodal symbolic matrix:")
    if isinstance(Y_nodal, sp.MatrixBase):
        for i in range(Y_nodal.rows):
            row_str = " | ".join(str(Y_nodal[i, j]) for j in range(Y_nodal.cols))
            print(row_str)
    else:
        # fallback if it's still a numpy array for some reason
        for row in Y_nodal:
            row_str = " | ".join(str(elem) for elem in row)
            print(row_str)

    Y_nodal_analytical = eigFromY.analytical_Y_nodal_dq
    print("\nY_nodal analytical symbolic matrix:")
    if isinstance(Y_nodal_analytical, sp.MatrixBase):
        for i in range(Y_nodal_analytical.rows):
            row_str = " | ".join(str(Y_nodal_analytical[i, j]) for j in range(Y_nodal_analytical.cols))
            print(row_str)
    else:
        for row in Y_nodal_analytical:
            row_str = " | ".join(str(elem) for elem in row)
            print(row_str)

    print("\nEigenvalues from EigFromY:")
    print(eigFromY.pretty_print_eigs())

    print("\nEigenvalues from EigFromY analytical:")
    print(eigFromY.pretty_print_analytical_eigs())

    #print eigenvalues from PowerSystemModelLinearization for comparison with only 3 significant digits
    print("\nEigenvalues from PowerSystemModelLinearization:")
    for eig in ps_lin.eigs:
        print(f"{eig:.3f}")

if __name__ == "__main__":
    main()