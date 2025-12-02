from powerdata import pp2ntnu, three_bus_example
import pandapower.networks as networks


net = networks.case14()  # Load an example pandapower network
ps = pp2ntnu(net)  # Convert the pandapower network to our format

#ps = three_bus_example()  # Load a simple 3-bus example


def solve(ps):
    """ A power flow solver using the Newton-Raphson method.
    Computes the complex node voltages that satisfies the power flow injections.

    :param ps: PowerSystem object
    """

    #  Your code here

    pass
