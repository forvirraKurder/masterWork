import pandapower as pp
import pandas as pd
from dataclasses import dataclass
#  The power system data structure is defined here
#  Also contains one converter from pandapower networks to our dataclasses


@dataclass
class PowerSystem:
    lines: []  # List of Line objects
    nodes: []  # List of Node objects

    def to_df(self):
        """ Convert the lists to pandas DataFrames """
        # Enables tabular view, and vectorized operations
        # Lists are easier, though!

        lines = pd.DataFrame([x.__dict__ for x in self.lines])
        nodes = pd.DataFrame([x.__dict__ for x in self.nodes])
        return lines, nodes


@dataclass
class Line:
    from_bus: int
    to_bus: int
    r: float  # resistance in pu
    x: float  # reactance in pu
    b: float  # susceptance in pu


@dataclass
class Node:
    idx: int  # Node index/identifier
    type: int  # 1: PQ, 2: PV, 3: Slack
    v: complex  # Complex voltage
    s: complex  # Power injection
    vm_set: float = None  # Voltage setpoint for PV nodes


def pp2ntnu(net):
    """ Convert a pandapower network to a "NTNU" PowerSystem object.
    Uses the pandapower converter to matpower format before converting to our format.

    :param net: pandapower network
    :return: PowerSystem object
    """
    lines = []
    nodes = []
    ps = pp.converter.to_ppc(net, init="flat")  # Convert the pandapower network to matpower/pypower format

    for i, line in enumerate(ps['branch']):
        lines.append(Line(*line[0:5]))
    for i, node in enumerate(ps['bus']):
        idx = node[0]
        type = node[1]
        s = -(node[2]+ 1j*node[3])/net.sn_mva
        # v = node[7] + 1j*node[8]  # Known solution
        v = 1+0j
        nodes.append(Node(idx, type, v, s))

    for i, gen in enumerate(ps['gen']):
        idx = int(gen[0])
        nodes[idx].s += (gen[1] + 1j*gen[2])/net.sn_mva
        nodes[idx].vm_set = gen[5]

    return PowerSystem(lines, nodes)


def three_bus_example():
    lines = [Line(0, 1, 0.02, 0.1, 0.0), Line(1, 2, 0.02, 0.1, 0.0), Line(0, 2, 0.02, 0.1, 0.0)]
    nodes = [Node(0, 3, 1.0, 0.0, 1), Node(1, 2, 1.0, 1.0, 1), Node(2, 1, 1.0, -1.0,  1)]
    return PowerSystem(lines, nodes)


def main():
    net = pp.networks.mv_oberrhein()  # Load an example pandapower network
    ps = pp2ntnu(net)  # Convert the pandapower network to our data structure
