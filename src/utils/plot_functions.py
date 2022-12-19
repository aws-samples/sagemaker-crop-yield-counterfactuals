import warnings
from typing import List, Tuple

from causalnex.plots import plot_structure
from causalnex.structure import StructureModel
from IPython.display import Image

warnings.filterwarnings("ignore")  # silence warnings


def plot_pretty_structure(
    g: StructureModel,
    edges_to_highlight: Tuple[str, str],
    default_weight: float = 0.2,
    weighted: bool = False,
):
    """
    Utility function to plot our networks in a pretty format

    Args:
        g: Structure model (directed acyclic graph)
        edges_to_highlight: List of edges to highlight in the plots
        default_weight: Default edge weight
        weighted: Whether the graph is weighted

    Returns:
        a styled pygraphgiz graph that can be rendered as an image
    """
    graph_attributes = {
        "splines": "spline",  # use splies
        "ordering": "out",
        "ratio": "fill",  # control the size of the image
        "size": "16,9!",  # set the size of the final image
        "fontcolor": "#FFFFFFD9",
        "bgcolor": "#1A1B23",
        "fontname": "Helvetica",
        "fontsize": 24,
        "labeljust": "c",
        "labelloc": "c",
        "pad": "1,1",
        "nodesep": 0.8,
        "ranksep": ".5 equally",
    }
    # Making all nodes hexagonal with black coloring
    node_attributes = {
        node: {
            "shape": "hexagon",
            "width": 2.2,
            "height": 2,
            "fillcolor": "#1A1B23",
            "penwidth": "5",
            "color": "#4a90e2d9",
            "fontsize": 20,
            "labelloc": "c",
            "labeljust": "c",
        }
        for node in g.nodes
    }
    # Customising edges
    if weighted:
        edge_weights = [(u, v, w if w else default_weight) for u, v, w in g.edges(data="weight")]
    else:
        edge_weights = [(u, v, default_weight) for u, v in g.edges()]

    edge_attributes = {
        (u, v): {
            "penwidth": w * 10 + 2,  # Setting edge thickness
            "weight": int(w),  # Higher "weight"s mean shorter edges
            "arrowsize": 2 - 2.0 * w,  # Avoid too large arrows
            "arrowtail": "dot",
            "color": "#DF5F00" if ((u, v) in set(edges_to_highlight)) else "#888888",
        }
        for u, v, w in edge_weights
    }
    return plot_structure(
        g,
        prog="dot",
        graph_attributes=graph_attributes,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
    )
