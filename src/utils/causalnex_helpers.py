import bisect
from typing import Dict

import numpy as np


def quantile_discretiser(data, num_buckets):
    """Allows the discretisation of numeric data. Only discrete distributions supported in Baysian networks."""

    x = data.values.flatten()
    x.sort()

    bucket_width = 1.0 / num_buckets
    quantiles = [bucket_width * (n + 1) for n in range(num_buckets - 1)]
    numeric_split_points = np.quantile(x, quantiles)

    data = np.digitize(data.values, numeric_split_points, right=False)
    data = data.astype("int8")

    return data, numeric_split_points


def generate_dag_constraints(mappping):

    """
    Takes in the mapping file with the crop phenology staging and returns constraints for the NOTEARS algortihm.
    https://papers.nips.cc/paper/8157-dags-with-no-tears-continuous-optimization-for-structure-learning.pdf

    1. list of nodes banned from being a child of any other nodes
    2. list of nodes banned from being a parent of any other nodes
    3. list of edges(from, to) not to be included in the graph.

    """

    nodes_list = list(mappping.variable.unique())

    # list of nodes banned from being a child of any other nodes
    tabu_child = list(mappping[mappping.level == mappping.level.min()].variable.values)

    # list of nodes banned from being a parent of any other nodes

    tabu_parents = []

    nodes_matrix = sorted(
        [(mappping[mappping.variable == node]["level"].values[0], node) for node in nodes_list]
    )

    # list of edges(from, to) not to be included in the graph.
    # this is done in order to forbid reverse temporal modeling 
    # observations collected at the later stage should not inform earlier stages

    tabu_edges = [
        (node_i, node_j)
        for idx, node_i in nodes_matrix
        for idy, node_j in nodes_matrix
        if idy <= idx and node_i != node_j
    ]

    # one node cannot inform more than 2 steps ahead
    tabu_edges_forwards = [
        (node_i, node_j)
        for idx, node_i in nodes_matrix
        for idy, node_j in nodes_matrix
        if idy > idx + 1 and node_i != node_j
    ]

    tabu_edges.extend(tabu_edges_forwards)

    # append the prohibited atmospheric edges 
    # within this model will disregard the correlation between temperature, radiation and precipitation
    
    atmospheric_nodes = [
        node for node in nodes_list if "tmean" in node or "rain" in node or "rad" in node
    ]
    tabu_edges_atmospheric = [
        (node_i, node_j)
        for node_i in atmospheric_nodes
        for node_j in atmospheric_nodes
        if node_i != node_j
    ]

    tabu_edges.extend(tabu_edges_atmospheric)

    # no node can inform on atmospheric variables
    tabu_edges_atmospheric_rest = [
        (node_i, node_j)
        for node_i in nodes_list
        for node_j in atmospheric_nodes
        if node_i != node_j
    ]

    tabu_edges.extend(tabu_edges_atmospheric_rest)

    tabu_edges = list(set(tabu_edges))

    return (tabu_edges, tabu_child, tabu_parents, nodes_list, nodes_matrix)


import bisect


def discretiser_inverse_transform(
    map_thresholds, request=True, request_nodes=[], response_nodes=[]
):
    """
    1. Encodes the request from real values to discretised values
    2. Decodes the reponse back to real value ranges

    """
    # encode request
    if request:

        discretised_nodes = []

        if request_nodes:
            for node in request_nodes:

                if node[0] in list(map_thresholds.keys()):

                    ind = bisect.bisect_left(map_thresholds[node[0]], node[1])

                else:
                    print(
                        f"nodes not in the graph \n Please select one from this list: {list(map_thresholds.keys())}"
                    )

                discretised_nodes.append((node[0], ind))

            return discretised_nodes

    # decode response
    else:

        decoded_nodes = []

        if response_nodes:
            for node in response_nodes:

                if node[0] in list(map_thresholds.keys()):

                    n_buckets = len(map_thresholds[node[0]])

                    if node[1] > n_buckets:

                        print(
                            f"bucket value: {node[1]} larger than the max {len(discretiser.map_thresholds[node[0]]) + 1}"
                        )
                        break

                    elif node[1] == 0:

                        split_point = map_thresholds[node[0]][0]
                        split_point = f"<{split_point:.2f}"

                    elif node[1] == n_buckets:

                        split_point = map_thresholds[node[0]][-1]
                        split_point = f">{split_point:.2f}"

                    else:

                        split_point = map_thresholds[node[0]][node[1]]
                        split_point_left = map_thresholds[node[0]][node[1] - 1]
                        split_point = f" >={split_point_left:.2f} | <{split_point:.2f}"

                    decoded_nodes.append((node[0], split_point))

                else:
                    print(
                        f"nodes not in the graph \n Please select one from this list: {list(map_thresholds.keys())}"
                    )

        return decoded_nodes


def format_inference_output(output):
    """
    Format ouptut by converting the marginals probabilities into buckets

    """
    output_decoded = []
    query_def = []

    if isinstance(output, Dict):
        method = output["method"]
    else:
        method = output[0]["method"]

    if method == "query":
        for out in output:
            bucket = max(out["marginals"], key=out["marginals"].get)
            output_decoded.append((out["target"], int(bucket)))
            query_def.append({"target": out["target"], "query": out["observation"]})

    else:

        bucket = max(output["marginals-before"], key=output["marginals-before"].get)
        output_decoded.append((output["target"], int(bucket)))

        bucket = max(output["marginals-after"], key=output["marginals-after"].get)
        output_decoded.append((output["target"], int(bucket)))
        query_def.append(
            {
                "target": output["target"],
                "query": output["query"],
                "interventions": output["interventions"],
            }
        )

    return output_decoded, query_def, method
