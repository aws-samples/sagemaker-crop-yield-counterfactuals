import glob
import json
import logging
import multiprocessing
import os

import networkx as nx
from causalnex.inference import InferenceEngine
from causalnex.network import BayesianNetwork
from pgmpy.models import BayesianNetwork as pgmpy_bn

JSON_CONTENT_TYPE = "application/json"

log_format = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(format=log_format)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def model_fn(model_dir):

    logger.info("Loading the Bayesian Network")
    node_states_path = os.path.join(model_dir, "models/node_states.json")
    structure_path = os.path.join(model_dir, "models/bn_structure.gml")
    model_path = os.path.join(model_dir, "models/bayesian_model.bif")

    print(glob.glob(f"{model_dir}/*/*"))

    try:
        # load the bayesian network model
        model = pgmpy_bn.load(model_path, filetype="bif")

        logger.info("bayesian network model loaded successfully")

    except Exception as e:
        error_msg = f"=== Error loading model: {model_path}  ==="
        logger.error(error_msg)
        raise e

    try:
        # load the node states
        with open(node_states_path, "r") as fp:
            node_states_dict = json.load(fp)
        node_states_dict = {
            k: {int(i): j for i, j in v.items()} for k, v in node_states_dict.items()
        }

        logger.info("Node states loaded successfully")

    except Exception as e:
        error_msg = f"=== Error loading node states: {node_states_path}  ==="
        logger.error(error_msg)
        raise e

    try:
        # load the DAG structure
        g = nx.read_gml(structure_path)

        logger.info("DAG structure loaded successfully")

    except Exception as e:
        error_msg = f"=== Error loading model structure: {structure_path}  ==="
        logger.error(error_msg)
        raise e

    try:
        # initilise the model
        bn = BayesianNetwork(g)
        bn._node_states = node_states_dict
        bn._model = model

        # An InferenceEngine provides methods to query marginals based on observations
        # and make interventions (Do-Calculus) on a BayesianNetwork.
        ie = InferenceEngine(bn)

        logger.info("Inference Engine initialised successfully")

    except Exception as e:
        error_msg = f"=== Error initializing the model  ==="
        logger.error(error_msg)
        raise e

    return ie


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):

    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data

    else:
        raise Exception("Requested unsupported ContentType in Accept: " + content_type)
        return


def predict_fn(input_object, model):

    print("request: {}".format(input_object))

    target = input_object["target"]

    if input_object["method"] == "query":

        # query the marginals with a list of observations
        pseudo_observation = [obs for obs in input_object["observations"]]
        marginals_multi = model.query(
            pseudo_observation,
            parallel=True,
            num_cores=multiprocessing.cpu_count() - 1,
        )

        response_marginals = []
        for i, obs in enumerate(pseudo_observation):

            marginals = marginals_multi[i][target]

            marginals = {str(k): v for k, v in marginals.items()}

            response_marginals.append(
                {
                    "method": input_object["method"],
                    "target": target,
                    "observation": obs,
                    "marginals": marginals,
                }
            )

        logger.info(f"Marginal of observed states \n {response_marginals}")

        output = response_marginals

    elif input_object["method"] == "do_calculus":

        # observed states of nodes in the Bayesian Network
        intervention_query = input_object["intervention_query"]

        # distribution before intervention
        marginals_before = model.query(intervention_query)[target]

        marginals_before = {str(k): v for k, v in marginals_before.items()}

        logger.info(
            f"Marginal of observed states {intervention_query} for {target} before intervention \n {marginals_before}"
        )

        # apply an intervention to all nodes in the data, updating its distribution using a do operator
        for node in input_object["interventions"]:
            model.do_intervention(node[0], node[1])

        # examining the effect of that intervention by querying marginals
        marginals_after = model.query(intervention_query)[target]

        marginals_after = {str(k): v for k, v in marginals_after.items()}

        logger.info(
            f"Marginal of observed states {intervention_query} for {target} after intervention \n {marginals_before}"
        )

        for node in input_object["interventions"]:
            model.reset_do(node[0])

        response_marginals = {
            "method": input_object["method"],
            "target": target,
            "query": intervention_query,
            "interventions": input_object["interventions"],
            "marginals-before": marginals_before,
            "marginals-after": marginals_after,
        }

        output = response_marginals

    else:
        raise Exception(f"Unsupported method type {input_object['method']}")
        return

    return output


def output_fn(prediction, accept=JSON_CONTENT_TYPE):

    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction), accept

    raise Exception("Requested unsupported ContentType in Accept: " + accept)
