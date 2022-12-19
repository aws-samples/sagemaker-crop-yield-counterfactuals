import datetime
import io
import json
import os
import re
import warnings
from typing import List

import boto3
import numpy as np
import pandas as pd
import sagemaker
import shap
import spyndex
from matplotlib.cm import get_cmap
from matplotlib.colors import CenteredNorm, Colormap, rgb2hex

warnings.simplefilter(action="ignore")

s3 = boto3.resource("s3")

def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = (
            obj.key
            if local_dir is None
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        )
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == "/":
            continue
        bucket.download_file(obj.key, target)


def prepare_axis_request_manifests(
    request_start_time,
    request_end_time,
    week,
    year,
    fips,
    spectral_indices,
    manifest_filename,
    bucket_name,
):
    """Prepare a json file for each input parameters

    Note:
        This will help run multiple processing jobs in parallel
    """
    request_manifest = {
        "startime": request_start_time,
        "endtime": request_end_time,
        "week": week,
        "year": year,
        "fips": fips,
        "spectralindices": spectral_indices,
    }

    s3object = s3.Object(bucket_name, manifest_filename)

    s3object.put(Body=(bytes(json.dumps(request_manifest).encode("UTF-8"))))

    return True


def reformat_formula(idx):
    """Reformats the spyndex formula"""

    # mapping between standard convention and Sentinel-2
    params = {
        "A": "coastal",
        "B": "blue",
        "G": "green",
        "R": "red",
        "RE1": "b05",
        "RE2": "b06",
        "RE3": "b07",
        "N": "nir",
        "N2": "b8a",
        "S1": "swir16",
        "S2": "swir22",
    }

    #     """
    #     Additional index parameters:

    #     g: Gain factor (e.g. Used for EVI).
    #     L: Canopy background adjustment (e.g. Used for SAVI and EVI).
    #     C1: Coefficient 1 for the aerosol resistance term (e.g. Used for EVI).
    #     C2: Coefficient 2 for the aerosol resistance term (e.g. Used for EVI).
    #     cexp: Exponent used for OCVI.
    #     nexp: Exponent used for GDVI.
    #     alpha: Weighting coefficient used for WDRVI, BWDRVI and NDPI.
    #     beta: Calibration parameter used for NDSIns.
    #     gamma: Weighting coefficient used for ARVI.
    #     omega: Weighting coefficient used for MBWI.
    #     sla: Soil line slope.
    #     slb: Soil line intercept.
    #     PAR: Photosynthetically Active Radiation.
    #     k: Slope parameter by soil used for NIRvH2.
    #     lambdaN: NIR wavelength used for NIRvH2 and NDGI.
    #     lambdaR: Red wavelength used for NIRvH2 and NDGI.
    #     lambdaG: Green wavelength used for NDGI.
    #     """

    for cnt in spyndex.constants:
        params.update({cnt: spyndex.constants[cnt].default})

    formula = spyndex.indices[idx].formula
    delimiters = set(re.findall(r"[\+\-*/()]", formula))

    def split(delimiters, string, maxsplit=0):
        import re

        regex_pattern = "|".join(map(re.escape, delimiters))
        return set(re.split(regex_pattern, string, maxsplit))

    keys = split(delimiters, formula)

    formula = " {} ".format(" ".join(formula))
    formula = formula.replace(" . ", ".")

    for key in keys:
        if key in params.keys():
            formula = formula.replace(" {} ".format(" ".join(key)), " {} ".format(str(params[key])))

    return formula


def spectral_indices_equation_prep(index):
    """Prepares the bandmath equations"""

    if not isinstance(index, list):
        index = [index]

    names = list(spyndex.indices.keys())

    spectral_indices = []
    for idx in index:
        if idx not in names:
            raise Exception(f"{idx} is not a valid Spectral Index!")
        else:
            spectral_indices.append((idx, reformat_formula(idx)))

    return spectral_indices


def jdtodatestd(jdate):
    """
    Coverts Julian dates to Gregorian dates
    """
    fmt = "%Y%j"
    datestd = datetime.datetime.strptime(jdate, fmt).date()

    return datestd


def convert_dates(data):
    """
    Coverts Julian dates to Gregorian dates
    """

    data["day_v5"] = data["day_v5"].astype("int")
    data["day_v5_greg"] = data["year"].astype(str) + data["day_v5"].astype(str)
    data["day_v5_greg"] = data["day_v5_greg"].apply(jdtodatestd)

    data["day_sow"] = data["day_sow"].astype("int")
    data["day_sow_greg"] = data["year"].astype(str) + data["day_sow"].astype(str)
    data["day_sow_greg"] = data["day_sow_greg"].apply(jdtodatestd)

    return data


def heatmap_pandas(corr):

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.tril_indices_from(mask)] = True
    corr[mask] = np.nan

    # display the styler
    display(
        corr.style.background_gradient(cmap="coolwarm", axis=None, vmin=-1, vmax=1)
        .highlight_null(null_color="#f1f1f1")  # Color NaNs grey
        .set_precision(2)
    )


def centered_gradient(s: pd.Series, cmap: Colormap, false_css: str = "") -> List[str]:
    # Find center point
    center = 1.5 * s.median()
    # Create normaliser centered on median
    norm = CenteredNorm(vcenter=center)
    
    return [
        # Conditionally apply gradient to values above center only
        f"background-color: {rgb2hex(rgba)}" if row > center else false_css
        for row, rgba in zip(s, cmap(norm(s)))
    ]


def global_shap_importance(model, X):
    """Return a dataframe containing the features sorted by Shap importance
    Parameters
    ----------
    model : The tree-based model
    X : pd.Dataframe
         training set/test set/the whole dataset ... (without the label)
    Returns
    -------
    pd.Dataframe
        A dataframe containing the features sorted by Shap importance
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    cohorts = {"": shap_values}
    cohort_labels = list(cohorts.keys())
    cohort_exps = list(cohorts.values())

    for i in range(len(cohort_exps)):
        if len(cohort_exps[i].shape) == 2:
            cohort_exps[i] = cohort_exps[i].abs.mean(0)

    features = cohort_exps[0].data
    feature_names = cohort_exps[0].feature_names
    values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))])

    feature_importance = pd.DataFrame(
        list(zip(feature_names, sum(values))), columns=["features", "importance"]
    )

    feature_importance.sort_values(by=["importance"], ascending=False, inplace=True)

    return feature_importance
