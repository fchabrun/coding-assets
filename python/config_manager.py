import argparse
import os
import json


def load_config_from_json_at_realpath(jsonpath: str) -> argparse.Namespace:
    with open(jsonpath, "r") as jsf:
        jsdata = json.load(jsf)
    return argparse.Namespace(**jsdata)


def load_config_from_json(wdir: str) -> argparse.Namespace:
    with open(os.path.join(wdir, "args_dict.json"), "r") as jsf:
        jsdata = json.load(jsf)
    return argparse.Namespace(**jsdata)


def save_config_to_json(wdir: str, cf: argparse.Namespace):
    with open(os.path.join(wdir, "args_dict.json"), "w") as jsf:
        json.dump(vars(cf), jsf, indent=4)


def print_config(cf: argparse.Namespace, name: str):
    print(f"{name}:")
    for key, value in vars(cf).items():
        print(f"    +{key}: {value}")


def impute_config_if_missing(cf: argparse.Namespace, key: str, value):
    cf_vars = vars(cf)
    if key not in cf_vars.keys():
        cf_vars[key] = value
