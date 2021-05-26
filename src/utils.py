import pandas as pd
import numpy as np
import yaml
import pickle


def load_data(impute=False, new_split=False):
    with open("config/vars/input_variables.yml") as inp_vars:
        if impute:
            input_variables = yaml.safe_load(inp_vars)["ALL"]
        else:
            input_variables = yaml.safe_load(inp_vars)["NO_NANS"]

    df_traffic = pd.read_csv("data/bym_nyc_study.csv")
    df_census = pd.read_csv("data/nyc_census_tracts.csv")
    df = df_traffic.merge(df_census, how="left", left_on="census_tract", right_on="CensusTract")

    # Get input, output and groups by counties.
    X = df[input_variables]
    if impute:
        X = X.fillna(X.median()).values
    y = df["ped_injury_5to18"].values
    k_maps = df["County"].factorize()
    k = k_maps[0]
    maps = k_maps[1]

    if new_split:
        # Get indeces for dividing data into training and test set.
        train_perc = 0.8  # percentage of training data
        split_point = int(train_perc * len(y))
        perm = np.random.permutation(len(y))
        ix_train = perm[:split_point]
        ix_test = perm[split_point:]
        with open("config/vars/train_test_ix.pkl", "wb") as f:
            pickle.dump({"ix_train": ix_train, "ix_test": ix_test}, f)
    else:
        # Get indeces for dividing data into training and test set.
        with open("config/vars/train_test_ix.pkl", "rb") as f:
            train_test_ixs = pickle.load(f)
            ix_train = train_test_ixs["ix_train"]
            ix_test = train_test_ixs["ix_test"]

    X_train = X[ix_train, :]
    X_test = X[ix_test, :]
    k_train = k[ix_train]
    k_test = k[ix_test]
    y_train = y[ix_train]
    y_test = y[ix_test]

    return (X_train, X_test), (y_train, y_test), (k_train, k_test), maps
