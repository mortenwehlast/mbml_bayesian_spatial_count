import pandas as pd
import yaml


def load_data():
    with open('config/vars/input_variables.yml') as inp_vars:
        input_variables = yaml.safe_load(inp_vars)['NO_NANS']
    
    df_traffic = pd.read_csv("data/bym_nyc_study.csv")
    df_census = pd.read_csv("data/nyc_census_tracts.csv")
    df = df_traffic.merge(df_census, how="left", left_on="census_tract", right_on="CensusTract")

    # Get input, output and groups by counties.
    X = df[input_variables]
    y = df['ped_injury_5to18']
    k = df['County'].factorize()[0]

    return X, y, k