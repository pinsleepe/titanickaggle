import numpy as np
import pandas as pd


def remove_unnamed_column(df) -> pd.DataFrame:
    """
    Remove columns leftover by df to csv conversion.
    """
    return df.drop('Unnamed: 0', 1)


def create_facility_df(df) -> pd.DataFrame:
    """
    remove patient info 
    sort fac_id in ascending order 
    drop columns with Nan (assumption: there is at least one fac_id
    row with the whole information)
    dropping ALL duplicte values but the first
    reset index
    sanity check: fac_id != NaN
    """
    df = df.drop(['pat_id', 'dob', 'gender'], 1)
    df.sort_values(by=['fac_id'], inplace=True)
    df = df.dropna()
    df.drop_duplicates(subset="fac_id",
                       keep='first',
                       inplace=True)
    df = df.reset_index(drop=True)
    assert df.fac_id.isnull().sum() == 0
    return df


def drop_null_gender(df):
    """
    Drop all records without gender field and reset index.
    """
    df.dropna(subset=['gender'], inplace=True)
    df = df.reset_index(drop=True)
    return df


def fill_missing_facid(df0, df1) -> pd.DataFrame:
    """
    Since lat and long all unique for all facilities we 
    can use one to match out fac_id. Find index of NaN fac_id and 
    search facility ID dataframe for match on longitude.
    If there is no long, as we will assume no geo location
    was captured, remove records.
    df0: patient dataframe
    df1: facilities dataframe
    """
    ind = df0['fac_id'].index[df0['fac_id'].apply(np.isnan)]
    empty_idx = []
    for i in ind:
        long0 = df0.loc[i].long
        try:
            facid = df1[df1['long'] == long0]['fac_id'].values[0]
            df0.loc[i, 'fac_id'] = facid
        except IndexError:
            empty_idx.append(i)
    if empty_idx:
        df0 = df0.drop(df0.index[empty_idx])
        df0 = df0.reset_index(drop=True)
    return df0


def remove_facilities(df0, df1) -> pd.DataFrame:
    """
    Remove facilities from patient dataframe that are not present in 
    facilities dataframe
    df0: patient dataframe
    df1: facilities dataframe
    """
    bad_facid = list(set(list(df0['fac_id'].unique())) - set(list(df1['fac_id'])))
    idx = [df0[df0['fac_id'] == i].index.tolist() for i in bad_facid]
    bad_idx = [item for sublist in idx for item in sublist]

    df0.drop(df0.index[bad_idx], inplace=True)
    df0 = df0.reset_index(drop=True)
    return df0


def fill_missing_values(df0, df1) -> pd.DataFrame:
    """
    Find index of NaN values. For each index read fac_id. 
    Using that fac_id read long, lat, region and district 
    from facilities dataframe and write them to patient dataframe.
    df0: patient dataframe
    df1: facilities dataframe
    """
    ind = df0[df0.isnull().any(axis=1)].index
    for i in ind:
        facid = df0.loc[i].fac_id
        df0.loc[i, 'long'] = df1[df1.fac_id == facid].long.values[0]
        df0.loc[i, 'lat'] = df1[df1.fac_id == facid].lat.values[0]
        df0.loc[i, 'region'] = df1[df1.fac_id == facid].region.values[0]
        df0.loc[i, 'district'] = df1[df1.fac_id == facid].district.values[0]
    return df0


def preprocess_patients(df0, df1) -> pd.DataFrame:
    """
    Bring all functions to create pre-processed df
    """
    df = remove_unnamed_column(df0)
    df = drop_null_gender(df)
    df = fill_missing_facid(df, df1)
    df = remove_facilities(df, df1)
    df = fill_missing_values(df, df1)
    return df
