import numpy as np
import pandas as pd


# ------------- PATIENTS


def remove_unnamed_column_patients(patients) -> pd.DataFrame:
    """
    Remove columns leftover by df to csv conversion.
    """
    return patients.drop('Unnamed: 0', 1)


def create_facility(patients) -> pd.DataFrame:
    """
    remove patient info
    sort fac_id in ascending order
    drop columns with Nan (assumption: there is at least one fac_id
    row with the whole information)
    dropping ALL duplicte values but the first
    reset index
    sanity check: fac_id != NaN
    """
    df = patients.drop(['pat_id', 'dob', 'gender'], 1)
    df.sort_values(by=['fac_id'], inplace=True)
    df = df.dropna()
    df.drop_duplicates(subset="fac_id",
                       keep='first',
                       inplace=True)
    df = df.reset_index(drop=True)
    assert df.fac_id.isnull().sum() == 0
    return df


def drop_null_gender(patients) -> pd.DataFrame:
    """
    Drop all records without gender field and reset index.
    """
    patients.dropna(subset=['gender'], inplace=True)
    patients = patients.reset_index(drop=True)
    return patients


def fill_missing_facid(patients, facilities) -> pd.DataFrame:
    """
    Since lat and long all unique for all facilities we
    can use one to match out fac_id. Find index of NaN fac_id and
    search facility ID dataframe for match on longitude.
    If there is no long, as we will assume no geo location
    was captured, remove records.
    df0: patient dataframe
    df1: facilities dataframe
    """
    ind = patients['fac_id'].index[patients['fac_id'].apply(np.isnan)]
    empty_idx = []
    for i in ind:
        long0 = patients.loc[i].long
        try:
            facid = facilities[facilities['long'] == long0]['fac_id'].values[0]
            patients.loc[i, 'fac_id'] = facid
        except IndexError:
            empty_idx.append(i)
    if empty_idx:
        patients = patients.drop(patients.index[empty_idx])
        patients = patients.reset_index(drop=True)
    return patients


def remove_facilities(patients, facilities) -> pd.DataFrame:
    """
    Remove facilities from patient dataframe that are not present in
    facilities dataframe
    df0: patient dataframe
    df1: facilities dataframe
    """
    bad_facid = list(set(list(patients['fac_id'].unique())) - set(list(facilities['fac_id'])))
    idx = [patients[patients['fac_id'] == i].index.tolist() for i in bad_facid]
    bad_idx = [item for sublist in idx for item in sublist]

    patients.drop(patients.index[bad_idx], inplace=True)
    df0 = patients.reset_index(drop=True)
    return df0


def fill_missing_values(patients, facilities) -> pd.DataFrame:
    """
    Find index of NaN values. For each index read fac_id.
    Using that fac_id read long, lat, region and district
    from facilities dataframe and write them to patient dataframe.
    df0: patient dataframe
    df1: facilities dataframe
    """
    ind = patients[patients.isnull().any(axis=1)].index
    for i in ind:
        facid = patients.loc[i].fac_id
        patients.loc[i, 'long'] = facilities[facilities.fac_id == facid].long.values[0]
        patients.loc[i, 'lat'] = facilities[facilities.fac_id == facid].lat.values[0]
        patients.loc[i, 'region'] = facilities[facilities.fac_id == facid].region.values[0]
        patients.loc[i, 'district'] = facilities[facilities.fac_id == facid].district.values[0]
    return patients


def preprocess_patients(patients, facilities) -> pd.DataFrame:
    """
    Bring all functions to create pre-processed df
    """
    df = remove_unnamed_column_patients(patients)
    df = drop_null_gender(df)
    df = fill_missing_facid(df, facilities)
    df = remove_facilities(df, facilities)
    df = fill_missing_values(df, facilities)
    return df


# ------------- IMMUNIZATION

def remove_unnamed_column_immunization(immunization) -> pd.DataFrame:
    """
    Remove columns leftover by df to csv conversion.
    """
    return immunization.drop('Unnamed: 0', 1)


def keep_successful(immunization)-> pd.DataFrame:
    """
    Remove unsuccessful immunizations.
    """
    df = immunization[immunization.successful == True]
    return df


def remove_unsuccesful_columns(immunization):
    """
    Remove unsuccessful reason and succesful columns
    """
    return immunization.drop(['successful', 'reason_unsuccesful'], 1)


def remove_records(patient, immunization) -> pd.DataFrame:
    """
    Remove records from immunization dataframe that correspond to unknown patients.
    df0: patient dataframe
    df1: immunization dataframe
    """
    pat_list = list(patient['pat_id'].unique())
    idx = [immunization[immunization['pat_id'] == i].index.tolist() for i in pat_list]
    good_idx = [item for sublist in idx for item in sublist]

    immunization = immunization.loc[good_idx]
    immunization = immunization.reset_index(drop=True)
    return immunization


def preprocess_immunization(immunization, preprocessed_patients) -> pd.DataFrame:
    """
    Bring all functions to create pre-processed df
    """
    df = remove_unnamed_column_immunization(immunization)
    df = keep_successful(df)
    df = remove_unsuccesful_columns(df)
    return remove_records(preprocessed_patients, df)


# ------------- PRIMARY TABLE

def remove_bad_pat_id(preprocessed_patients, preprocessed_immunization):
    """
    Remove patients without immunizations records.
    """
    bad_pat_id = list(set(list(preprocessed_patients.pat_id.unique())) -
                      set(list(preprocessed_immunization.pat_id.unique())))
    return preprocessed_patients[~preprocessed_patients.pat_id.isin(bad_pat_id)]


def build_primary_table(clean_patients, preprocessed_immunization):
    """
    Join preprocessed_patients and preprocessed_immunization
    dataframes. Drop successful == False records.
    """
    df = pd.merge(clean_patients, preprocessed_immunization, on='pat_id', how='outer')
    df = df.reset_index(drop=True)
    return df
