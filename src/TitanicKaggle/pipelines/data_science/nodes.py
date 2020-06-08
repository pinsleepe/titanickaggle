import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder

# ------------- FEATURE TABLE


def extract_temp_df(primary_table, pid):
    """
    Extract all records for a patient from the primary table
    and sort by immunization date.
    pid: patient id
    """
    df = primary_table[primary_table.pat_id == pid].sort_values('im_date')
    df['im_date'] = pd.to_datetime(df['im_date'], format='%Y-%m-%d')
    df['dob'] = pd.to_datetime(df['dob'], format='%Y-%m-%d')
    return df


def date2weeks(im_date, dob):
    """
    im_date, dob in datetime format
    """
    return math.floor(((im_date - dob) / np.timedelta64(1, 'W')))


def feature_first_vaccine(df):
    """
    First vaccine, accepts temporarys dataframe for
    a patient.
    """
    idx = df['im_date'].idxmin()
    return df.vaccine.loc[idx]


def feature_enrollment_age(df):
    """
    Enrollment age in weeks.
    """
    return math.floor(((df.im_date.min() - df.dob.min()) / np.timedelta64(1, 'W')))


def feature_exit_age(df):
    """
    Exit age in weeks.
    """
    return math.floor(((df.im_date.max() - df.dob.min()) / np.timedelta64(1, 'W')))


def feature_opv_by_4mths(df):
    """
    Number of OPV vaccines received by 4 months
    4 months == 16 weeks
    """
    return df[(df.vaccine == 'OPV') & (df.im_date <= 16)].shape[0]


def feature_opv_by_6mths(df):
    """
    Number of OPV vaccines received by 4 months
    6 months == 24 weeks
    """
    return df[(df.vaccine == 'OPV') & (df.im_date <= 24)].shape[0]


def feature_dtp_by_4mths(df):
    """
    Number of OPV vaccines received by 4 months
    4 months == 16 weeks
    """
    return df[(df.vaccine == 'DTP') & (df.im_date <= 16)].shape[0]


def feature_dtp_by_6mths(df):
    """
    Number of OPV vaccines received by 4 months
    6 months == 24 weeks
    """
    return df[(df.vaccine == 'DTP') & (df.im_date <= 24)].shape[0]


def build_feature_table(primary_table):
    """
    Features:
        Gender
        Facility
        Region
        First vaccine type
        Enrollment age in weeks
        Exit age in weeks
        Number of OPV vaccines received by 4 months
        Number of OPV vaccines received by 6 months
        Number of DTP vaccines received by 4 months
        Number of DTP vaccines received by 6 months
    """
    pat_id = list(primary_table.pat_id.unique())
    feature_df_list = []
    for p in pat_id:
        pat_dict = dict()
        pat_dict['pat_id'] = p

        pat_df = extract_temp_df(primary_table, p)

        facility = pat_df.iloc[-1].fac_id
        pat_dict['facility'] = facility

        region = pat_df.iloc[-1].region
        pat_dict['region'] = region

        gender = pat_df.iloc[-1].gender
        pat_dict['gender'] = gender

        vaccine = feature_first_vaccine(pat_df)
        pat_dict['first_vaccine'] = vaccine

        pat_df['im_date'] = pat_df.apply(lambda x: date2weeks(x.im_date, x.dob), axis=1)
        pat_dict['enrollment_age'] = pat_df.im_date.min()
        pat_dict['exit_age'] = pat_df.im_date.max()

        pat_dict['opv_by_4mths'] = feature_opv_by_4mths(pat_df)
        pat_dict['opv_by_6mths'] = feature_opv_by_6mths(pat_df)
        pat_dict['dtp_by_4mths'] = feature_dtp_by_4mths(pat_df)
        pat_dict['dtp_by_6mths'] = feature_dtp_by_6mths(pat_df)

        feature_df_list.append(pat_dict)

    return pd.DataFrame(feature_df_list)

# ------------- LABEL TABLE


def assign_label(x0, x1, x2, x3):
    """
    Assign `high` and `low` label to each patient based on a simple logic.
    """
    if (x0 < 4) & (x1 < 3) & (x2 < 4) & (x3 < 3):
        label = 'high'
    elif (x0 == 4) & (x3 < 3):
        label = 'high'
    elif (x3 == 3) & (x2 < 4):
        label = 'high'
    else:
        label = 'low'
    return label


def cast2category(df):
    """
    Cast categorical data from object type to category.
    """
    df['region'] = df['region'].astype('category')
    df['gender'] = df['gender'].astype('category')
    df['first_vaccine'] = df['first_vaccine'].astype('category')
    return df


def create_label_table(feature_table):
    """
    Create dataframe with label column (based on feature table) and
    cast categorical data from object type to category.
    """
    df = cast2category(feature_table)
    df['label'] = df.apply(lambda x: assign_label(x.opv_by_4mths,
                                                  x.dtp_by_4mths,
                                                  x.opv_by_6mths,
                                                  x.dtp_by_6mths), axis=1)
    return df


# ------------- MODEL TABLE

def encode_categories(label_table):
    """
    Convert each value in a column to a number.
    """
    lb_make = LabelEncoder()
    label_table['region_code'] = lb_make.fit_transform(label_table['region'])
    label_table['gender_code'] = lb_make.fit_transform(label_table['gender'])
    label_table['first_vaccine_code'] = lb_make.fit_transform(label_table['first_vaccine'])
    return label_table


def model_input_table(label_table):
    """
    Create table for the model training.
    """
    df = label_table[['facility', 'first_vaccine_code', 'gender_code', 'region_code', 'dtp_by_4mths',
                      'opv_by_4mths', 'enrollment_age', 'label']]
    return df


def create_model_table(label_table):
    """
    Create table for the model training.
    """
    df = encode_categories(label_table)
    return model_input_table(df)
