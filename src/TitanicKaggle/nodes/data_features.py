import pandas as pd
import math


def remove_bad_pat_id(preprocessed_patients, preprocessed_immunization):
    """
    Remove patients without immunizations records.
    """
    bad_pat_id = list(set(list(preprocessed_patients.pat_id.unique())) -
                      set(list(preprocessed_immunization.pat_id.unique())))
    return preprocessed_patients[~preprocessed_patients.pat_id.isin(bad_pat_id)]


def build_primary_table(df0, df1):
    """
    Join preprocessed_patients and preprocessed_immunization
    dataframes. Drop successful == False records.
    """
    df = remove_bad_pat_id(df0, df1)
    df = pd.merge(df0, df1, on='pat_id', how='outer')
    df = df[df.successful == True]
    df.drop(['successful', 'reason_unsuccesful'], 1)
    df = df.reset_index(drop=True)
    return df


def extract_temp_df(df, pid):
    """
    Extract all records for a patient from the primary table
    and sort by immunization date.
    """
    df = df[df.pat_id == pid].sort_values('im_date')
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


def build_feature_table():
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
    feature_df_list = []
    for p in pat_id:
        pat_dict = dict()
        pat_dict['pat_id'] = p

        pat_df = extract_temp_df(df_outer, p)

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
