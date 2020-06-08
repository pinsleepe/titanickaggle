import math
from typing import Dict, List
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import ShuffleSplit
import pandas as pd


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
    df['label'] = df['label'].astype('category')
    return df


def create_label_table(feature_table):
    """
    Create dataframe with label column (based on feature table) and
    cast categorical data from object type to category.
    """
    feature_table['label'] = feature_table.apply(lambda x: assign_label(x.opv_by_4mths,
                                                  x.dtp_by_4mths,
                                                  x.opv_by_6mths,
                                                  x.dtp_by_6mths), axis=1)
    df = cast2category(feature_table)
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
    label_table['label_code'] = lb_make.fit_transform(label_table['label'])
    return label_table


def model_input_table(label_table):
    """
    Create table for the model training.
    """
    df = label_table[['facility', 'first_vaccine_code', 'gender_code', 'region_code', 'dtp_by_4mths',
                      'opv_by_4mths', 'enrollment_age', 'label_code']]
    return df


def create_model_table(label_table):
    """
    Create table for the model training.
    """
    df = encode_categories(label_table)
    return model_input_table(df)


def split_data(data: pd.DataFrame, parameters: Dict) -> List:
    """
    Splits data into training and test sets.
    """
    X = data[
        [
            'facility',
            'first_vaccine_code',
            'gender_code',
            'region_code',
            'dtp_by_4mths',
            'opv_by_4mths',
            'enrollment_age',
        ]
    ].values
    y = data["label_code"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    return [X_train, X_test, y_train, y_test]

# ------------- MODEL


def parameter_tuning_randomized_search(X_train, y_train):
    """

    """
    n_estimators = [int(x) for x in np.linspace(start=20, stop=100, num=5)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 50, num=5)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    n_jobs = [4]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'n_jobs': n_jobs}

    # Create the base model to tune
    rfc = RandomForestClassifier(random_state=8)
    # Definition of the random search
    random_search = RandomizedSearchCV(estimator=rfc,
                                       param_distributions=random_grid,
                                       n_iter=50,
                                       scoring='accuracy',
                                       cv=3,
                                       verbose=1,
                                       random_state=8,
                                       n_jobs=4)
    # Fit the random search model
    random_search.fit(X_train, y_train)

    logger = logging.getLogger(__name__)
    logger.info("The best hyperparameters from Random Search are:")
    logger.info(random_search.best_params_)
    logger.info("The mean accuracy of a model with these hyperparameters is:")
    logger.info(random_search.best_score_)
    # best model
    best_rfc = random_search.best_estimator_

    return best_rfc


def parameter_tuning_grid(X_train, y_train):
    """

    """
    bootstrap = [False]
    max_depth = [30, 40, 50]
    max_features = ['sqrt']
    min_samples_leaf = [1, 2, 4]
    min_samples_split = [5, 10, 15]
    n_estimators = [800]
    n_jobs = [4]

    param_grid = {
        'bootstrap': bootstrap,
        'max_depth': max_depth,
        'max_features': max_features,
        'min_samples_leaf': min_samples_leaf,
        'min_samples_split': min_samples_split,
        'n_estimators': n_estimators,
        'n_jobs': n_jobs
    }

    # Create a base model
    rfc = RandomForestClassifier(random_state=8)
    # Manually create the splits in CV
    cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=8)
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rfc,
                               param_grid=param_grid,
                               scoring='accuracy',
                               cv=cv_sets,
                               verbose=1,
                               n_jobs=4)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    logger = logging.getLogger(__name__)
    logger.info("The best hyperparameters from Grid Search are:")
    logger.info(grid_search.best_params_)
    logger.info("The mean accuracy of a model with these hyperparameters is:")
    logger.info(grid_search.best_score_)

    # best model
    best_rfc = grid_search.best_estimator_

    return best_rfc


def model_fit_and_performance(best_rfc, X_train, y_train, X_test, y_test):
    """
    For performance analysis the the classification report
    and the accuracy on both training and test data
    """
    best_rfc.fit(X_train, y_train)
    rfc_pred = best_rfc.predict(X_test)

    logger = logging.getLogger(__name__)
    logger.info("The training accuracy is: ")
    logger.info(accuracy_score(y_train, best_rfc.predict(X_train)))
    logger.info("The test accuracy is: ")
    logger.info(accuracy_score(y_test, rfc_pred))
    logger.info("Classification report")
    logger.info(classification_report(y_test, rfc_pred))

    return rfc_pred
