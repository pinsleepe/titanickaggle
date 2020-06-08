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


def create_label_table(feature_table):
    """
    Create dataframe with label column (based on feature table).
    """
    feature_table['label'] = feature_table.apply(lambda x: assign_label(x.opv_by_4mths,
                                                                        x.dtp_by_4mths,
                                                                        x.opv_by_6mths,
                                                                        x.dtp_by_6mths), axis=1)
    return feature_table
