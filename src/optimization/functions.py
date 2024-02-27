import pandas as pd


def opt_value(df, lambda_=0.7, var1='precision', var2='shap_q2'):
    """
    Sorts the DataFrame based on a weighted sum of 'precision' and 'shap_q2'.

    Parameters:
    - df: pandas.DataFrame containing the columns 'precision' and 'shap_q2'.
    - lambda_: float, the weight for the 'precision' in the weighted sum.
               'shap_q2' is weighted by (1 - lambda_).

    Returns:
    - sorted_df: pandas.DataFrame sorted by the calculated 'opt_value'.
    """
    # Calculate the weighted sum and assign it to a new column 'opt_value'
    df["opt_value"] = df[var1] * lambda_ + df[var2] * (1 - lambda_)

    # Sort the DataFrame by 'opt_value' in descending order to have the best values at the top
    sorted_df = df.sort_values("opt_value", ascending=False)

    return sorted_df
