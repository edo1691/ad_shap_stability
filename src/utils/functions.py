def generate_estimators_max_features(total_features, steps, min_value=2):
    """
    Generate a list of feature estimator counts based on percentage steps.

    Parameters:
    - total_features: int, total number of features.
    - steps: int, how many steps (percentages) to include up to 100%.
    - min_value: int, minimum value for any step, default is 2.

    Returns:
    - List of integers representing feature estimator counts at various percentages of the total,
      with a minimum value enforced.
    """
    percentages = [i / steps for i in range(1, steps + 1)]  # Generate percentages based on the number of steps
    estimators = [max(min_value, int(total_features * pct)) for pct in percentages]  # Calculate estimators and enforce minimum value
    return estimators