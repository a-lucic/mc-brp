import pandas as pd
import numpy as np
import random
import math


def get_tukeys_fences(dataset, feature):
    q1 = np.quantile(dataset[:, feature], 0.25)
    q3 = np.quantile(dataset[:, feature], 0.75)
    tf1 = q3 - 1.5 * (q3 - q1)
    tf2 = q3 + 1.5 * (q3 - q1)
    return tf1, tf2


# Create random perturbation
def random_perturbation(lower_bound, upper_bound):
    perturb = random.randint(lower_bound, upper_bound)
    return perturb


def get_perturbations(dataset, obs, feature, num_mc=1000):
    tf1, tf2 = get_tukeys_fences(dataset, feature)
    x = dataset[obs, :]
    x = np.array([x] * num_mc)
    new_fv = np.random.uniform(tf1, tf2, size=len(range(num_mc)))
    x[:, feature] = new_fv
    return x


def get_bounds_and_trend(feature, perturbs):
    mu = perturbs.iloc[:, feature].mean()
    sigma = perturbs.iloc[:, feature].std()
    p_corr = perturbs.loc[:, 'y_pred'].corr(perturbs.iloc[:, feature])

    if (math.isnan(mu) == False) & (math.isnan(sigma) == False):
        bounds = [round(mu - sigma), round(mu + sigma)]
    else:
        bounds = "No reasonable bounds"
    return [feature, bounds, p_corr]


def get_explanation(stats, features, instance, train_columns):
    mcbrp_exp = pd.DataFrame(stats, columns=['Feature', 'Bounds', 'Trend'])
    mcbrp_exp['Values'] = instance.iloc[features].values

    # Replace feature ids with names
    mcbrp_exp['Feature'] = train_columns[mcbrp_exp['Feature']]

    # Replace p_corr with trend
    trend = []
    for t in mcbrp_exp['Trend']:
        if t > 0:
            trend.append('As input increases, sales increase')
        elif t < 0:
            trend.append('As input increases, sales decrease')
        else:
            trend.append('No trend between input and sales')

    mcbrp_exp['Trend'] = trend
    mcbrp_exp = mcbrp_exp[['Feature', 'Values', 'Bounds', 'Trend']]
    return mcbrp_exp