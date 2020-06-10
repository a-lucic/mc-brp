import pandas as pd
import numpy as np
import argparse
import pickle
import lime
import lime.lime_tabular
import random
import re
from src.mc_brp_functions import *


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', type=str, required=True)
	parser.add_argument('--test', type=str, required=True)
	parser.add_argument('--model', type=str, required=True)
	parser.add_argument('--num_features', type=int, required=True)
	parser.add_argument('--num_mc', type=int, required=True)

	args = parser.parse_args()
	train = args.train
	test = args.test
	model = args.model
	num_features = args.num_features
	num_mc = args.num_mc


	# Import
	train = pd.read_csv(train, sep='\t', index_col=0)
	test = pd.read_csv(test, sep='\t', index_col=0)
	model = pickle.load(open(model, 'rb'))
	X_train = train.iloc[:, :-1]
	X_test = test.iloc[:, :-1]
	y_train = train.iloc[:, -1]
	y_test = test.iloc[:, -1]


	# Label test set as large errors/not large errors
	X_test_stats = X_test.copy()
	X_test_stats['y_pred'] = model.predict(X_test_stats)
	X_test_stats['y_test'] = y_test
	X_test_stats['abs_error'] = abs(X_test_stats['y_pred'] - X_test_stats['y_test'])
	q1, q3 = X_test_stats['abs_error'].quantile(0.25), X_test_stats['abs_error'].quantile(0.75)
	upper_bound = q3 + 1.5 * (q3 - q1)
	large_error = np.array(X_test_stats['abs_error'] > upper_bound)
	X_test_stats['large_error'] = large_error.astype(np.int)


	# Select large error to be explained by MC-BRP
	idx_list = np.argwhere(X_test_stats['large_error'] == 1).flatten()
	idx = random.choice(idx_list)
	print("Upper_bound for large errors: {}".format(upper_bound))
	print("Number of large errors in test set: {}".format(len(idx_list)))
	print("Index of test instance to be explained: {}".format(idx))


	# Get top n features from LIME
	lime_explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train),
													   		feature_names=X_train.columns,
													   		verbose=False,
													   		mode='regression')
	lime_exp = lime_explainer.explain_instance(	np.array(X_test.iloc[idx]),
									 		   	model.predict,
									 			num_features=num_features).as_list()
	top_features = []
	for l in lime_exp:
		top_features.append((" ".join(re.findall("[a-zA-Z]+", l[0])).replace(" ", "_")))
	feature_ids = [X_train.columns.get_loc(f) for f in top_features]
	np_X_test = np.array(X_test)
	print("Top {} features: {}".format(num_features, top_features))


	# Do Monte Carlo simulations for each feature in top features
	perturbations = []
	for f in feature_ids:
		x_perturbs = get_perturbations(np_X_test, idx, f, num_mc)
		perturbations.append(x_perturbs)
	perturbations = np.vstack(perturbations) 		# shape = num_mc*len(feature_ids), len(X_train.columns)


	# Label perturbations as large errors/not large errors
	perturbations_df = pd.DataFrame(perturbations, columns=X_train.columns)
	perturbations_df['y_pred'] = model.predict(perturbations)
	perturbations_df['y_test'] = np.full(len(perturbations_df), y_test.loc[idx])
	perturbations_df['abs_error'] = abs(perturbations_df['y_pred'] - perturbations_df['y_test'])
	large_error_p = np.array(perturbations_df['abs_error'] > upper_bound)
	perturbations_df['large_error'] = large_error_p.astype(np.int)


	# Identify perturbations resulting in reasonable errors (R')
	R_prime = perturbations_df[perturbations_df['large_error'] == 0]
	stats = []
	if len(R_prime) == 0:
		print("All perturbations result in large errors -- try increasing num_mc or num_features")
	else:
		for f in feature_ids:
			stats.append(get_bounds_and_trend(f, R_prime))
		mcbrp_explanation = get_explanation(stats, feature_ids, X_test.iloc[idx], X_train.columns)
		mcbrp_explanation.to_csv('../results/MCBRP_explanation.csv')
		print("MC-BRP Explanation: {}".format(mcbrp_explanation))


if __name__ == '__main__':
	main()
