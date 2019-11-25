import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import pickle


def main():

	df = pd.read_csv('../data/train.csv')
	X = df.iloc[:, :-1]
	y = df.iloc[:, -1]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

	model = GradientBoostingRegressor(loss='ls', learning_rate=0.5, n_estimators=300)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	print("R^2 score:", r2_score(y_pred, y_test))

	# Save train, test and model
	X_train['class'] = y_train
	X_test['class'] = y_test

	train = X_train.copy().reset_index()
	test = X_test.copy().reset_index()

	del train['index']
	del test['index']

	train.to_csv('../data/superconductivity_data_train.tsv', sep='\t')
	test.to_csv('../data/superconductivity_data_test.tsv', sep='\t')
	pickle.dump(model, open('../results/model_superconductivity', 'wb'))


if __name__ == main():
	main()
