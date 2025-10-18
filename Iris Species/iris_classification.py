# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import os

# Load the dataset
df = pd.read_csv('archive/Iris.csv')

# Preprocessing
# Drop the 'Id' column as it is not required for training
df = df.drop(columns=['Id'])

# Handle missing values (if any) - in this dataset, there are no missing values, but this is a good practice
df.dropna(inplace=True)

# Encode the target variable 'Species'
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

# Separate features (X) and target (y)
X = df.drop(columns=['Species'])
y = df['Species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
def train_and_save(model_path='models'):
	# Train the Decision Tree Classifier
	clf = DecisionTreeClassifier()
	clf.fit(X_train, y_train)

	# Make predictions
	y_pred = clf.predict(X_test)

	# Evaluate the model
	accuracy = accuracy_score(y_test, y_pred)
	precision = precision_score(y_test, y_pred, average='weighted')
	recall = recall_score(y_test, y_pred, average='weighted')

	# Ensure model directory exists
	os.makedirs(model_path, exist_ok=True)

	# Save model and label encoder
	model_file = os.path.join(model_path, 'decision_tree.joblib')
	encoder_file = os.path.join(model_path, 'label_encoder.joblib')
	joblib.dump(clf, model_file)
	joblib.dump(le, encoder_file)

	# Print the evaluation metrics
	print(f"Model saved to: {model_file}")
	print(f"Encoder saved to: {encoder_file}")
	print(f"Accuracy: {accuracy}")
	print(f"Precision: {precision}")
	print(f"Recall: {recall}")


if __name__ == '__main__':
	train_and_save()
