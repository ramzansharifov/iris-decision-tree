"""
Iris Flower Classification Using Decision Tree

This project uses the Iris dataset and a Decision Tree classifier
to predict the species of iris flowers based on four measurements.
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    # Load the Iris dataset from scikit-learn
    iris = load_iris()

    # Convert the dataset into a pandas DataFrame for easier analysis
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Add the target column
    df["species"] = iris.target

    # Add readable species names
    df["species_name"] = df["species"].apply(lambda x: iris.target_names[x])

    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nClass distribution:")
    print(df["species_name"].value_counts())

    # Features are the input variables used for prediction
    X = df[iris.feature_names]

    # Target is the class we want to predict
    y = df["species"]

    # Split the data into training and testing sets
    # 80% is used for training, 20% is used for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create the Decision Tree model
    # max_depth limits the size of the tree and makes it easier to understand
    model = DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    )

    # Train the model using the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    print("\nModel accuracy:")
    print(round(accuracy, 4))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Visualize the Decision Tree
    plt.figure(figsize=(14, 8))

    plot_tree(
        model,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True
    )

    plt.title("Decision Tree for Iris Classification")
    plt.tight_layout()

    # Save the tree image to the project folder
    plt.savefig("decision_tree.png", dpi=300)

    # Show the tree
    plt.show()


if __name__ == "__main__":
    main()