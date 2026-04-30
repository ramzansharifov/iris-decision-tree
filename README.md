# Iris Flower Classification Using Decision Tree

## Project Overview

This project is a simple machine learning study based on the classic **Iris Dataset**.

The main goal of the project is to classify iris flowers into one of three species using a **Decision Tree Classifier**.

The project was prepared as a homework assignment for the course **Artificial Intelligence Systems**.

---

## Research Goal

The goal of this research is to build and evaluate a machine learning model that can predict the species of an iris flower based on its physical measurements.

The project solves a **multiclass classification problem**, because the model predicts one class out of three possible iris species.

---

## Dataset

The project uses the built-in Iris dataset from the `scikit-learn` library.

The dataset contains **150 samples** of iris flowers.  
Each sample includes four numerical features and one target class.

```python
from sklearn.datasets import load_iris

iris = load_iris()
```

---

## Features

The model uses the following input features:

| Feature | Description |
|---|---|
| sepal length | Length of the sepal in centimeters |
| sepal width | Width of the sepal in centimeters |
| petal length | Length of the petal in centimeters |
| petal width | Width of the petal in centimeters |

In the code, these features are stored in the variable `X`:

```python
X = df[iris.feature_names]
```

---

## Target Classes

The model predicts one of the following iris species:

| Class Number | Species |
|---|---|
| 0 | setosa |
| 1 | versicolor |
| 2 | virginica |

The target variable is stored in `y`:

```python
y = df["species"]
```

---

## Technologies Used

The project uses the following Python libraries:

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

Main tools and libraries:

- Python
- Jupyter Notebook
- pandas
- matplotlib
- scikit-learn

---

## Project Workflow

The project includes the following steps:

1. Loading the Iris dataset
2. Converting the dataset into a pandas DataFrame
3. Displaying basic information about the dataset
4. Preparing features and target variables
5. Splitting the dataset into training and testing sets
6. Creating the Decision Tree model
7. Training the model
8. Making predictions on the test data
9. Evaluating the model
10. Visualizing the Decision Tree

---

## Data Preparation

The dataset is converted into a pandas DataFrame to make it easier to analyze and display.

```python
df = pd.DataFrame(iris.data, columns=iris.feature_names)

df["species"] = iris.target
df["species_name"] = df["species"].apply(lambda x: iris.target_names[x])
```

Basic dataset information is displayed using:

```python
print(df.head())
print(df.shape)
print(df["species_name"].value_counts())
```

---

## Train-Test Split

The dataset is divided into training and testing sets.

- 80% of the data is used for training
- 20% of the data is used for testing

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The `stratify=y` parameter keeps the class distribution balanced in both training and testing sets.

---

## Model Training

The main algorithm used in this project is the **Decision Tree Classifier**.

A Decision Tree is a machine learning algorithm that makes decisions by splitting the data according to feature values.  
Its logic is easy to understand because it works similarly to a sequence of simple if-else rules.

```python
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)
```

The parameter `max_depth=3` limits the depth of the tree and makes the model easier to understand and visualize.

---

## Prediction

After training, the model is used to make predictions on the test data.

```python
y_pred = model.predict(X_test)
```

---

## Model Evaluation

The model is evaluated using three main metrics:

- Accuracy score
- Classification report
- Confusion matrix

```python
accuracy = accuracy_score(y_test, y_pred)

print(round(accuracy, 4))
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print(confusion_matrix(y_test, y_pred))
```

---

## Results

The model achieved the following accuracy:

```text
0.9667
```

This means that the model correctly classified approximately **96.67%** of the test samples.

The result shows that the Decision Tree model performs well on the Iris dataset.

---

## Decision Tree Visualization

The trained Decision Tree is visualized using `plot_tree`.

```python
plt.figure(figsize=(14, 8))

tree_artists = plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=10
)

for artist in tree_artists:
    artist.set_color("black")

plt.title("Decision Tree for Iris Classification")
plt.tight_layout()

plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")
plt.show()
```

The visualization is saved as:

```text
decision_tree.png
```

---

## Project Structure

Example project structure:

```text
iris-decision-tree/
│
├── iris_decision_tree.ipynb
├── decision_tree.png
├── README.md
├── .gitignore
└── requirements.txt
```

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone <repository-url>
```

### 2. Open the project folder

```bash
cd iris-decision-tree
```

### 3. Install the required libraries

```bash
pip install pandas matplotlib scikit-learn notebook ipykernel
```

### 4. Open the Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```text
iris_decision_tree.ipynb
```

Run the notebook cells from top to bottom.

---

## Requirements

The project requires the following Python libraries:

```text
pandas
matplotlib
scikit-learn
notebook
ipykernel
```

If you use `requirements.txt`, it can contain:

```text
pandas
matplotlib
scikit-learn
notebook
ipykernel
```

Then install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Conclusion

In this project, a Decision Tree model was trained to classify iris flowers based on four physical measurements.

The model achieved high accuracy and produced an interpretable tree structure.  
The most important features for classification are mainly related to the petals, especially petal length and petal width.

This project demonstrates the basic workflow of a machine learning classification task: loading data, preparing features, training a model, evaluating results, and visualizing the decision-making process.