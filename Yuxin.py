from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

###########################        KNN         ###################################
# load data and assign them to X and y
iris = load_iris()
X, y = iris.data, iris.target

# split data into 75% training and 25% test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 26)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

knn1 = KNeighborsClassifier(n_neighbors = 8)
knn1.fit(X_train, y_train)
test_pred = knn1.predict(X_test)
cm1 = confusion_matrix(y_test, test_pred)
print(cm1)
print(accuracy_score(y_test, test_pred))

#########################    Decision Tree       ############################
# check accuracy for different max_depth for entropy
depth_range = range(1, 10)
scores = []
for d in depth_range:
    entropy = DecisionTreeClassifier(criterion="entropy", random_state=3, max_depth=d)
    entropy.fit(X_train, y_train)
    y_pred_en = entropy.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred_en))
plt.plot(depth_range, scores, 'ro')
plt.title('Decision Tree: Varying Number of max-depth')
plt.xlabel('Max depth of decision tree')
plt.ylabel('Accuracy')
plt.show()

# when max-depth=3, accuracy is the highest if using entropy index
entropy = DecisionTreeClassifier(criterion="entropy", random_state=3, max_depth=3)
entropy.fit(X_train, y_train)
y_pred_en = entropy.predict(X_test)

# confusion matrix and accuracy
cm = confusion_matrix(y_test, y_pred_en)
print(cm)
print(accuracy_score(y_test, y_pred_en))

### Name
print("My name is Yuxin Sun")
print("My NetID is: yuxins5")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")