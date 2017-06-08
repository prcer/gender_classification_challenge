from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier()
# CHALLENGE - create 3 more classifiers
clf1 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
clf2 = SVC(kernel = 'linear', random_state = 0)
clf3 = LogisticRegression(random_state = 0)

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X, Y)

# Predictions vector to test the models (chosen independently from the training set)
Ytest = [[190, 70, 43], [150, 70, 30], [170, 80, 38], [200, 90, 44], [160, 50, 32]]

# Predit genders for samples in the test set for each classifier
prediction = clf.predict(Ytest)
prediction1 = clf1.predict(Ytest)
prediction2 = clf2.predict(Ytest)
prediction3 = clf3.predict(Ytest)

# Ground truth for the predictions
predTrue = ['male', 'female', 'male', 'male', 'female']

# Calculate accuracy for each classifier
s1 = accuracy_score(predTrue, prediction1)
s2 = accuracy_score(predTrue, prediction2)
s3 = accuracy_score(predTrue, prediction3)

# CHALLENGE: compare their results and print the best one!

# Print predicted values and accuracy in % for each classifier
print('RandomForestClassifier: ', s1 * 100, '%')
print(prediction1)
print('SVC: ', s2 * 100, '%')
print(prediction2)
print('LogisticRegression ', s3 * 100, '%')
print(prediction3)

# Print the best classifier
if s1 >= s2 and s1 >= s3:
    print('Best Classifier: RandomForestClassifier')
elif s2 >= s1 and s2 >= s3:
    print('Best Classifier: SVC')
else:
    print('Best Classifier: LogisticRegression')
