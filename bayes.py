from sklearn.naive_bayes import GaussianNB
import numpy as np
features = np.array([[0, 3, 6, 9], [0, 3, 6, 8], [2,3,6,9],[1, 4, 6, 9], [1, 5, 7, 9]])
play = np.array([0, 0, 1, 1, 1])
model = GaussianNB()
model.fit(features, play)
predicted= model.predict([[0,3,6,9],[2,3,6,9]])
print(predicted)
