import matplotlib.pyplot as plt
import numpy as np
import csv

# Task 1
# set 1 as the first column
def loadDataSet(filetpath):
    X = []
    y = []
    with open(filetpath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            xline = [1.0]
            for s in line[1:-1]:
                xline.append(float(s))
            X.append(xline)
            y.append(float(line[0]))
    return X, y

# deal with the x and y
xArr1, yArr1 = loadDataSet('lr_training.csv')

# deal with "np.array"
X = np.array(xArr1)
y = np.transpose(np.array(yArr1))


# display b_opt
inv = np.linalg.pinv
b_opt = np.dot(inv(np.dot(X.T, X)), np.dot(X.T, y))

print("b_opt: ")
print(b_opt)

# classify test data
xArr2, yArr2 = loadDataSet('lr_test.csv')

y_pred = []

for x in range(len(xArr2)):
    data = np.dot(xArr2[x], b_opt)
    if data > 0.5:
        cval = 1.0
    else:
        cval = 0.0
    y_pred.append(cval)

# print Accuracy
print("Accuracy: ", (np.array(y_pred) == np.array(yArr2)).mean())

# Task 2
# Gradient Descent
def gradient(xval, yval, step, learningRate):
    acost = []
    points = len(xval[0])
    theta = np.zeros(points)

    for i in range(step):
        ate = 0
        thetaL = np.zeros(points)

        for i in range(len(xval)):
            hpred = np.dot(xval[i], theta)
            thetaL = thetaL + np.dot(hpred-yval[i], xval[i])
            ate = ate + (1/2.0)*(hpred-yval[i])**2

        acost.append(ate/points)

        theta = theta - learningRate * thetaL/points

    return theta, acost

# running gradient d
b_est, cost = gradient(X,y,1000,0.0000001)


plt.xlabel('iteration')
plt.ylabel('cost')
plt.plot(cost)
plt.show()

# display the b_est
print("b_est: ")
print(b_est)
print("cost: ", cost)

# display accuracy
y_pred1 = []
for x in range(len(xArr2)):
    data1 = np.dot(xArr2[x], b_est)
    if data1 > 0.5:
        cval = 1.0
    else:
        cval = 0.0
    y_pred1.append(cval)

# print Accuracy
print("Accuracy: ", (np.array(y_pred1) == np.array(yArr2)).mean())

# display the total differences
td = sum(abs(b_opt-b_est))
print("Diff: ", td)