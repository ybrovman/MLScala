import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

numData = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
predictionTime = [12.205973, 49.650052, 124.38651, 233.504311, 382.431598, 467.915364, 652.775037]

k = range(1,10)
error1000 = [0.145,0.145,0.1525,0.1575,0.165,0.1625,0.165,0.17,0.1725]
error2000 = [0.11625,0.11625,0.1025,0.1075,0.10875,0.10625,0.11875,0.12375,0.1275]
error3000 = [0.09166666666666666,0.09166666666666666,0.075,0.07583333333333334,0.0825,0.08583333333333333,0.08666666666666667,0.08916666666666667,0.09083333333333334]
error4000 = [0.086875,0.086875,0.081875,0.08375,0.086875,0.0875,0.093125,0.0975,0.103125]
error5000 = [0.0765,0.0765,0.0815,0.0835,0.082,0.0855,0.084,0.086,0.0895]
error6000 = [0.07125,0.07125,0.07166666666666667,0.07458333333333333,0.0725,0.07416666666666667,0.07708333333333334,0.07583333333333334,0.075]

k7000 = range(1,30)
error7000 = [0.061785714285714284,0.061785714285714284,0.06285714285714286,0.06321428571428571,0.06535714285714286, \
         0.06607142857142857,0.06821428571428571,0.06892857142857142,0.06678571428571428,0.06714285714285714, \
         0.06928571428571428,0.07035714285714285,0.0725,0.07428571428571429,0.075,0.075,0.08,0.08178571428571428, \
         0.08357142857142857,0.08464285714285714,0.08464285714285714,0.08642857142857142,0.08571428571428572, \
         0.08642857142857142,0.08678571428571429,0.09035714285714286,0.09035714285714286,0.09178571428571429, \
         0.09321428571428571]

plt.close('all')
plt.figure(1)
# plt.plot(numData, predictionTime, '-o')
plt.plot(k, error1000, '-o', label='num examples = 1000')
plt.plot(k, error2000, '-o', label='num examples = 2000')
plt.plot(k, error3000, '-o', label='num examples = 3000')
plt.plot(k, error4000, '-o', label='num examples = 4000')
plt.plot(k, error5000, '-o', label='num examples = 5000')
plt.plot(k, error6000, '-o', label='num examples = 6000')
plt.plot(k7000, error7000, '-o', label='num examples = 7000')

plt.xlabel("k")
plt.ylabel("test error")
plt.title("MNIST data, kNN classifier")

def trainPy():
    mnist = fetch_mldata("MNIST original")
    # rescale the data, use the traditional train/test split
    X, y = mnist.data / 255., mnist.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=3000, test_size=2000, random_state=42)
    print "size=", len(y_train)+len(y_test)

    errorTrain = []
    errorTest  = []

    for kVal in range(1,30):
        clf = KNeighborsClassifier(n_neighbors=kVal)
        clf.fit(X_train, y_train)
        errorTrain.append(1.0 - clf.score(X_train, y_train))
        errorTest.append(1.0 - clf.score(X_test, y_test))

    # plt.plot(k, errorTrain, '-o', label='PyTrain', linewidth=2)
    plt.plot(k7000, errorTest, '-o', label='5000PyTest', linewidth=2)

plt.legend(loc='upper right')
plt.show(block=False)