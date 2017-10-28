import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
import numpy as np
# The digits dataset
digits = datasets.load_digits()
#print (digits['target'])
#for key,value in digits.items() :
#    try:
#        print (key,value.shape)
#    except:
#        print (key)
        
images_and_labels = list(zip(digits.images, digits.target))
#print (digits.images[:1])
#print (digits.target[:1])
#print (images_and_labels[:1])

#for index, (image, label) in enumerate(images_and_labels[:4]):
#    plt.subplot(1, 4, index + 1)# row, colum, plot number 
#    plt.axis('off')
#    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='bicubic')
#    plt.title('Training: %i' % label)

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
preprocessing.scale(data)
SVM_C = svm.SVC(gamma=0.001)
x_train, x_test, t_train, t_test = train_test_split(data, digits.target
                                                    , test_size = 0.5)
SVM_C.fit(x_train, t_train)
expected = t_test
predicted = SVM_C.predict(x_test)
#print(expected[:10])
#print(predicted[:10])

#cross validation, find average value
#print(SVM_C.score(x_test, expected))
#scores = cross_val_score(SVM_C, data, digits.target, cv=5, scoring='accuracy')
#print(scores.mean())

#cross validation, find best gamma
#gamma = np.arange(0.001,0.01,0.001)
#g_scores = []
#for g in gamma:
#    SVM_C = svm.SVC(gamma=g)
###    loss = -cross_val_score(knn, X, y, cv=10, scoring='mean_squared_error') # for regression
#    scores = cross_val_score(SVM_C, data, digits.target, cv=10, scoring='accuracy') # for classification
#    g_scores.append(scores.mean())
#    
#plt.plot(gamma, g_scores)
#plt.xlabel('Value of g for SVM_C')
#plt.ylabel('Cross-Validated Accuracy')
#plt.show()

#confusion matrix
print("Confusion matrix:\n%s"
    % metrics.confusion_matrix(expected, predicted))

print("Classification report for classifier %s:\n%s\n"
    % (SVM_C, metrics.classification_report(expected, predicted)))

