import numpy
import xlsxwriter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_validate
X_train = numpy.loadtxt('X_train.txt', delimiter=' ')
X_test = numpy.loadtxt('X_test.txt', delimiter=' ')
Y_train = numpy.loadtxt('y_train.txt')
Y_test = numpy.loadtxt('y_test.txt')
X = numpy.concatenate((X_train, X_test))
Y = numpy.concatenate((Y_train, Y_test))
pca = PCA(n_components=2)
pca.fit(X)
pca_X = pca.transform(X)
original_train_dataset = train_test_split(X, Y, test_size=0.3)
original_train_x = original_train_dataset[0]
original_train_y = original_train_dataset[2]
pca_train_dataset = train_test_split(pca_X, Y, test_size=0.3)
pca_train_x = pca_train_dataset[0]
pca_test_x = pca_train_dataset[1]
pca_train_y = pca_train_dataset[2]
pca_test_y = pca_train_dataset[3]
pca_score = cross_validate(svm.SVC(), pca_train_x, pca_train_y, cv=5)
original_score = cross_validate(svm.SVC(), original_train_x, original_train_y, cv=5)
workbook = xlsxwriter.Workbook('dim_reduction.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write('A1', ' ')
worksheet.write('B1', 'Przed redukcja')
worksheet.write('C1', 'Po redukcji')
worksheet.write('A2', 'Precyzja')
worksheet.write('B2', original_score.get('test_score').mean())
worksheet.write('C2', pca_score.get('test_score').mean())
worksheet.write('A3', 'Czas z. trening')
worksheet.write('B3', original_score.get('fit_time').mean())
worksheet.write('C3', pca_score.get('fit_time').mean())
worksheet.write('A4', 'Czas z. test')
worksheet.write('B4', original_score.get('score_time').mean())
worksheet.write('C4', pca_score.get('score_time').mean())
workbook.close()
