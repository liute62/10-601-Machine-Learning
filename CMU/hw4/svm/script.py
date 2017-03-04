import run_svm as rs
import numpy as np

(X,Y) = rs.get_data("hw4data2.mat")
C = 4
if False:
    X = np.delete(X, 19, 0)
    Y = np.delete(Y,19, 0)
if False:
    X = np.delete(X, 0, 0)
    Y = np.delete(Y,0, 0)
C = 1
model = rs.run_svm(X,Y,C)
print model
