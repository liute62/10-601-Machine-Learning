import numpy as np

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

filter1 = np.matrix('1 1; 2 0')
filter2 = np.matrix('2 1; 0 2')
img = np.matrix('0 0 4 0; 0 2 5 0; 0 0 1 0; 0 0 2 0')
learn_rate = 0.01
conv_1 = np.zeros((3,3))
conv_2 = np.zeros((3,3))
print '[INPUT IMAGE]'
print img

for i in range(0,conv_1.shape[0]):
    for j in range(0,conv_1.shape[1]):
        m1 = img[i:i + 2, j:j + 2]
        m2 = filter1
        m3 = filter2
        m1 = np.ndarray.flatten(m1)
        m2 = np.ndarray.flatten(m2).transpose()
        m3 = np.ndarray.flatten(m3).transpose()
        res1 = np.dot(m1, m2)
        res2 = np.dot(m1, m3)
        conv_1[i,j] = res1
        conv_2[i,j] = res2

print '[CONV RESULT]'
print conv_1
print conv_2

max_pool_1 = np.zeros((2,2))
max_pool_2 = np.zeros((2,2))

for i in range(0,max_pool_1.shape[0]):
    for j in range(0,max_pool_1.shape[1]):
        m1 = conv_1[i:i + 2, j:j + 2]
        m2 = conv_2[i:i + 2, j:j + 2]
        m1 = np.ndarray.flatten(m1)
        m2 = np.ndarray.flatten(m2)
        max_pool_1[i,j] = np.max(m1)
        max_pool_2[i,j] = np.max(m2)

print '[MAX_POOL RESULT]'
print max_pool_1
print max_pool_2

print '[IP RESULT] w/b matrix and input x, output y'
w1 = np.matrix('1, 0.5, 1, 2, -1, 1, 1, -1; 0.5, 1, 2, 1, -0.5, 0.5, -1, 1; '
               '0.25, -1, 1, 1, 0.5, -1, -1, -1')
b = np.matrix('0;0;0')
print w1
print b
a1 = np.ndarray.flatten(max_pool_1)
a2 = np.ndarray.flatten(max_pool_2)
x = np.concatenate((a1,a2))
print x
y1 = sum([w1[0,i]*x[i] for i in range(0,len(x))]) + b[0]
y2 = sum([w1[1,i]*x[i] for i in range(0,len(x))]) + b[1]
y3 = sum([w1[2,i]*x[i] for i in range(0,len(x))]) + b[2]
print y1,y2,y3

print '[ACTIVATION RESULT]'
y1 = max(y1,0)
y2 = max(y2,0)
y3 = max(y3,0)
print y1,y2,y3

print '[SOFTMAX RESULT]'
sum = np.exp(y1) + np.exp(y2) + np.exp(y3)
s1 = np.exp(y1) / sum
s2 = np.exp(y2) / sum
s3 = np.exp(y3) / sum
print s1,s2,s3

print '[LOSS RESULT]'
print -np.log(s2)

print '[dJ / dy2]'
print -(1 / s2)

print '[dJ / dx2 = 3y2 - 1]'
print 3 * s2 - 1

print '[dJ / dW1(1,1) = '
print x[0] * 1 * (2* s1 - 1 + s2)

print '[dJ / dF1(1,1)]'
sum_derive = 0
multiply = 0
part_derive = [0,4,2,5]
for s in [s1,s2,s3]:
    s = 3 * s - 1
    d_d = 0
    x_d = 0
    print 's:'
    print s
    for i in range(0,len(x) / 2):
        d_d += x[i] * s * part_derive[i]
    print 'd_d'
    print d_d
print d_d

array = [0,0,0,0]
print np.max(array)