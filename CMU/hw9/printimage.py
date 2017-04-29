import scipy.io
import cv2
import load

file_dir = 'data_mat/test_data.mat'
prefix = ''
data = scipy.io.loadmat(file_dir)
data = data['data']
images,images2 = load._convert_images(data)
output_dir = 'output/'
count = 0
for item in images:
    print item.shape
    cv2.imwrite(output_dir+str(count)+'.jpg',item)
    count += 1