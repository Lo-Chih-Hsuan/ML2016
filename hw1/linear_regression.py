import numpy as np
from scipy import stats
import pandas as pd

def data_extract(train_list, start_point):
	train_list= train_list.T
	mylist= []
	for i in range(start_point, 4321, 18):
		mylist.extend(train_list[i][3:])
	
	return mylist


def gradient_linear(x_list, w_list, yn_list, b, lamda): #regularization, x_list, y_list-> hole batch list		
	_sum= np.dot(x_list.T, (yn_list- (np.dot(x_list, w_list)+ b))*(-1)) / len(x_list)
	_sum_b= np.sum((yn_list- (np.dot(x_list, w_list)+ b))* (-1)) / len(x_list)
	gradient= 2*_sum+ 2*w_list*lamda
	gradient_b= 2*_sum_b

	return gradient, gradient_b


def iteration_batch(x_list, yn_list, batch_size):
	indices= np.arange(len(x_list))
	np.random.shuffle(indices)
	for ii in range(0, len(x_list)-batch_size+1, batch_size):
		excerpt= indices[ii: ii+ batch_size]
		yield x_list[excerpt], yn_list[excerpt]  

def loss_function(x_list, yn_list, w_list, b, lamda):
	_sum= np.sum(np.square(yn_list- (np.dot(x_list, w_list)+ b))) / len(x_list)
	for j in range(len(w_list)):
		_sum+= lamda* (w_list[j])**2

	return _sum

		
if __name__ == '__main__':

	lr= 0.0000001
	iteration= 300000
	lamda= 0
	batch_size= 100 
	w_list=np.zeros(9)
	b= 0
	#y= b+ x0w0+ x1w1+ ....+ x8w8
 	#L= segma((yni-yi)^2) + lamda*segma(wi^2)
	
	train = pd.read_csv('data/train.csv')
	test = pd.read_csv('data/test_X.csv', header= None)
	PM25_train = data_extract(train, 9)
	
	x_list= [] #471
	yn_list= []
	start= 0 
	for ii in range(12):
		count= 0
		temp_start = start
		for j in range(temp_start, temp_start+ 480):	
			if count > 8:
				yn_list.append(PM25_train[j])
			if temp_start+ count < temp_start+ 471:
				x_list.append(PM25_train[j:j+ 9])
			start+= 1
			count+= 1	
	x_list= np.array(x_list, dtype= int)
	yn_list= np.array(yn_list, dtype= int)
	#np.int64(x_list)
	#np.int64(yn_list)

	grad_mag= np.array([], dtype= float)
	grad_mag_b= 0
	for ii in range(iteration):
		for batch in iteration_batch(x_list, yn_list, batch_size):
			x, yn= batch
			#print x, yn
			gradient, gradient_b =  gradient_linear(x, w_list, yn, b, lamda)
			w_list = w_list- lr*gradient
			b = b-lr*gradient_b
			if ii == 0:
				grad_mag= np.absolute(lr*gradient)
				grad_mag_b= np.absolute(lr*gradient_b)
		
		##print loss##
		loss= loss_function(x_list, yn_list, w_list, b, lamda)
		if ii% 100 == 0:
			print('[{}]loss: {}'.format(ii, (loss)**0.5))
	
	print w_list, b 

	####testing#####
	test= test.T
	PM25_test= []
	for ii in range(9, 4320, 18):
		PM25_test.append(np.array(test[ii][2:]))

	PM25_test= np.array(PM25_test, dtype= int)
	prediction= np.dot(PM25_test, w_list)+ b
	#prediction= prediction.astype(int)
	prediction= np.hstack(('value', prediction))

	id_index= ['id']
	for j in range(0, 240):
		id_index.append('id_'+ str(j))
	table= pd.Series(prediction, index= id_index)  
	table.to_csv('linear_regression.csv')
	 


