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
	power_arr = np.append(np.ones(62), np.array([2]), axis= 0)
	#np.power(x_list, power_arr)		
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
	#power_arr = [1, 1, 1, 1, 1, 2, 3, 4, 5]
	#np.power(x_list, power_arr)		
	_sum= np.sum(np.square(yn_list- (np.dot(x_list, w_list)+ b))) / len(x_list)
	"""
	if lamda == 0:
		print yn_list
		print (np.dot(x_list, w_list)+ b)
	"""
	#for j in range(len(w_list)):
	#	_sum+= lamda* (w_list[j])**2

	return _sum

		
if __name__ == '__main__':

	lr= 0.000005
	iteration= 50000
	lamda= 0
	batch_size= 565 
	w_list=np.zeros(63)
	b= 0
	#y= b+ x0w0+ x1w1+ ....+ x8w8
 	#L= segma((yni-yi)^2) + lamda*segma(wi^2)
	
	train = pd.read_csv('data/train.csv')
	test = pd.read_csv('data/test_X.csv', header= None)
	
	####training data#####
	CO_train = np.array(data_extract(train, 2))
	NO2_train = np.array(data_extract(train, 5))
	NOX_train = np.array(data_extract(train, 6))
	O3_train = np.array(data_extract(train, 7))
	PM10_train = np.array(data_extract(train, 8))
	PM25_train = np.array(data_extract(train, 9))
	SO2_train = np.array(data_extract(train, 12))	
	#PMtable= pd.Series(PM25_train)
	#PMtable.to_csv('PM25.csv')

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
				temp_list= np.concatenate((CO_train[j:j+9],NO2_train[j:j+9],NOX_train[j:j+9],O3_train[j:j+9],PM10_train[j:j+9], PM25_train[j:j+9], SO2_train[j:j+9]), axis=0)
				x_list.append(temp_list)
			start+= 1
			count+= 1	
	x_list= np.array(x_list, dtype= float)
	yn_list= np.array(yn_list, dtype= float)
	#np.int64(x_list)
	#np.int64(yn_list)
	
	####testing data#####
	test= test.T
	CO_test= []
	NO2_test= []
	NOX_test= []
	O3_test= []
	PM10_test= []
	PM25_test= []
	SO2_test= []
	for ii in range(2, 4320, 18):
		CO_test.append(np.array(test[ii][2:]))	
	for ii in range(5, 4320, 18):
		NO2_test.append(np.array(test[ii][2:]))	
	for ii in range(6, 4320, 18):
		NOX_test.append(np.array(test[ii][2:]))	
	for ii in range(7, 4320, 18):
		O3_test.append(np.array(test[ii][2:]))	
	for ii in range(8, 4320, 18):
		PM10_test.append(np.array(test[ii][2:]))	
	for ii in range(9, 4320, 18):
		PM25_test.append(np.array(test[ii][2:]))
	for ii in range(12, 4320, 18):
		SO2_test.append(np.array(test[ii][2:]))

	test_list= np.concatenate((CO_test,NO2_test,NOX_test,O3_test,PM10_test, PM25_test, SO2_test), axis= 1)
	test_list= np.array(test_list, dtype= float)

	#### test myself#####
	my_test= np.array(pd.read_csv('answer.csv', header= None)).flatten()

	best_w= [] 
	best_b= []
	best_loss= 100
	for ii in range(iteration):
		for batch in iteration_batch(x_list, yn_list, batch_size):
			x, yn= batch
			#print x, yn
			gradient, gradient_b =  gradient_linear(x, w_list, yn, b, lamda)
			if ii > 10000:
				gradient= gradient*0.1
				gradient_b= gradient_b*0.1
			if ii > 30000:
				gradient= gradient*0.1
				gradient_b= gradient_b*0.1
			if ii > 50000:
				gradient= gradient*0.01
				gradient_b= gradient_b*0.01
			if ii > 100000:
				gradient= gradient*0.01
				gradient_b= gradient_b*0.01
			if ii > 200000:
				gradient= gradient*0.01
				gradient_b= gradient_b*0.01
	
	
			w_list = w_list- lr*gradient
			b = b-lr*gradient_b
			if ii == 0:
				grad_mag= np.absolute(lr*gradient)
				grad_mag_b= np.absolute(lr*gradient_b)
		
		##print loss##
		#loss= loss_function(x_list, yn_list, w_list, b, lamda)
		#print('[{}]loss: {}'.format(ii, (loss)**0.5))
		"""
		if ii% 1000 == 0:
			prediction= np.dot(test_list, w_list)+ b
			my_loss= loss_function(prediction, my_test, 1, 0, 0)
			if my_loss < best_loss:
				best_loss= my_loss
				best_w= w_list
				best_b= b
			
			print ('[{}]test error: {}'.format(ii ,(my_loss)**0.5))
		"""
	print w_list, b 

	
	

	prediction= np.dot(test_list, best_w)+ best_b
	last_loss= loss_function(prediction, my_test, 1, 0, 0)
	print ('[{}]best error: {}'.format(ii ,(last_loss)**0.5))
	#prediction= prediction.astype(int)
	prediction= np.hstack(('value', prediction))
	
	id_index= ['id']
	for j in range(0, 240):
		id_index.append('id_'+ str(j))
	table= pd.Series(prediction, index= id_index)  
	table.to_csv('linear_regression.csv')
	 


