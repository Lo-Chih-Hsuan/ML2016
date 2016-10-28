import numpy as np
import pandas as pd
import sys 

def iteration_batch(x_list, yn_list, batch_size):
	indices= np.arange(len(x_list))
	np.random.shuffle(indices)
	for ii in range(0, len(x_list), batch_size):
		if ii+ batch_size < len(x_list) :
			excerpt= indices[ii: ii+ batch_size]
		else: 
			excerpt= indices[ii: len(x_list)]
		yield x_list[excerpt], yn_list[excerpt]	

def logistic_regression(x_list, yn_list, w_list): #regularization, x_list, y_list-> hole batch list       
	gradient = np.dot(x_list.T* (-1), (yn_list- 1.0/ (1.0+ np.exp(-np.dot(x_list, w_list))))) / len(x_list)		 
	return gradient

def adagrad(g_sum, g_now):
	g = ((g_sum)**2 + g_now**2)**0.5
	return g
	

if __name__ == '__main__':
	
	lr = 0.1
	iteration = 5000
	batchsize= 3800
	w_list = np.random.normal(0.0, 0.01, [58])
	#b= np.random.normal(1, 0.1)
	
	train_data_file = sys.argv[1]
	out_model_file = sys.argv[2]
	
	train_data= np.array(pd.read_csv(train_data_file, header= None))
	tr_list= []
	tr_tar_list= []

	for data in train_data:
		target = data[-1]
		data = np.append( data[1:-1], 1)
		tr_list.append(data) 
		tr_tar_list.append(target)
	tr_list = np.array(tr_list)
	tr_tar_list = np.array(tr_tar_list)
	
	best_w = list()
	best_error= float("inf")
	g_sum= 0
	for ii in range(iteration):
		count = 0
		for batch in iteration_batch(tr_list, tr_tar_list, batchsize):
			if count < len(tr_list) / batchsize:
				train, target = batch
				gradient =  logistic_regression(train, target, w_list)
				g_sum= adagrad(g_sum, gradient)
				w_list = w_list- lr*gradient/ g_sum
				f = (np.array(1+ np.exp(-np.dot(train, w_list)), dtype= float))**(-1)
				error= np.mean(-(target* np.log(f+1e-30)+ (1- target)* np.log(1- f+1e-30)))
				#print ('[{}]loss: {}'.format(ii ,error))
			
			else:
				va, va_tar= batch
				f = (np.array(1+ np.exp(-np.dot(va, w_list)), dtype= float))**(-1)
				error= np.mean(-(va_tar*np.log(f+1e-30)+ (1- va_tar)* np.log(1- f+1e-30)))
				if error <= best_error:
					best_w = w_list
					best_error= error	
				print ('[{}]validation loss: {}'.format(ii ,best_error))
			count+= 1			
	np.save(out_model_file, w_list)				
