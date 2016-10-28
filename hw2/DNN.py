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

def adagrad(g_sum, g_now):
	g = ((g_sum)**2 + g_now**2)**0.5
	return g
def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))
def sigmoid_prime(x):
	return sigmoid(x)*(1.0-sigmoid(x))

if __name__ == '__main__':
	
	lr = 0.15
	iteration = 1500
	batchsize= 3000
	#w_list = np.random.normal(0.0, 0.01, [58])
	#b= np.random.normal(1, 0.1)
	
	train_data_file = sys.argv[1]
	out_model_file = sys.argv[2]
	
	train_data= np.array(pd.read_csv(train_data_file, header= None))
	tr_list= []
	tr_tar_list= []
	for data in train_data:
		target = data[-1]
		data = np.append(1, data[1:-1])
		tr_list.append(data) 
		tr_tar_list.append(target)
	tr_list = np.array(tr_list)
	tr_tar_list = np.array(tr_tar_list)
			
	weight= []
	##input layer: 58*31###
	r1= 0.01*np.random.random((58, 64))
	weight.append(r1)
	######################
	#rr= 0.01*np.random.random((32, 16))
	#weight.append(rr)
	##output layer: (31+ 1 bias)*1###
	r2= 0.01*np.random.random((64, 1))
	weight.append(r2)
			
		
	best_w = list()
	best_error= float("inf")
	g_sum= [0.0, 0.0]
	for ii in range(iteration):
		count = 0
		for batch in iteration_batch(tr_list, tr_tar_list, batchsize):
			if count < len(tr_list) / batchsize:
				train, target = batch
				target= np.array([target]).T
				
				###Forward propagation###
				a = [train]
				for l in range(len(weight)):
					z= np.dot(a[l], weight[l]) 
					activation= 1.0/ (1.0+ np.exp(-z))
					if l < len(weight)-1:
						activation.T[0].fill(1.0)	
					a.append(activation)
				
				loss = np.mean(-((target*np.log(a[-1]+1e-30))+ (1-target)* np.log(1- a[-1]+1e-30)))
				if ii% 1000 ==0:
					print loss
				
				###Backward propagation###
				error= -(target/(a[-1]+1e-30)-(1-target)/ (1-a[-1]+1e-30))* (a[-1]*(1-a[-1]))
				deltas= [error]#* sigmoid_prime(a[-1])] 
				for l in range(len(a)-2, 0, -1):
					deltas.append(np.dot(deltas[-1],weight[l].T)*(a[l]*(1-a[l])))
				deltas.reverse()
				for i in range(len(weight)): # a1*delta2, a2*delta3
					g_sum[i]= adagrad(g_sum[i], np.dot(a[i].T,deltas[i]))
					weight[i]-= lr* np.dot(a[i].T,deltas[i])/ g_sum[i]
				
			else:
				va, va_tar= batch
				a = va
				for l in range(len(weight)):
					a= 1.0/ (1.0 +np.exp(-np.dot(a, weight[l])))
					if l < len(weight)-1:
						a.T[0].fill(1.0)
				a = a.reshape(len(a))
				for j in range(len(a)):
					if a[j] < 0.5:
						a[j]= 0.0
					else:
						a[j]= 1.0
				#error= 1-np.mean(np.abs(a- va_tar))
				error = np.mean(-((va_tar*np.log(a+1e-30))+ (1-va_tar)* np.log(1- a+1e-30)))
				if error <= best_error:
					best_w = weight
					best_error= error
				if ii%100==0:	
					print ('[{}]validation loss: {}'.format(ii , best_error))
			count+= 1			
	np.save(out_model_file, weight)				
