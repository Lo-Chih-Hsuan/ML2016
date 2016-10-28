import numpy as np
import pandas as pd
import sys

if __name__== '__main__':
	weight= np.load(sys.argv[1])
	test_data= np.array(pd.read_csv(sys.argv[2], header= None))
	test_list= []
	for test in test_data:
		test= np.append([1], test[1:])
		test_list.append(test)
	test_list= np.array(test_list)
	predict= test_list
	for l in range(len(weight)):
		predict = 1.0/ (1.0+np.exp(-np.dot(predict, weight[l])))
		if l == 0:
			predict.T[0].fill(1.0)
	predict= predict.reshape(len(predict))
	for ii in range(len(predict)):
		if predict[ii] < 0.5:
			predict[ii]= int(0)
		else:
			predict[ii]= int(1)
	predict= predict.astype(int)
	predict= np.hstack(('label', predict))
	
	id_list= ['id']
	for i in range(1, len(test_data)+1):
		id_list.append(i)
	table= pd.Series(predict, index= id_list)
	table.to_csv(sys.argv[3])
		
