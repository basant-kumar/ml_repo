#!/usr/bin/python

import numpy as np 


class SI_LMS:
	def __init__(self, features=1, dataset_size=500, nodes=3, iterations=100):
		self.nodes=nodes
		self.features=features
		self.dataset_size=dataset_size
		self.nodes=nodes
		self.dataset=None
		self.iterations=iterations
		self.learning_rate=0.01

	def data_preparation(self):
		dataset=np.random.rand(self.dataset_size,self.features) - 0.5
		z1=np.zeros((self.nodes-1))
		z2=np.zeros((self.nodes-1))
		w=np.random.rand(self.nodes)
		self.z1=z1
		self.z2=z2
		self.w=w
		self.dataset=dataset

	def direct_model(self):
		dataset=self.dataset
		for i in range(self.iterations):
			avg=np.zeros(self.features)
			self.z1=np.zeros((self.nodes-1))
			self.z2=np.zeros((self.nodes-1))
			for j in range(self.dataset_size):
				y_orig=self.original_model(dataset[j])
				y_simu=self.simulated_model(dataset[j])
				print(y_orig)
				print(y_simu)
				error=y_orig - y_simu
				#print(error)
				last_w=self.w
				self.w=self.w + (2*self.learning_rate*dataset[j]*error)
				#print(self.w)
				avg+=(np.sum(error)/len(error))
			if((last_w==self.w).all()):
				print("Done")
			#print("iteration:{}  Error:{}".format(i,avg/len(avg)))



	def original_model(self,x):
		total_sum=0.0
		total_sum+=np.sin(x)
		total_sum+=np.sin(self.z1[0])
		total_sum+=np.sin(self.z1[1])
		self.z1[0]=x
		self.z1[1]=self.z1[0]
		return total_sum

	def simulated_model(self,x):
		total_sum=0.0
		total_sum+=x*self.w[0]
		total_sum+=self.z2[0]*self.w[1]
		total_sum+=self.z2[1]*self.w[2]
		self.z2[0]=x
		self.z2[1]=self.z2[0]
		return total_sum
		



def main():
	si=SI_LMS(features=1,dataset_size=10,nodes=3,iterations=100)
	si.data_preparation()
	si.direct_model()

if __name__=="__main__":
	main()