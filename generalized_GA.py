#!/usr/bin/python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import *

class Real_GA:
	def __init__(self,N=10,features=1):
		self.seed=10
		self.prob_crossover=0.8
		self.DI_crossover=20
		self.prob_mutation=0.2
		self.DI_mutation=20
		self.N=N
		self.features=features
		

	def data_genration(self):
		#np.random.seed(self.seed)
		population=np.random.uniform(0,2*np.pi,size=(self.N,self.features))
		self.population=population
		return population

	def set_function(self,y):
		self.y=y

	def f(self,x):
		yy=self.y
		for i in xrange(self.features):
			bkm=str('x')+str(i+1)
			yy=yy.replace(str(bkm),str(x[i]))
			#print(yy)
		yy=eval(yy)
		#print("inside {}".format(yy))
		return yy
	

	def check_boundary(self,x):
		if(x>=0 and x<2*np.pi):
			return x
		else:
			#return x
			return np.random.rand(0,2*np.pi)
		
	def evaluate_population(self,population):
		#tournament selection
		new=np.copy(population)
		for i in xrange(self.N):
			first=np.random.randint(0,new.shape[0])
			second=np.random.randint(0,new.shape[0])
			#print("calling")
			f_cost=self.f(new[first])
			#print("f_cost {}".format(f_cost))
			s_cost=self.f(new[second])
			#print("s_cost {}".format(s_cost))
			if(f_cost < s_cost):
				population[i]=new[first]
				new=np.delete(new,(first),axis=0)
			else:
				population[i]=new[second]
				new=np.delete(new,(second),axis=0)
		#singal point crossover
		new=np.copy(population)
		n_child=int(self.prob_crossover*self.N)
		children=np.array(np.zeros((n_child,self.features)))
		i=int(0)
		while(i<n_child):
			first=np.random.randint(0,new.shape[0])
			second=np.random.randint(0,new.shape[0])
			r=np.random.rand()
			if(r<=0.5):
				b=np.power( (2*r), float((1/(self.DI_crossover + 1))) )
			else:
				b=np.power( (1/(2*(1-r))), (1/(self.DI_crossover + 1)) )
			#print(float((1/(self.DI_crossover + 1)))
			for j in xrange(self.features):
				children[i][j]=self.check_boundary((1/2)*((1+int(b))*population[int(first)][j] + (1-int(b))*population[int(second)][j]))
				#print((1/2)*((1+b)*population[first] + (1-b)*population[second]))
			i+=int(1)
			for j in xrange(self.features):
				children[i]=self.check_boundary((1/2)*((1-int(b))*population[int(first)][j] + (1+int(b))*population[int(second)][j]))
			i+=int(1)

			new=np.delete(new,[int(first),int(second)],axis=0)
		#print(children)
		
		#mutation
		new=np.copy(children)
		n_mut=int(self.prob_mutation*self.N)
		for i in xrange(n_mut):
			first=np.random.randint(0,new.shape[0])
			r=np.random.rand()
			if(r<=0.5):
				d=np.power((2*r),(1/(self.DI_mutation + 1))) - 1
			else:
				d=1 - np.power((2*(1-r)),(1/(self.DI_mutation + 1)))
			for j in xrange(self.features):
				children[first][j]=self.check_boundary(children[first][j] + int(d))

			new=np.delete(new,(first),axis=0)
		population=np.append(population,children,axis=0)
		#print(children)
		#print("tournament over")
		new=np.array([[i,j] for i,j in enumerate(np.zeros((self.N+n_child)))])
		#print(new)
		for i in xrange(population.shape[0]):
			cost=self.f(population[i])
			new[i][1]=cost
		new=new[np.argsort(new[:,1])]
		#print(new)
		#select new N low cost chromosomes
		new_ppl=np.array(np.zeros((self.N,self.features)))
		for i in xrange(self.N):
			new_ppl[i]=population[int(new[i][0])]
		
		
		cost=np.delete(new,(0),axis=1)
		cost=cost[:self.N].reshape(self.N,)
		#print(cost[:10])
		#print(new_ppl)
		return new_ppl,cost


def main():
	print("\n******************************************\n")
	print("Enter the function equation....\ne.g. if function is y=sin(x) then enter sin(x1)")
	print("or if function is y=sin(x1)+cos(x2) then enter sin(x1)+cos(x2)")
	print("give x as x1,x2,x3... always")
	print("don't give input like sin(x)+cos(y) etc...")
	print("\n******************************************\n")
	fun=raw_input("y = ")
	print("\n******************************************")
	print("number of features is number of variable in function equation")
	print("******************************************\n")
	print("enter the number of features (# variables in one example)\n")
	features=int(input())
	size=int(input("enter the size of the population\n"))
	rga=Real_GA(size,features)
	rga.set_function(fun)
	ppl=rga.data_genration()
	ppl,cost=rga.evaluate_population(ppl)
	print("\nfunction is minimum at given value and cost....")
	print("value : {}, cost : {}".format(ppl[0],cost[0]));
	plt.show()

if __name__=="__main__":
	main()

