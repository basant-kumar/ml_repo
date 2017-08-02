#!/usr/bin/python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

class Real_GA:
	def __init__(self,N=10):
		self.seed=10
		self.prob_crossover=0.8
		self.DI_crossover=20
		self.prob_mutation=0.2
		self.DI_mutation=20
		self.N=N
		

	def data_genration(self):
		#np.random.seed(self.seed)
		population=np.random.uniform(0,2*np.pi,size=self.N)
		self.population=population
		return population

	def f(self,x):
		y = np.sin(x)
		return y

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
			f_cost=self.f(new[first])
			s_cost=self.f(new[second])
			if(f_cost < s_cost):
				population[i]=new[first]
				new=np.delete(new,(first),axis=0)
			else:
				population[i]=new[second]
				new=np.delete(new,(second),axis=0)
		#singal point crossover
		new=np.copy(population)
		n_child=int(self.prob_crossover*self.N)
		children=np.array(np.zeros((n_child)))
		i=0
		while(i<n_child):
			first=np.random.randint(0,new.shape[0])
			second=np.random.randint(0,new.shape[0])
			r=np.random.rand()
			if(r<=0.5):
				b=np.power( (2*r), float((1/(self.DI_crossover + 1))) )
			else:
				b=np.power( (1/(2*(1-r))), (1/(self.DI_crossover + 1)) )
			#print(float((1/(self.DI_crossover + 1)))
			children[i]=self.check_boundary((1/2)*((1+b)*population[first] + (1-b)*population[second]))
			#print((1/2)*((1+b)*population[first] + (1-b)*population[second]))
			i+=1
			children[i]=self.check_boundary((1/2)*((1-b)*population[first] + (1+b)*population[second]))
			i+=1

			new=np.delete(new,[first,second],axis=0)
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

			children[first]=self.check_boundary(children[first] + d)

			new=np.delete(new,(first),axis=0)
		population=np.append(population,children,axis=0)
		#print(children)
		new=np.array([[i,j] for i,j in enumerate(np.zeros((self.N+n_child)))])
		for i in xrange(population.shape[0]):
			cost=self.f(population[i])
			new[i][1]=cost
		new=new[np.argsort(new[:,1])]
		#print(new)
		#select new N low cost chromosomes
		new_ppl=np.array(np.zeros((self.N)))
		for i in xrange(self.N):
			new_ppl[i]=population[new[i][0]]
		
		
		cost=np.delete(new,(0),axis=1)
		cost=cost[:self.N].reshape(self.N,)
		#print(cost[:10])
		return new_ppl,cost


def main():
	rga=Real_GA(10)
	ppl=rga.data_genration()
	cost=[rga.f(x) for x in ppl]
	avg_cost=[]
	plt.figure(1)
	x=np.arange(0,2*np.pi,0.1)
	y=np.sin(x)

	plt.subplot(221)	
	plt.title("initial state")
	plt.plot(x,y,'k',ppl,cost,'bo')
	plt.xlabel('x')
	plt.ylabel('sin(x)')

	for i in range(30):
		ppl,cost=rga.evaluate_population(ppl)
		if(i==4):
			plt.subplot(222)
			plt.title("after 5 generation")
			plt.plot(x,y,'k',ppl,cost,'bo')
			plt.xlabel('x')
			plt.ylabel('sin(x)')
		elif(i==9):
			plt.subplot(223)
			plt.title("after 10 generation")
			plt.plot(x,y,'k',ppl,cost,'bo')
			plt.xlabel('x')
			plt.ylabel('sin(x)')
		elif(i==29):
			plt.subplot(224)
			plt.title("after 30 generation")
			plt.plot(x,y,'k',ppl,cost,'bo')
			plt.xlabel('x')
			plt.ylabel('sin(x)')
	
	print(ppl)
	print(cost)
	plt.show()

if __name__=="__main__":
	main()

