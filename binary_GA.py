#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

class Binary_GA:
	def __init__(self,N=10,c_size=10):
		self.seed=10
		self.prob_crossover=0.8
		self.DI_crossover=20
		self.prob_mutation=0.2
		self.DI_mutation=20
		self.N=N
		self.c_size=c_size

	def data_genration(self):
		#np.random.seed(self.seed)
		population=np.random.rand(self.N,self.c_size)
		population=self.convert_to_binary(population)
		self.population=population
		return population
		
	def convert_to_binary(self,population):
		for i in xrange(self.N):
			for j in xrange(self.c_size):
				if(population[i][j]>0.5):
					population[i][j]=1
				else:
					population[i][j]=0
		return population

	def bool2int(self,x):
		return sum(1<<i for i,b in enumerate(x) if b)
		'''#same as
			y = 0
		    for i,j in enumerate(x):
		        y += j<<i
		    return y
		'''

	def f(self,x):
		y = np.sin(x)
		return y

	def evaluate_population(self,population):
		#tournament selection
		new=np.copy(population)
		for i in xrange(self.N):
			first=np.random.randint(0,new.shape[0])
			second=np.random.randint(0,new.shape[0])
			f_cost=self.f(self.bool2int(list(new[first][::-1])))
			s_cost=self.f(self.bool2int(list(new[second][::-1])))
			if(f_cost<s_cost):
				population[i]=new[first]
				new=np.delete(new,(first),axis=0)
			else:
				population[i]=new[second]
				new=np.delete(new,(second),axis=0)
		#singal point crossover
		new=np.copy(population)
		n_child=int(self.prob_crossover*self.N)
		children=np.array(np.zeros((n_child,self.c_size)))
		i=0
		while(i<n_child):
			first=np.random.randint(0,new.shape[0])
			second=np.random.randint(0,new.shape[0])
			c_line=np.random.randint(0,10)
			children[i]=np.append(new[first][:c_line],new[second][c_line:])
			i+=1
			children[i]=np.append(new[second][:c_line],new[first][c_line:])
			i+=1
			new=np.delete(new,[first,second],axis=0)

		#mutation
		new=np.copy(children)
		n_mut=int(self.prob_mutation*self.N)
		for i in xrange(n_mut):
			first=np.random.randint(0,new.shape[0])
			m_line=np.random.randint(0,10)
			if(children[first][m_line]==0):
				children[first][m_line]=1
			else:
				children[first][m_line]=0
			new=np.delete(new,(first),axis=0)
		population=np.append(population,children,axis=0)
		
		new=np.array([[i,j] for i,j in enumerate(np.zeros((self.N+n_child)))])
		for i in xrange(population.shape[0]):
			cost=self.f(self.bool2int(list(population[i][::-1])))
			new[i][1]=cost
		new=new[np.argsort(new[:,1])]
		
		#select new N low cost chromosomes
		new_ppl=np.array(np.zeros((self.N,self.c_size)))
		for i in xrange(self.N):
			new_ppl[i]=population[new[i][0]]
		
		#print(new[:10])
		cost=np.delete(new,(0),axis=1)
		cost=cost[:self.N].reshape(self.N,)
		return new_ppl,cost


			



def main():
	bga=Binary_GA(50,10)
	ppl=bga.data_genration()
	avg_cost=[]
	for i in range(50):
		ppl,cost=bga.evaluate_population(ppl)
		avg_cost=np.append(avg_cost,np.sum(cost)/50)
	
	plt.plot(avg_cost,'bo')
	plt.xlabel('x')
	plt.ylabel('f(x)')
	
	plt.show()

if __name__=="__main__":main()

