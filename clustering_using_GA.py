#!/usr/bin/python

from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import random 



class CGA:
	def __init__(self,K=3):
		self.K=K
		self.prob_crossover=0.8
		self.DI_crossover=20
		self.prob_mutation=0.2
		self.DI_mutation=20


	def kmeans_plot(self,cluster,center_list,x,y):
		# Set the size of the plot
		plt.figure(figsize=(14,7))

		# Create a colormap
		if(self.K>9):
			print("please add more colors in colormap array at line 33")
			exit(0)
		colormap = np.array(['red', 'lime', 'black','blue','yellow','green','brown','orange','violet'])
		 
		# Plot the Original Classifications
		plt.subplot(1, 2, 1)
		plt.scatter(x[:,0], x[:,1], c=colormap[cluster], s=40)
		plt.title('K Means Clustering')
		plt.ylabel('Sepal_Length')
		plt.xlabel('Sepal_width')
		
		 
		# Plot the Models Classifications
		plt.subplot(1, 2, 2)
		plt.scatter(x[:,2], x[:,3], c=colormap[cluster], s=40)
		plt.title('K Means Clustering')
		plt.ylabel('Petal_Length')
		plt.xlabel('Petal_width')
		plt.show()


	def data_praparation(self):
		data = [line.strip() for line in open('iris.data') if line.strip()]
		features = [tuple(map(float, x.split(',')[:-1])) for x in data]
		labels = [x.split(',')[-1] for x in data]

		return np.array(features), np.array(labels)

	def euclidean_distance(self,w1,x1,y1,z1,w2,x2,y2,z2):
		dist=np.sqrt(   np.power((w2-w1),2) +
						np.power((x2-x1),2) +
						np.power((y2-y1),2) +
						np.power((z2-z1),2) )
		return dist

	def population_init(self,x):
		unique=random.sample(range(0,self.P),self.K)
		pop=np.array([np.ravel([x[unique[i]] for i in xrange(self.K)])
					 for j in xrange(self.P)])
		return pop
	def fit_fun(self,x,center_list):
		fit_value=0
		center_list=center_list.reshape((self.K,self.N))
		for j in xrange(x.shape[0]):
			dist=np.array([ self.euclidean_distance(x[j][0],x[j][1],x[j][2],x[j][3],
							center_list[k][0],center_list[k][1],center_list[k][2],center_list[k][3])
							for k in xrange(self.K) ])
			fit_value=np.add(fit_value,np.sum(dist))
		return fit_value

	def fitness(self,x,center_list):
		fit_value=0
		center_list=center_list.reshape((self.K,self.N))
		cluster=np.zeros(x.shape[0])
		points_in_cluster=np.zeros(self.K)
		for j in xrange(x.shape[0]):
			dist=np.array([ self.euclidean_distance(x[j][0],x[j][1],x[j][2],x[j][3],
							center_list[k][0],center_list[k][1],center_list[k][2],center_list[k][3])
							for k in xrange(self.K) ])
			fit_value=np.add(fit_value,np.sum(dist))
			c=np.argmin(dist)
			cluster[j]=c 
			for k in xrange(self.K):
				if(c==k):
					points_in_cluster[k]+=1
		#print(points_in_cluster)
		avg=np.zeros((self.K,x.shape[1]))
		for j in xrange(x.shape[0]):
			avg[int(cluster[j])][0]+=x[j][0]
			avg[int(cluster[j])][1]+=x[j][1]
			avg[int(cluster[j])][2]+=x[j][2]
			avg[int(cluster[j])][3]+=x[j][3]
		#print(points_in_cluster)
		for j in xrange(self.K):
			avg[j]=np.divide(avg[j],points_in_cluster[j])

		return np.ravel(avg),cluster,np.divide(1,fit_value)


	def evaluate_population(self,x,population):
		#tournament selection
		#print(population.shape)
		new=np.copy(population)
		for i in xrange(self.P):
			first=np.random.randint(0,new.shape[0])
			second=np.random.randint(0,new.shape[0])
			f_cost=self.fitness_value[first]
			s_cost=self.fitness_value[second]
			if(f_cost>s_cost):
				population[i]=new[first]
				new=np.delete(new,(first),axis=0)
			else:
				population[i]=new[second]
				new=np.delete(new,(second),axis=0)
		#singal point crossover
		new=np.copy(population)
		n_child=int(self.prob_crossover*self.P)
		children=np.zeros((n_child,self.N*self.K))
		i=0
		while(i<n_child):
			first=np.random.randint(0,new.shape[0])
			second=np.random.randint(0,new.shape[0])
			c_line=np.random.randint(0,self.N*self.K)
			children[i]=np.append(new[first][:c_line],new[second][c_line:])
			i+=1
			children[i]=np.append(new[second][:c_line],new[first][c_line:])
			i+=1
			new=np.delete(new,[first,second],axis=0)

		#mutation
		new=np.copy(children)
		n_mut=int(self.prob_mutation*self.P)
		for i in xrange(n_mut):
			first=np.random.randint(0,new.shape[0])
			m_line=np.random.randint(0,self.N*self.K)

			delta = random.uniform(0,1)
			v=children[first][m_line]
			if random.uniform(0,1) <= 0.5:
				children[first][m_line] = v + 2*delta*v if v!=0 else 2*delta
			else:
				children[first][m_line] = v + 2*delta*v if v!=0 else 2*delta

			new=np.delete(new,(first),axis=0)
		population=np.append(population,children,axis=0)
		
		new=np.array([[i,j] for i,j in enumerate(np.zeros((self.P+n_child)))])
		#print(population.shape)
		for i in xrange(population.shape[0]):
			#print(np.unique(population[i]))
			cost=self.fit_fun(x,population[i])
			new[i][1]=cost
		new=new[np.argsort(new[:,1])]
		
		#select new P low cost chromosomes
		new_ppl=np.array(np.zeros((self.P,self.N*self.K)))
		for i in xrange(self.P):
			new_ppl[i]=population[int(new[int(self.P-1-i)][0])]
		
		#print(new[:10])
		cost=np.delete(new,(0),axis=1)
		cost=cost[:self.P].reshape(self.P,)
		return new_ppl,cost


	def cga(self,x,maxiter=200):
		self.N=x.shape[1]
		self.P=x.shape[0]
		self.pop_clusters=np.zeros((self.P,self.P))
		self.fitness_value=np.zeros((self.P,1))
		pop=self.population_init(x)
		best_centers=[]
		for j in xrange(maxiter):
			for i in xrange(self.P):
				pop[i], self.pop_clusters[i], self.fitness_value[i] = self.fitness(x,pop[i])

			pop,cost=self.evaluate_population(x,pop)
			best_centers=pop[0]
			if(np.array_equal(best_centers,pop[0])):
				break
			else:
				best_centers=pop[0]
		best_centers, self.pop_clusters[0], _ = self.fitness(x,pop[0])
		return best_centers.reshape((self.K,self.N)), self.pop_clusters[0]

	

def main():
	print("\ncurrently it is using iris flower dataset......\n")
	size=int(input("enter the number of clusters\n"))
	iterr=int(input("enter the number of iterations(min=150)\n"))
	cga=CGA(size)
	x,y=cga.data_praparation()
	centers, cluster=cga.cga(x,iterr)
	#print(centers)
	#print(cluster.shape)
	cga.kmeans_plot(cluster.astype(np.int64),centers,x,y)
	

if __name__=="__main__":main()