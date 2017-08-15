#!/usr/bin/python

from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt 


class Kmeans:
	def __init__(self,centers=3):
		self.centers=centers

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

	def kmeans_plot(self,cluster,center_list,x,y):
		# Set the size of the plot
		plt.figure(figsize=(14,7))

		# Create a colormap
		if(self.centers>9):
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


	def clustering(self,x,iter=10):
		center_list=np.array([x[np.random.randint(0,x.shape[0])] for i in xrange(self.centers)])

		for i in xrange(iter):
			cluster=np.zeros(x.shape[0])
			points_in_cluster=np.zeros(self.centers)
			for j in xrange(x.shape[0]):
				dist=np.array([ self.euclidean_distance(x[j][0],x[j][1],x[j][2],x[j][3],
								center_list[k][0],center_list[k][1],center_list[k][2],center_list[k][3])
								for k in xrange(self.centers) ])
				c=np.argmin(dist)
				cluster[j]=c 
				for k in xrange(self.centers):
					if(c==k):
						points_in_cluster[k]+=1
			#print(points_in_cluster)
			#print(cluster)
			#print(cluster)
			avg=np.zeros((self.centers,x.shape[1]))
			for j in xrange(x.shape[0]):
				avg[int(cluster[j])][0]+=x[j][0]
				avg[int(cluster[j])][1]+=x[j][1]
				avg[int(cluster[j])][2]+=x[j][2]
				avg[int(cluster[j])][3]+=x[j][3]
			#print(avg)
			for j in xrange(self.centers):
				avg[j]=np.divide(avg[j],points_in_cluster[j])

			#print(avg)

			if(np.array_equal(center_list,avg)):
				break
			else:
				center_list=avg

		return cluster,center_list


def main():
	size=int(input("enter the number of clusters\n"))
	k=Kmeans(size)
	x,y=k.data_praparation()
	cluster,center_list=k.clustering(x,100)
	#print(cluster.shape)
	k.kmeans_plot(cluster.astype(np.int64),center_list,x,y)

if __name__=="__main__":main()