# k-means clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# plotting 3D
from mpl_toolkits.mplot3d import Axes3D

# for cluster model # the elbow method (plot)
from sklearn.cluster import KMeans

# for feature scaling
from sklearn.preprocessing import StandardScaler

# for splitting data into trains and tests
from sklearn.model_selection import train_test_split

def k_means():
    # importing data
    dataset = pd.read_csv('Mall_Customers.csv')
    # print(dataset)

    '''
         CustomerID   Genre  Age  Annual Income (k$)  Spending Score (1-100)
    0             1    Male   19                  15                      39
    1             2    Male   21                  15                      81
    [200 rows x 5 columns]
    '''

    # encoding Genre
    gender = {'Male': 1, 'Female': 2}
    dataset.Genre = [gender[Item] for Item in dataset.Genre]

    '''
         CustomerID  Genre  Age  Annual Income (k$)  Spending Score (1-100)
    0             1      1   19                  15                      39
    1             2      1   21                  15                      81
    '''

    # independent
    independent = dataset.iloc[:, [2, 3, 4]].values

    # no dependent variable as we are creating the variable

    # plotting 3D scatter with raw data the elbow method
    fig = plt.figure()
    ax = Axes3D(fig) # == fig.add_subplot(111, projection='3d')
    item = np.asarray(dataset['Genre']) # array values can return
    row = 0 # travel in the rows
    p1, p2 = ax.scatter(xs=0, ys=0, zs=0), ax.scatter(xs=0, ys=0, zs=0) # creating scatter for the loop
    p1.remove(), p2.remove() # remove 0 for the plotting
    for i in item:
        if i == 1: #male
            x1, y1, z1 = dataset.at[row, 'Age'], dataset.at[row, 'Annual Income (k$)'], dataset.at[row, 'Spending Score (1-100)']
            p1 = ax.scatter(xs=x1, ys=y1, zs=z1, marker='o', c='b')
        elif i == 2: #femnale
            x2, y2, z2 = dataset.at[row, 'Age'], dataset.at[row, 'Annual Income (k$)'], dataset.at[row, 'Spending Score (1-100)']
            p2 = ax.scatter(xs=x2, ys=y2, zs=z2, marker='x', c='r')
        row = row + 1 # must have +1 for traveling the rows
    np.meshgrid(dataset['Age'], dataset['Annual Income (k$)'])
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Income (k$)')
    ax.set_zlabel('Spending Score (1-100)')
    ax.legend([p1, p2],['Blue: Male', 'Red: Female'])
    plt.show()

    # Using the elbow method to find te optimal number of clusters
    # create 10 different clusters
    wcss = []
    for i in range (1, 11):
        Kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42) # N_cluster is up to 10 and avoiding k_means trap with 'k-means++'
        Kmeans.fit(independent) # no y (dependent variable)
        wcss.append(Kmeans.inertia_) # inertia_ gives WCSS values 
    # plotting    
    plt.plot(range (1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Num of Cluster')
    plt.ylabel('WCSS')
    plt.show()

    # from the plot, we got 5 is optimal

    # training dataset
    Kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)

    dependent_pred = Kmeans.fit_predict(independent)
    print(dependent_pred) # tells where the customer belongs to the clusters

    # Visualization the clusters
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    row = 0
    item2 = np.asarray(dependent_pred)
    for i, j in zip(item, item2):
        if i == 1: # male
            if j == 0:
                x1, y1, z1 = dataset.at[row, 'Age'], dataset.at[row, 'Annual Income (k$)'], dataset.at[row, 'Spending Score (1-100)']
                ax2.scatter(xs=x1, ys=y1, zs=z1, marker='o', c='b')
            elif j == 1:
                x2, y2, z2 = dataset.at[row, 'Age'], dataset.at[row, 'Annual Income (k$)'], dataset.at[row, 'Spending Score (1-100)']
                ax2.scatter(xs=x2, ys=y2, zs=z2, marker='o', c='r')
            elif j == 2:
                x3, y3, z3 = dataset.at[row, 'Age'], dataset.at[row, 'Annual Income (k$)'], dataset.at[row, 'Spending Score (1-100)']
                ax2.scatter(xs=x3, ys=y3, zs=z3, marker='o', c='pink')
            elif j == 3:
                x4, y4, z4 = dataset.at[row, 'Age'], dataset.at[row, 'Annual Income (k$)'], dataset.at[row, 'Spending Score (1-100)']
                ax2.scatter(xs=x4, ys=y4, zs=z4, marker='o', c='y')
            elif j == 4:
                x5, y5, z5 = dataset.at[row, 'Age'], dataset.at[row, 'Annual Income (k$)'], dataset.at[row, 'Spending Score (1-100)']
                ax2.scatter(xs=x5, ys=y5, zs=z5, marker='o', c='g')
        elif i == 2: #female
            if j == 0:
                x1, y1, z1 = dataset.at[row, 'Age'], dataset.at[row, 'Annual Income (k$)'], dataset.at[row, 'Spending Score (1-100)']
                ax2.scatter(xs=x1, ys=y1, zs=z1, marker='x', c='b')
            elif j == 1:
                x2, y2, z2 = dataset.at[row, 'Age'], dataset.at[row, 'Annual Income (k$)'], dataset.at[row, 'Spending Score (1-100)']
                ax2.scatter(xs=x2, ys=y2, zs=z2, marker='x', c='r')
            elif j == 2:
                x3, y3, z3 = dataset.at[row, 'Age'], dataset.at[row, 'Annual Income (k$)'], dataset.at[row, 'Spending Score (1-100)']
                ax2.scatter(xs=x3, ys=y3, zs=z3, marker='x', c='pink')
            elif j == 3:
                x4, y4, z4 = dataset.at[row, 'Age'], dataset.at[row, 'Annual Income (k$)'], dataset.at[row, 'Spending Score (1-100)']
                ax2.scatter(xs=x4, ys=y4, zs=z4, marker='x', c='y')
            elif j == 4:
                x5, y5, z5 = dataset.at[row, 'Age'], dataset.at[row, 'Annual Income (k$)'], dataset.at[row, 'Spending Score (1-100)']
                ax2.scatter(xs=x5, ys=y5, zs=z5, marker='x', c='g')
        row = row + 1
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Annual Income (k$)')
    ax2.set_zlabel('Spending Score (1-100)')
    plt.show()


if __name__ == '__main__':
    k_means()