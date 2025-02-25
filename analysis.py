
import sys
import numpy as np
from sklearn.metrics import silhouette_score
from symnmfmodule import symnmf, sym 


def read_data(file_name):
    """
    Reads input data from a file and returns it as a NumPy array.
    """
    try:
        data = np.loadtxt(file_name, delimiter=',')
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)
    return data


def Kmeans( k ,iter  , input_data  ):
    Validate_input(iter,  input_data )
    #print("Valid input")
    data_points = Read_input_data(input_data)
    #print("Data points read /n" , data_points)
    centroids = Initialize_centroids(k, data_points)
    #print("Centroids initialized /n" , centroids)
    Converged = False
    i=0
    while (i<iter and not  Converged):
        clusters = Assign_clusters(data_points, centroids)
        new_centroids = Calculate_centroids(clusters)
        Converged = Check_convergence(centroids, new_centroids)
        centroids = new_centroids
        i += 1
    if centroids is None:
        raise ValueError("NA")
    else :
        printer(centroids)
    return None
    
"""validate the input data
    - the number of iterations should be between 1 and 1000
    - the input data should be a text file
"""
def Validate_input(iter,  input_data ):
    
    if not (1<iter<1000):
        raise ValueError("Invalid number of iterations!")
    
    if not input_data.endswith('.txt'):
        raise ValueError("NA")
    
"""read the input data from the file and return a list of data points"""      
def Read_input_data(input_data):
 
    data_points = []
    try:
        with open (input_data, 'r') as file :
            for line in file :
                vector= line.strip().split(',')
                data_point=[float(x) for x in vector]
                data_points.append(data_point)
    except Exception as e:
        raise Exception("An Error Has Occurred")
    return data_points
              
"""initialize the centroids by assigning the first k data points to the first k centroids"""
def Initialize_centroids(k, data_points):
    N= len(data_points)
    if not  (1<k<N):
        raise ValueError("Invalid number of clusters!")
    centroids = []
    for i in range(k):
        centroids.append(data_points[i])
    return centroids

"""assign each data point to the nearest centroid"""
def Assign_clusters(data_points, centroids):
    clusters = {i: [] for i in range(len(centroids))}
    for data_point in data_points:
        nearest_centroid = min([(i, Calculate_distance(data_point, centroid)) for i, centroid in enumerate(centroids)], key=lambda x: x[1])[0]
        clusters[nearest_centroid].append(data_point)
    return clusters

"""a helper function to calculate the distance between a data point and a centroid"""
def Calculate_distance(data_point, centroid):
    distance = math.sqrt(sum([(x-y)**2 for x, y in zip(data_point, centroid)]))
    return distance

"""calculate the new centroid of each cluster"""
def Calculate_centroids(clusters):
    new_centroids = []
    for cluster in clusters.values():
        new_centroids.append([sum(x)/len(cluster) for x in zip(*cluster)])
    return new_centroids

"""check if all  centroids have converged such that epsilon is 0.001"""
def Check_convergence(centroids, new_centroids):
   return all([Calculate_distance(centroid, new_centroid) < 0.001 for centroid, new_centroid in zip(centroids, new_centroids)])

"""rounds the centroids to 4 decimal places and prints them as a comma-separated list"""
def printer(centroids):
    for centroid in centroids:
        centroid = [round(x, 4) for x in centroid if x is not None]
        print(",".join(map(str, centroid)).strip('[]'))
    return None

def symnmf_clustering(data, k):
    """
    Perform SymNMF clustering using the provided symnmfmodule.
    """
    similarity_matrix = sym(data.tolist())
    initial_H = np.random.uniform(0, 2 * np.sqrt(np.mean(similarity_matrix) / k), (similarity_matrix.shape[0], k))
    final_H = symnmf(similarity_matrix, initial_H, k)
    cluster_labels = np.argmax(final_H, axis=1)  # Assign clusters based on highest association
    return cluster_labels


def main():
    if len(sys.argv) != 3:
        print("An Error Has Occurred")
        sys.exit(1)

    k = int(sys.argv[1])
    file_name = sys.argv[2]

    # Load the data
    data = read_data(file_name)

    # Perform SymNMF clustering
    symnmf_labels = symnmf_clustering(data, k)
    symnmf_score = silhouette_score(data, symnmf_labels)

    # Perform K-means clustering
    kmeans_labels = Kmeans(k, 200, file_name)
    kmeans_score = silhouette_score(data, kmeans_labels)

    # Print results
    print(f"nmf: {symnmf_score:.4f}")
    print(f"kmeans: {kmeans_score:.4f}")


if __name__ == "__main__":
    main()
