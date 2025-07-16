#take in inputs for the data files
#import argparse
#parser=argparse.ArgumentParser(description="input data files")
#parser.add_argument("minima_data_path", type=str)
#parser.add_argument("ts_data_path", type=str)
#args=parser.parse_args()

#import libraries
import igraph
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load in data files
min_data = pd.read_table('C:/Users/ckcho/OneDrive/Desktop/KCL Bioinformatics/Research_project/TAR/min.data', header=None, sep='\s+')
ts_data = pd.read_table('C:/Users/ckcho/OneDrive/Desktop/KCL Bioinformatics/Research_project/TAR/ts.data', header=None, sep='\s+')
print(min_data.shape, ts_data.shape)

from collections import defaultdict
#slice out free energy values and indices
min_energy = min_data[0]
min_energy.index +=1

ts_index = ts_data[[3,4]]
ts_energy = ts_data[[0]]
ts_index.index +=1
ts_energy.index +=1

#add extra row with dummy data to ts indexing data to ensure indexing consistency
ts_index.loc[0] = [0,0]
ts_index.sort_index(inplace=True) 

#add extra row with dummy data to ts energy data to ensure indexing consistency
ts_energy.loc[0] = [np.nan]
ts_energy.sort_index(inplace=True) 
ts_energy = list(zip(ts_energy[0]))

# igraph uses vertices and edge IDs, not actual values so use index as IDs
from igraph import Graph
g = Graph()
g = Graph.DataFrame(ts_index, directed=False)
print(g)

#assign minima energy to the verticies
for i in range(1,(len(min_energy)+1)):
    g.vs[i]["energy"] = [min_energy[i]]

#calculate edge weights
edge_weights = list()
for i in range (1, (len(ts_index))):
    temp = abs(ts_energy[i]-(np.nanmin(min_energy)))
    edge_weights.append(temp)

#insert a zero to weight the dummy edge
edge_weights.insert(0, np.float64(0))

#loop to assign weights to each edge/transition state
for i in range (0, (len(ts_energy))):
    g.es[i]["weight"] = [edge_weights[i]]

#check if graph is now weighted
g.is_weighted()

#function for randomly picking out a given number of minima pairs
import random
import itertools

def image_generator(iterable, groups, samplesize):
    grouped = (i for i in itertools.product(iterable,repeat=groups) if i[0]!=i[1])
    return random.sample(list(grouped), samplesize)

sample = image_generator(min_energy, 2, 50)

#splitting selected pairs into 1st half and 2nd half for graphing and calculation later
sample1 = list(list(zip(*sample))[0])
sample2 = list(list(zip(*sample))[1])

#loop to find the energy differences between each pair
energy_diff=list()
for i in range(0,len(sample)):
    temp = abs(sample1[i]-sample2[i])
    energy_diff.append(temp)

#filter that removes energy values greater than 150 from the energy differences
for i in range(0, 50):
    if abs(energy_diff[i]) > 150:
        sample.remove(sample[i])
        sample1.remove(sample1[i])
        sample2.remove(sample2[i])
        energy_diff.remove(energy_diff[i])
    else:
        break

#match energies from chosen minima pairs with original min energy and get indicies
start = [i for i, item in enumerate(min_energy) if item in sample1]
end = [i for i, item in enumerate(min_energy) if item in sample2]

#getting list of edges and verticies for the shortest paths 
#this outputs a list of lists of lists
#the outer list containts 50 lists for each starting vertex, the inner lists contain paths starting from a starting vertex and every other ending vertex
#the amount of paths generated from this can vary depending on the minima chosen as sometimes paths between certain minima cannot be found, from testing this can range from around 40 paths to over 70 paths
edge_paths = list()
vertex_paths = list()
for i in range(0, len(start)):
    e_results = g.get_shortest_paths(v=start[i], to=end, weights=edge_weights, output="epath") #getting the edge list for each path found
    v_results = g.get_shortest_paths(v=start[i], to=end, weights=edge_weights, output="vpath") #getting the vertex list for each path found
    edge_paths.append(e_results)
    vertex_paths.append(v_results)

#converting the vertex IDs into actual energies by using the vertex IDs as an index for the minima energy list
minima = list()
for i in vertex_paths: #iterating through each starting vertex list
    for v in i: #iterate through every path on the list 
        minima.append(min_energy[v]) #using the vertex ID as an reference index, find the relevant values from the min energy list and append them to a new list

#identify the lengths of each starting verticies list, which should correspond to the number of paths for each starting vertex
length = list()
for i in vertex_paths:
    length.append(len(i))

#convert from a pandas series to np array to avoid weird data type interactions and looping logic down the line
ts_energy = np.array(ts_energy)

#converting the edgeIDs into actual energies by using the edge IDs as an index for the transition state energy list
tstates = list()
for i in edge_paths: #iterating through each edge list for each starting vertex
    for v in i: #iterate through every path on the list 
        tstates.append(ts_energy[v]) #using the edge ID as an reference index, find the relevant values from the transition state energy list and append them to a new list

#function to flatten a list of lists
def flatten(xss):
    return [x for xs in xss for x in xs]

#the way that the transition states were sliced earlier puts each value into their own array single element list, this makes working with them really awkward
#this loop fixes that issue using the flatten function
new_tstates = list()
for i in tstates:
    temp = flatten(i)
    new_tstates.append(temp)

#splitting the transition state energy lists into chunks according to the number of vertex paths for each starting vertex
new_tstates = [new_tstates[i:i+(length[0])] for i in range(0,len(new_tstates),(length[0]))]

#getting the max transitions state for each path 
max_energy = list()
max_val = list()
for starting_points in new_tstates: #iterating through every starting vertex list of paths
    for paths in starting_points: #iterating through every path
        for ts in paths: #iterating across every transition state in each path
            max_val = paths[0] #setting the initial max transition state energy to be the 1st energy in the current path
            if ts > max_val: #checking if the element (i.e energy) the loop is currently on is larger than the current transition state energy
                max_val = ts #if the current energy is larger than the current max energy then replace that value
        max_energy.append(max_val) #once fully iterated over a path, append the max energy for that path to a list

#splitting the max transition state energies into chunks using the known number of paths for each starting vertex so i can compare them for each starting vertex as the indicies should match up
max_energy = [max_energy[i:i+(length[0])] for i in range(0,len(max_energy),(length[0]))]

#finding the max energies across each path in each starting vertex list
final_max_energy = list()
for starting_points in max_energy: #iterate across the the ~50 or so starting vertex lists
    temp = max(starting_points) #for each path in the list find the max value
    final_max_energy.append(temp) #append the max value to a new list to get a list of highest values in each starting vertex

#get each edge list/path containing the max transition state energy
index = list()
for i in range(len(final_max_energy)): #iterate for the length of the max energy list, which should be around 50 as it contains the max energy for every starting vertex 
    starting_points = new_tstates[i] #using the number of i make sure the index of both the max energy list and the ts energy list match up, so the max energy and transition states for each starting vertex matches up 
    max_e = final_max_energy[i]  #ensuring that max energies from other starting points are not used
    for path in starting_points: #iterate over each path in the starting vertex list
        for ts in path: # iterate over each transition state in each path
            if (ts == max_e).any(): #check if any of the transition states in the list match with the max energy  
                index.append(path) #if they transition state and max energy are the same, then this path contains the max energy and can be appended to a new list

#lists are flattened from this point forward we have a list of edge paths that contain the max energies, so we can flatten the vertex and edge path list of lists of lists into a list of ~1000 arrays and then match them, knowing the starting vertex should not matter past this point any more
flat_v_path = flatten(vertex_paths)
flat_e_path = flatten(edge_paths)
new_tstates = flatten(new_tstates)

#match the list of edge paths containing the highest transition state energies with the flattened list of all edge paths and get their indices
edge_path_index = list()
for i in index:
    for x in range(len(new_tstates)):
        if len(new_tstates[x]) != len(i):
            print ('Path lengths do not match')
        else:
            path = new_tstates[x]
            if path == i:
                edge_path_index.append(x)

#the indices of the edge paths and the vertex paths should be identical (ie: path 1 in both edge and vertex path lists refer to the same path)
#this means we can use the edge list index to find these paths and combine them together in an alternating way to regenerate the full path of verticies and edges
training_data = list()
for i in edge_path_index: #loop over the list of indices
    temp = [x for x in itertools.chain.from_iterable(itertools.zip_longest(flat_v_path[i], flat_e_path[i])) if x] #combine the lists of indices that match together in an alternating way
    training_data.append(temp) #append the newly regenerated path to a new list

#write each full path to a separate file (so if there are 50 paths, 50 files will be created)
#additionally, the length of the path will be added to the start of the file
for i in range(len(training_data)): #loop over the number of paths in the training data
    length = len(training_data[i]) #get the length of each path
    with open(f"path{i+1}.txt", "w") as file: #write the path to a text file
        file.write(str(length) + "\n") #write the length of the path to 1st line of the file
        file.write("\n".join(str(n) for n in training_data[i]) + "\n") #put each element of the list (ie: each vertex and edge) on a new line
print('files written')
