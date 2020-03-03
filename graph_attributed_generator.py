import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import torch

#PARAMETERS
no_of_Communities = 4
no_of_nodeattributes = 4
no_of_nodes = 100
in_community_probability= 0.6
out_community_probability = 0.2
null_attribute_prob = 0.1
beta = 0.3

Nodes = list(range(0,no_of_nodes))
Graph = torch.zeros([no_of_nodes,no_of_nodes], dtype=torch.int32)
size_of_community = int(no_of_nodes/no_of_Communities)

#CREATE COMMUNITIES
community_nodes = torch.zeros([no_of_Communities, size_of_community], dtype=torch.int32)
print(community_nodes.shape)
for c in range(0,no_of_Communities):
    ran = random.sample(Nodes,size_of_community)
    Nodes = [i for i in Nodes if (i in Nodes and i not in ran)]
    community_nodes[c] = torch.IntTensor(ran)
#print(community_nodes)

#CREATE ATTRIBUTE SET
attribute_set = torch.zeros([no_of_Communities, no_of_nodeattributes], dtype=torch.int32)
for a in range(0,no_of_Communities):
    ran = []
    for b in range(0, no_of_nodeattributes):
        ran.append(random.randint(0,1))
    attribute_set[a] = torch.IntTensor(ran)
#print(attribute_set)

#ASSIGN ATTRIBUTES TO EACH NODE
attribute_matrix = torch.zeros([no_of_nodes, no_of_nodeattributes], dtype=torch.int32)
i = 0

for community in community_nodes:
    for node in community:
        attribute_matrix[node]= attribute_set[i]
    i+=1

no_of_edges = 0
#ASSIGN EDGES INSIDE COMMUNITY
adj_matrix= torch.zeros([no_of_nodes, no_of_nodes], dtype=torch.int32)
for i in range(0,no_of_Communities):
    community = community_nodes[i]
    for node1_index in range(0,len(community)):
        for node2_index in range(node1_index+1,len(community)):
            node1 = community[node1_index]
            node2 = community[node2_index]
            if node1==node2:
                adj_matrix[node1][node2] = 1
                no_of_edges+=1
            else:
                r = random.randint(0,100)
                if(r < in_community_probability*100):
                    adj_matrix[node1][node2] = 1
                    adj_matrix[node2][node1] = 1
                    no_of_edges += 1

#ASSIGN EDES BETWEEN COMMUNITY
for i in range(0, no_of_Communities):
    for j in range(i+1, no_of_Communities):
        community1 = community_nodes[i]
        community2 = community_nodes[j]
        for node1 in community1:
            for node2 in community2:
                r = random.randint(0,100)
                if(r < out_community_probability*100):
                    adj_matrix[node1][node2] = 1
                    adj_matrix[node2][node1] = 1
                    no_of_edges += 1

#CREATE A DEGREE MATRIX
D = torch.zeros([no_of_nodes, no_of_nodes], dtype = torch.int32)
for i in range(0, no_of_nodes):
    node_matrix = adj_matrix[i]
    degree = 0
    for j in range(0, no_of_nodes):
        if node_matrix[j] == 1:
            degree +=1
    D[i][i] = degree

D = D.numpy()
Droot = np.sqrt(D)
D_hat = np.linalg.inv(Droot)
D_hat = torch.tensor(D_hat)
print(D_hat)
A_hat = D_hat * adj_matrix * D_hat
print(A_hat)

B = torch.zeros([no_of_nodes, no_of_nodes], dtype = torch.int32)
for i in range(0, no_of_nodes):
    for j in range(0, no_of_nodes):
        B[i][j] = D[i][i]*D[j][j]/(2*no_of_edges)

S = torch.zeros([no_of_nodes, no_of_nodes], dtype = torch.int32)
for i in range(0, no_of_nodes):
    for j in range(0, no_of_nodes):
        S[i][j] = attribute_matrix[i]*attribute_matrix[j].t()/(torch.sum(attribute_matrix[i])*torch.sum(attribute_matrix[j]))

K = torch.zeros([no_of_nodes, no_of_nodes], dtype = torch.int32)
for i in range(0, no_of_nodes):
    for j in range(0, no_of_nodes):
        K[i][j] = B[i][j] + (beta*S[i][j]/torch.sum(S[i][j], dim = 0)[i])















