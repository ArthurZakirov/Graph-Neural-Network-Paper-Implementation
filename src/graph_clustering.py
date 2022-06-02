import numpy as np
import numpy.random as nr
from numpy.linalg import inv

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def k_eigs(A, K=None, smallest=True, coverage=0.9):
    eigs = list()
    V = list()

    if smallest:
        eig_max, _ = power_iteration(A)
        I = np.eye(len(A))
        A = A - eig_max * I
        eig_sum = np.abs(np.trace(A))

    if K == None:
        k_eig_sum = 0
        while k_eig_sum < eig_sum * coverage:
            eig, v = power_iteration(A)
            eigs.append(eig)
            V.append(v)
            A = A - eig * v @ v.T
            k_eig_sum = np.abs(np.array(eigs).sum())
    else:
        for k in range(K):
            eig, v = power_iteration(A)
            eigs.append(eig)
            V.append(v)
            A = A - eig * v @ v.T

    eigs = np.array(eigs)
    if smallest:
        eigs += eig_max
    V = np.concatenate(V, axis=1)
    return eigs, V


def power_iteration(A, num_iter=25):
    # get eigenvector
    v = nr.random(size=(len(A), 1))
    v = v / v.sum()
    for i in range(num_iter):
        v = A @ v
        v = v / np.linalg.norm(v)

    # get eigenvalue
    eig = (v.T @ A @ v).squeeze()
    return eig, v


def cluster_graph(G, K, visualize=True):
    L = G.normalized_laplacian_matrix()
    eig, V = k_eigs(L, K=K)

    k_means = KMeans(n_clusters=K)
    k_means.fit(V)
    cluster = k_means.predict(V)
    print(f"Number of clusters:", K)
    if visualize:
        t_sne = TSNE(n_components=2)
        t_sne.fit(V)
        V_emb = t_sne.embedding_

        for k in range(K):
            x = V_emb[cluster == k, 0]
            y = V_emb[cluster == k, 1]
            plt.scatter(x, y)

        for edge in G.edges.keys():
            pos_1 = V_emb[edge[0] - 1]
            pos_2 = V_emb[edge[1] - 1]
            plt.plot(
                [pos_1[0], pos_2[0]],
                [pos_1[1], pos_2[1]],
                color="k",
                linewidth=0.1,
            )
    plt.axis("off")
    plt.savefig("../img/visualize_graph_clustered.png")

    return cluster


def remove_random_labels_from_graph(G):
    """
    Arguments
    ---------
    G : Graph
    
    Returns
    -------
    labeled : np.array [N]
        randomly created mask for which nodes in graph are labeled
    """
    labeled = np.random.randint(2, size=(len(G.nodes))) == 1
    return labeled


def label_propagation(G, labeled):
    """
    Arguments
    ---------
    G : Graph
    labeled : np.array [N]
        randomly created mask for which nodes in graph are labeled
    
    Returns
    -------
    y_U : np.array [N_unseen]
        category prediction for unlabeled nodes
    """
    y = cluster_graph(G, K=2)
    L = G.laplacian_matrix()

    L_SS = L[labeled][:, labeled]
    L_SU = L[labeled][:, ~labeled]
    L_UU = L[~labeled][:, ~labeled]
    L_US = L[~labeled][:, labeled]
    y_S = y[labeled]

    # predict
    y_U = -inv(L_UU) @ L_US @ y_S

    # control
    # y_U == y[~labeled]
    return y_U
