import math
import random
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, groupby
import time
from tqdm import tqdm


def draw_plot(dots_num, time_taken_krask, time_taken_prima):
    plt.plot(dots_num, time_taken_krask, label='Kraskall', color='green', linewidth=3)
    plt.plot(dots_num, time_taken_prima, label='Prima', color='purple', linewidth=3)
    plt.xlabel('number of dots')
    plt.ylabel('time taken')
    plt.title('Kruskal and Prima algorithms compairing')
    plt.legend()
    plt.show()



def gnp_random_connected_graph(num_of_nodes: int,
                               completeness: float,
                               draw: bool = False):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is conneted
    """

    edges = combinations(range(num_of_nodes), 2)
    G = nx.Graph()
    G.add_nodes_from(range(num_of_nodes))
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < completeness:
                G.add_edge(*e)
    for (u, v, w) in G.edges(data=True):
        w['weight'] = random.randint(0, 100)
    if draw:
        plt.figure(figsize=(10, 6))
        nx.draw(G, node_color='lightblue',
                with_labels=True,
                node_size=500)
    return G


def krask_edge_check(nodes_set, edge):
    first_set, second_set = -1, -1
    for i in range(len(nodes_set)):
        if edge[0] in nodes_set[i]:
            first_set = i
    if first_set == -1:
        return False
    for i in range(len(nodes_set)):
        if (edge[1] in nodes_set[i]) and (i != first_set):
            second_set = i
    if second_set == -1:
        return False
    return first_set, second_set


def krask_main(edges, nodes):
    edges.sort(key=lambda x: x[2])
    nodes_set = []
    for i in range(len(nodes)):
        nodes_set.append({i})
    karkas = []
    for edge in edges:
        position = krask_edge_check(nodes_set, edge[:-1])
        # print(edge, position, nodes_set)
        if position:
            nodes_set[position[0]] = nodes_set[position[0]].union(nodes_set[position[1]])
            del nodes_set[position[1]]
            karkas.append(edge)
            if len(nodes_set) == 1:
                weight = 0
                for _ in range(len(karkas)):
                    weight += karkas[_][2]
                return weight


def el_primo_get_min(used_nodes, edges):
    min_edge = (-1, -1, math.inf)
    for edge in edges:
        if ((edge[0] in used_nodes and edge[1] not in used_nodes)
                or (edge[1] in used_nodes and edge[0] not in used_nodes)):
            return edge


def el_primo_body(nodes, edges):
    carkas = []
    used_nodes = {0}
    edges.sort(key=lambda x: x[2])
    while len(used_nodes) != len(nodes):
        edge = el_primo_get_min(used_nodes, edges)
        if edge[2] == math.inf:
            break
        used_nodes.add(edge[0])
        used_nodes.add(edge[1])
        carkas.append(edge)
    weight = 0
    for _ in range(len(carkas)):
        weight += carkas[_][2]
    return weight


if __name__ == '__main__':
    dots = 20
    density = 0.1
    NUM_OF_ITERATIONS = 66
    time_taken_krask = 0
    time_taken_prima = 0
    # kraskall
    krask_results = []
    prima_results = []
    krask_time = []
    prima_time = []
    dots_num = []
    for i in tqdm(range(NUM_OF_ITERATIONS)):
        dots = math.floor(dots * 1.1)
        dots_num.append(dots)
        G = gnp_random_connected_graph(dots, density)
        edges = list(map(lambda x: (x[0], x[1], x[2]['weight']), G.edges.data()))
        nodes = list(G.nodes)
        #kraskall
        start = time.time()
        krask_results.append(krask_main(edges, nodes))
        end = time.time()
        krask_time.append(end - start)
        time_taken_krask += end - start
        # prima
        start = time.time()
        prima_results.append(el_primo_body(nodes, edges))
        end = time.time()
        prima_time.append(end - start)
        time_taken_prima += end - start


    #print('AVG Kraskall algoritms time is', time_taken_krask / NUM_OF_ITERATIONS)
    print('Kraskall results are:', krask_results)
    #print('AVG Prima algoritms time is', time_taken_prima / NUM_OF_ITERATIONS)
    print('Prima results are:', prima_results)
    print('Nums of iterations are:', dots_num)
    print('Time of Cruskall:', krask_time)
    print('Time of Prima:', prima_time)
    draw_plot(dots_num, krask_time, prima_time)
