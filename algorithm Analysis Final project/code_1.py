import networkx as nx
import numpy as np
import random
import math
import heapq

def read_input(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    adjacency_matrix = [list(map(int, line.strip().split(':'))) for line in lines]

    num_nodes = len(adjacency_matrix)
    bandwidth_matrix = np.random.uniform(3, 10, size=(num_nodes, num_nodes))
    delay_matrix = np.random.uniform(1, 5, size=(num_nodes, num_nodes))
    reliability_matrix = np.random.uniform(0.95, 0.99, size=(num_nodes, num_nodes))

    return np.array(adjacency_matrix), bandwidth_matrix, delay_matrix, reliability_matrix

def heuristic_function(node, my_graph):
    degree_heuristic = len(list(my_graph.neighbors(node)))
    random_heuristic = random.uniform(0, 1)
    
    return degree_heuristic + random_heuristic

def dijkstra_algorithm(my_graph, source, destination, bandwidth_constraint):
    try:
        path = nx.shortest_path(my_graph, source=source, target=destination, weight='bandwidth')
        return path
    except nx.NetworkXNoPath:
        print(f"No path found from node {source} to {destination}.")
        return []

def bellman_ford_algorithm(my_graph, source, destination, bandwidth_constraint):
    distances = {node: float('inf') for node in my_graph.nodes}
    distances[source] = 0

    for _ in range(len(my_graph.nodes) - 1):
        for edge in my_graph.edges:
            edge_bandwidth = my_graph[edge[0]][edge[1]]['bandwidth']
            if distances[edge[0]] + edge_bandwidth < distances[edge[1]]:
                distances[edge[1]] = distances[edge[0]] + edge_bandwidth

    path = nx.shortest_path(my_graph, source=source, target=destination, weight='bandwidth')

    return path

def a_star_algorithm(my_graph, source, destination, bandwidth_constraint):
    def heuristic(node):
        return heuristic_function(node, my_graph)

    distances = {node: float('inf') for node in my_graph.nodes}
    distances[source] = 0

    priority_queue = [(0 + heuristic(source), source)]

    while priority_queue:
        _, current_node = heapq.heappop(priority_queue)

        for neighbor in my_graph.neighbors(current_node):
            edge_bandwidth = my_graph[current_node][neighbor]['bandwidth']
            if distances[current_node] + edge_bandwidth < distances[neighbor]:
                distances[neighbor] = distances[current_node] + edge_bandwidth
                heapq.heappush(priority_queue, (distances[neighbor] + heuristic(neighbor), neighbor))

    path = nx.shortest_path(my_graph, source=source, target=destination, weight='bandwidth')

    return path

def floyd_warshall_algorithm(my_graph, source, destination, bandwidth_constraint):
    pass

def simulated_annealing_algorithm(my_graph, source, destination, bandwidth_constraint, delay_constraint, reliability_constraint):
    current_solution = dijkstra_algorithm(my_graph, source, destination, bandwidth_constraint)
    current_cost = calculate_cost(my_graph, current_solution)

    temperature = 100.0
    cooling_rate = 0.005

    while temperature > 1.0:
        neighbor_solution = get_neighbor_solution(my_graph, current_solution)
        neighbor_cost = calculate_cost(my_graph, neighbor_solution)

        delta_cost = neighbor_cost - current_cost

        print(f"Temperature: {temperature}, Current Cost: {current_cost}, Neighbor Cost: {neighbor_cost}, Delta Cost: {delta_cost}")

        print(f"Probability: {math.exp(-delta_cost / temperature)}")

        if delta_cost < 0 or random.uniform(0, 1) < math.exp(-delta_cost / temperature):
            current_solution = neighbor_solution
            current_cost = neighbor_cost
            print("Accepted!")

        temperature *= 1 - cooling_rate

    return current_solution



def tabu_search_algorithm(my_graph, source, destination, bandwidth_constraint, delay_constraint, reliability_constraint):
    tabu_list = []
    current_solution = dijkstra_algorithm(my_graph, source, destination, bandwidth_constraint)
    current_cost = calculate_cost(my_graph, current_solution)

    iterations = 100
    tabu_size = 10

    for _ in range(iterations):
        neighbor_solution = get_neighbor_solution(my_graph, current_solution)

        while neighbor_solution in tabu_list:
            neighbor_solution = get_neighbor_solution(my_graph, current_solution)

        neighbor_cost = calculate_cost(my_graph, neighbor_solution)

        if neighbor_cost < current_cost:
            current_solution = neighbor_solution
            current_cost = neighbor_cost

        tabu_list.append(neighbor_solution)

        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

    return current_solution

def ant_colony_algorithm(my_graph, source, destination, bandwidth_constraint, delay_constraint, reliability_constraint):
    num_ants = 10
    pheromone_decay = 0.1
    pheromone_weight = 1.0
    visibility_weight = 2.0

    pheromone_matrix = np.ones_like(my_graph['weight_matrix']) * 0.1

    for iteration in range(100):
        ant_solutions = []

        for ant in range(num_ants):
            ant_solution = construct_solution(my_graph, pheromone_matrix, visibility_weight, source, destination)
            ant_solutions.append((ant_solution, calculate_cost(my_graph, ant_solution)))

        update_pheromones(pheromone_matrix, ant_solutions, pheromone_decay, pheromone_weight)

    best_solution = min(ant_solutions, key=lambda x: x[1])[0]
    return best_solution

def construct_solution(my_graph, pheromone_matrix, visibility_weight, source, destination):
    current_node = source
    solution = [current_node]

    while current_node != destination:
        possible_moves = [(neighbor, pheromone_matrix[current_node][neighbor] ** visibility_weight * (1 / my_graph[current_node][neighbor]['bandwidth'])) for neighbor in my_graph.neighbors(current_node)]

        if not possible_moves:
            break

        next_node = max(possible_moves, key=lambda x: x[1])[0]
        solution.append(next_node)
        current_node = next_node

    return solution


def update_pheromones(pheromone_matrix, ant_solutions, pheromone_decay, pheromone_weight):
    pheromone_matrix *= (1.0 - pheromone_decay)

    for solution, cost in ant_solutions:
        for i in range(len(solution) - 1):
            pheromone_matrix[solution[i]][solution[i + 1]] += pheromone_weight / cost

def bee_colony_algorithm(my_graph, source, destination, bandwidth_constraint, delay_constraint, reliability_constraint):
    num_scout_bees = 5
    num_best_bees = 3
    num_best_sites = 5
    num_selected_sites = 2
    num_elite_sites = 1
    max_epochs = 100

    best_solution = dijkstra_algorithm(my_graph, source, destination, bandwidth_constraint)
    best_cost = calculate_cost(my_graph, best_solution)

    for epoch in range(max_epochs):
        scout_bees = [get_neighbor_solution(my_graph, best_solution) for _ in range(num_scout_bees)]
        best_bees = sorted(scout_bees, key=lambda x: calculate_cost(my_graph, x))[:num_best_bees]

        for best_bee_solution in best_bees:
            neighbor_solution = get_neighbor_solution(my_graph, best_bee_solution)
            neighbor_cost = calculate_cost(my_graph, neighbor_solution)

            if neighbor_cost < best_cost:
                best_solution = neighbor_solution
                best_cost = neighbor_cost

    return best_solution

def firefly_algorithm(my_graph, source, destination, bandwidth_constraint, delay_constraint, reliability_constraint):
    num_fireflies = 10
    max_iterations = 100
    attractiveness_base = 1.0
    beta = 1.0

    fireflies = [get_neighbor_solution(my_graph, dijkstra_algorithm(my_graph, source, destination, bandwidth_constraint))
                 for _ in range(num_fireflies)]

    for iteration in range(max_iterations):
        for i in range(num_fireflies):
            for j in range(num_fireflies):
                if calculate_cost(my_graph, fireflies[i]) > calculate_cost(my_graph, fireflies[j]):
                    attractiveness = attractiveness_base * math.exp(-beta * calculate_cost(my_graph, fireflies[i]))
                    move_firefly(my_graph, fireflies, i, j, attractiveness)

    best_solution = min(fireflies, key=lambda x: calculate_cost(my_graph, x))
    return best_solution

def move_firefly(my_graph, fireflies, i, j, attractiveness):
    alpha = 0.5
    beta = 1.0
    gamma = 1.0

    if calculate_cost(my_graph, fireflies[i]) > calculate_cost(my_graph, fireflies[j]):
        distance = my_graph['weight_matrix'][fireflies[i][0]][fireflies[j][0]]

        attractiveness *= math.exp(-beta * distance ** gamma)

        for k in range(len(fireflies[i])):
            if random.uniform(0, 1) < attractiveness:
                fireflies[i][k] = fireflies[j][k]

def calculate_cost(my_graph, path):
    total_cost = 0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_data = my_graph.get_edge_data(u, v)
        
        if edge_data is not None:
            total_cost += edge_data.get('bandwidth', 0)

    return total_cost

def get_neighbor_solution(my_graph, current_solution):
    neighbor_solution = current_solution.copy()

    node_to_modify = random.choice(range(len(neighbor_solution)))
    neighbors = list(my_graph.neighbors(neighbor_solution[node_to_modify]))

    if neighbors:
        new_neighbor = random.choice(neighbors)
        neighbor_solution[node_to_modify] = new_neighbor

    return neighbor_solution

def main():
    adjacency, bandwidth, delay, reliability = read_input("input.txt")

    my_graph = nx.Graph()
    my_graph.add_nodes_from(range(len(adjacency)))

    for i in range(len(adjacency)):
        for j in range(len(adjacency[i])):
            if adjacency[i][j] != 0:
                my_graph.add_edge(i, j, bandwidth=adjacency[i][j])

    nx.set_edge_attributes(my_graph, dict(zip(my_graph.edges, bandwidth.flatten())), 'bandwidth')

    source_node = 0
    destination_node = 5
    bandwidth_requirement = 5
    delay_threshold = 40
    reliability_threshold = 0.70 

    if source_node not in my_graph.nodes:
        print(f"Error: Source node {source_node} is not present in the graph.")
        return

    # Example using Simulated Annealing algorithm
    path_simulated_annealing = simulated_annealing_algorithm(my_graph, source_node, destination_node,bandwidth_requirement, delay_threshold, reliability_threshold)
    print("Shortest Path (Simulated Annealing):", path_simulated_annealing)

    # Example using Tabu Search algorithm
    path_tabu_search = tabu_search_algorithm(my_graph, source_node, destination_node,bandwidth_requirement, delay_threshold, reliability_threshold)

    print("Shortest Path (Tabu Search):", path_tabu_search)

    # Example using Ant Colony algorithm
    path_ant_colony = ant_colony_algorithm(my_graph, source_node, destination_node,bandwidth_requirement, delay_threshold, reliability_threshold)

    print("Shortest Path (Ant Colony):", path_ant_colony)

    # Example using Bee Colony algorithm
    path_bee_colony = bee_colony_algorithm(my_graph, source_node, destination_node,bandwidth_requirement, delay_threshold, reliability_threshold)

    print("Shortest Path (Bee Colony):", path_bee_colony)

    # Example using Firefly algorithm
    path_firefly = firefly_algorithm(my_graph, source_node, destination_node,bandwidth_requirement, delay_threshold, reliability_threshold)
    print("Shortest Path (Firefly):", path_firefly)


if __name__ == "__main__":
    main()
