import heapq

# Class representing a network request
class Request:
    def __init__(self, source, destination, bandwidth):
        self.source = source
        self.destination = destination
        self.bandwidth = bandwidth

def dijkstra(adjacency_matrix, weights, start, end):
    n = len(adjacency_matrix)
    min_dist = [float('inf')] * n
    min_dist[start] = 0
    heap = [(0, start, [])]  

    while heap:
        current_dist, current_node, current_path = heapq.heappop(heap)

        if current_node == end:
            return current_path + [end]

        if current_node >= len(weights):
            continue

        for neighbor, weight in enumerate(weights[current_node]):
            if weight > 0 and min_dist[current_node] + weight < min_dist[neighbor]:
                min_dist[neighbor] = min_dist[current_node] + weight
                heapq.heappush(heap, (min_dist[neighbor], neighbor, current_path + [current_node]))

    return []

def is_feasible(path, bandwidth_matrix, delay_matrix, reliability_matrix, request):
    for i in range(len(path) - 1):
        source = path[i]
        destination = path[i + 1]

        if bandwidth_matrix[source][destination] < request.bandwidth:
            return False

        if delay_matrix[source][destination] > 0:  # Assuming delay should be greater than 0 for feasibility
            return False

        if reliability_matrix[source][destination] < 1.0:  # Assuming reliability should be 1.0 for feasibility
            return False

    return True

def find_optimal_path(adjacency_matrix, bandwidth_matrix, delay_matrix, reliability_matrix, request):
    start = request.source
    end = request.destination

    shortest_path = dijkstra(adjacency_matrix, bandwidth_matrix, start, end)

    for path in shortest_path:
        if is_feasible(path, bandwidth_matrix, delay_matrix, reliability_matrix, request):
            return path

    return []

def handle_requests(adjacency_matrix, bandwidth_matrix, delay_matrix, reliability_matrix, requests):
    results = []

    for request in requests:
        path = find_optimal_path(adjacency_matrix, adjacency_matrix, delay_matrix, reliability_matrix, request)
        results.append((request, path))

    return results

def write_output(results, output_file):
    with open(output_file, 'w') as file:
        for request, path in results:
            if path:
                file.write(f"Request: {request.source} -> {request.destination}\n")
                file.write(f"Path: {' -> '.join(map(str, path))}\n\n")
            else:
                file.write(f"No feasible path found for request: {request.source} -> {request.destination}\n\n")

def read_input_file(file_path):
    adjacency_matrix = []
    bandwidth_matrix = []
    delay_matrix = []
    reliability_matrix = []

    with open(file_path, 'r') as file:
        lines = file.read().splitlines()

    section = 1
    for line in lines:
        if line:
            if line.startswith("0:") or line.startswith("1:"):
                row = [int(float(x)) if x.strip() else 0 for x in line.split(":")[1].split(":")]
                adjacency_matrix.append(row)
            elif line.startswith("0.0:"):
                row = [float(x) if x.strip() else 0.0 for x in line.split(":")[1].split(":")]
                reliability_matrix.append(row)
            else:
                row = [float(x) if x.strip() else 0 for x in line.split(":")]
                if section == 2:
                    bandwidth_matrix.append(row)
                elif section == 3:
                    delay_matrix.append(row)

        else:
            section += 1

    return adjacency_matrix, bandwidth_matrix, delay_matrix, reliability_matrix

def main():
    input_file = 'input.txt'
    output_file = 'output.txt'

    # Reading matrices from the input file
    adjacency_matrix, bandwidth_matrix, delay_matrix, reliability_matrix = read_input_file(input_file)

    requests = [
        Request(0, 10, 5),
        Request(3, 7, 8),
        Request(0, 15, 5),
    ]

    results = handle_requests(adjacency_matrix, bandwidth_matrix, delay_matrix, reliability_matrix, requests)

    write_output(results, output_file)

if __name__ == "__main__":
    main()
