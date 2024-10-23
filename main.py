from collections import deque
import heapq

def main():
    # Depth-First Search (DFS)
    def dfs(graph, start, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)
        print(start, end=' ')
        for neighbor in graph[start]:
            if neighbor not in visited:
                dfs(graph, neighbor, visited)

    # Breadth-First Search (BFS)
    def bfs(graph, start):
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            vertex = queue.popleft()
            print(vertex, end=' ')
            
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    # Dijkstra's Algorithm
    def dijkstra(graph, start):
        pq = [(0, start)]  # (distance, node)
        distances = {node: float('inf') for node in graph}
        distances[start] = 0

        while pq:
            current_distance, current_node = heapq.heappop(pq)
            
            if current_distance > distances[current_node]:
                continue

            for neighbor, weight in graph[current_node]:
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))

        return distances

    # Binary Search
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    # Merge Sort
    def merge_sort(arr):
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])

        return merge(left, right)

    def merge(left, right):
        sorted_arr = []
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                sorted_arr.append(left[i])
                i += 1
            else:
                sorted_arr.append(right[j])
                j += 1

        sorted_arr.extend(left[i:])
        sorted_arr.extend(right[j:])

        return sorted_arr

    # Quick Sort
    def quick_sort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)

    # Topological Sort (using DFS)
    def topological_sort(graph):
        visited = set()
        stack = []

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in graph[node]:
                dfs(neighbor)
            stack.append(node)

        for node in graph:
            if node not in visited:
                dfs(node)

        return stack[::-1]

    # Union-Find (Disjoint Set)
    class UnionFind:
        def __init__(self, size):
            self.root = [i for i in range(size)]
            self.rank = [1] * size

        def find(self, x):
            if x == self.root[x]:
                return x
            self.root[x] = self.find(self.root[x])
            return self.root[x]

        def union(self, x, y):
            rootX = self.find(x)
            rootY = self.find(y)

            if rootX != rootY:
                if self.rank[rootX] > self.rank[rootY]:
                    self.root[rootY] = rootX
                elif self.rank[rootX] < self.rank[rootY]:
                    self.root[rootX] = rootY
                else:
                    self.root[rootY] = rootX
                    self.rank[rootX] += 1

        def connected(self, x, y):
            return self.find(x) == self.find(y)

    # Knapsack Problem (Dynamic Programming)
    def knapsack(weights, values, W):
        n = len(values)
        dp = [[0] * (W + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            for w in range(W + 1):
                if weights[i-1] <= w:
                    dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
                else:
                    dp[i][w] = dp[i-1][w]

        return dp[n][W]

    # Example Usage
    if __name__ == "__main__":
        graph = {
            0: [1, 2],
            1: [2],
            2: [0, 3],
            3: [3]
        }

        print("DFS:")
        dfs(graph, 2)
        print("\nBFS:")
        bfs(graph, 2)

        print("\nDijkstra's Algorithm:")
        weighted_graph = {
            'A': [('B', 1), ('C', 4)],
            'B': [('A', 1), ('C', 2), ('D', 5)],
            'C': [('A', 4), ('B', 2), ('D', 1)],
            'D': [('B', 5), ('C', 1)]
        }
        print(dijkstra(weighted_graph, 'A'))

        arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        print("\nBinary Search:", binary_search(arr, 7))

        arr_to_sort = [3, 6, 8, 10, 1, 2, 1]
        print("\nMerge Sort:", merge_sort(arr_to_sort))
        print("\nQuick Sort:", quick_sort(arr_to_sort))

        dag = {
            0: [1, 2],
            1: [3],
            2: [3],
            3: []
        }
        print("\nTopological Sort:", topological_sort(dag))

        uf = UnionFind(10)
        uf.union(1, 2)
        uf.union(3, 4)
        print("\nUnion Find - Connected 1 and 2:", uf.connected(1, 2))
        print("Union Find - Connected 1 and 3:", uf.connected(1, 3))

        values = [60, 100, 120]
        weights = [10, 20, 30]
        W = 50
        print("\nKnapsack Problem:", knapsack(weights, values, W))


if __name__ == "__main__":
    main()
    # main([1, 2, 3, 4, 6])
