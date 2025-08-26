import math

def parse_atoms(file_path):
    """Reads atom data from the file."""
    atoms = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()
            serial = int(parts[0])
            x, y, z = map(float, parts[1:])
            atoms.append((serial, x, y, z))
    return atoms

def calculate_distance(atom1, atom2):
    """Calculates the Euclidean distance between two atoms."""
    _, x1, y1, z1 = atom1
    _, x2, y2, z2 = atom2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def build_graph(atoms, alpha_distance=3.8, tolerance=0.1):
    """Constructs a graph of candidate alpha-carbon atoms."""
    n = len(atoms)
    adjacency = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            dist = calculate_distance(atoms[i], atoms[j])
            if abs(dist - alpha_distance) <= tolerance:
                adjacency[i].append(j)
                adjacency[j].append(i)
    return adjacency


def find_longest_chain(adjacency):
    """Finds the longest chain of connected atoms."""
    visited = set()
    longest_chain = []

    def dfs(node, path):
        nonlocal longest_chain
        visited.add(node)
        path.append(node)

        # Update longest chain if this path is longer
        if len(path) > len(longest_chain):
            longest_chain = path[:]

        # Explore neighbors
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                dfs(neighbor, path)

        # Backtrack
        path.pop()
        visited.remove(node)

    # Try starting DFS from each node
    for node in adjacency:
        dfs(node, [])

    return longest_chain

def calculate_angle(a, b, c):
    """Calculates the angle (in degrees) between three atoms."""
    import math
    vec1 = (b[1] - a[1], b[2] - a[2], b[3] - a[3])
    vec2 = (c[1] - b[1], c[2] - b[2], c[3] - b[3])
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    mag1 = math.sqrt(sum(v**2 for v in vec1))
    mag2 = math.sqrt(sum(v**2 for v in vec2))
    cos_theta = dot_product / (mag1 * mag2)
    return math.degrees(math.acos(cos_theta))

def filter_by_angles(chain, atoms):
    """Filters chains based on typical alpha-carbon angles."""
    refined_chain = []
    for i in range(1, len(chain) - 1):
        angle = calculate_angle(atoms[chain[i - 1]], atoms[chain[i]], atoms[chain[i + 1]])
        if 110 <= angle <= 120:
            refined_chain.append(chain[i])
    return refined_chain

if __name__ == "__main__":
    atoms = parse_atoms("data_q2.txt")
    adjacency = build_graph(atoms)

    # Find the longest chain of candidate alpha-carbons
    longest_chain = find_longest_chain(adjacency)

    # Refine the chain using angle criteria (optional)
    refined_chain = filter_by_angles(longest_chain, atoms)

    # Print the results
    print("Longest Chain Length:", len(longest_chain))
    print("Longest Chain (Refined):", refined_chain)
