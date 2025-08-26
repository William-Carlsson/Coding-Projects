"""
Author: William Carlsson

This program identifies the sequence of alpha-carbon atoms in a molecular chain 
based on their 3D coordinates. The solution is implemented as follows:

1. Reading Coordinates:
   - The `read_coordinates` function reads atomic coordinates from a specified input file.
     Each line in the file contains an atom's serial number and its 3D coordinates (x, y, z).

2. Calculating Distances:
   - The `calculate_distance` function computes the Euclidean distance between two atoms 
     using their 3D coordinates.

3. Building Adjacency Based on Distance:
   - The `find_alpha_chain` function identifies neighboring atoms in the chain 
     based on a fixed alpha-carbon distance of approximately 3.8 Å (with a tolerance).
   - It constructs an adjacency list where each atom is linked to its neighbors.

4. Sorting Neighbors:
   - To ensure consistent traversal, the neighbors of each atom in the adjacency list 
     are sorted by increasing distance.

5. Depth-First Search (DFS) Traversal:
   - Starting from a specific atom (atom 8, at index 7), the program performs a DFS 
     to traverse the chain. This ensures that all atoms in the chain are visited in sequence.

6. Chain Construction:
   - The serial numbers of the atoms are recorded in the order they are visited during the DFS.

7. Execution and Output:
   - The `main` function specifies the input file, runs the algorithm, and prints the 
     total number of alpha-carbon atoms in the chain and their sequence.

How to run:
- Execute the script using `python task1.py`.
- Modify the `input_file` variable in the `main` function to specify the input file.
"""

import math

def read_coordinates(filename):
    """Reads coordinates from a file and returns a list of (serial, x, y, z)."""
    atoms = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            serial = int(parts[0])
            x, y, z = map(float, parts[1:])
            atoms.append((serial, x, y, z))
    return atoms

def calculate_distance(atom1, atom2):
    """Calculates the Euclidean distance between two atoms."""
    _, x1, y1, z1 = atom1
    _, x2, y2, z2 = atom2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def find_alpha_chain(atoms, tolerance=0.1):
    """Identifies the order of alpha-carbon atoms in the chain."""
    alpha_distance = 3.8
    n = len(atoms)
    adjacency = {i: [] for i in range(n)}
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = calculate_distance(atoms[i], atoms[j])
            if abs(dist - alpha_distance) <= tolerance:
                adjacency[i].append(j)
                adjacency[j].append(i)

    # Sort adjacency list by distance for consistent traversal
    # Visit neighbors in increasing order of their distance from the current atom
    for key in adjacency:
        adjacency[key].sort(key=lambda x: calculate_distance(atoms[key], atoms[x]))
    
    # Index of the starting atom (atom 8 in this case)
    start_idx = 7
    
    chain = []
    visited = set()
    
    def dfs(atom_idx, chain):
        visited.add(atom_idx)
        chain.append(atoms[atom_idx][0]) 
        for neighbor in adjacency[atom_idx]:
            if neighbor not in visited:
                dfs(neighbor, chain)
    
    dfs(start_idx, chain)
    return chain


def main():
    input_file = 'test_q1.txt'  # Change this to 'data_q1.txt' if testing with other data
    atoms = read_coordinates(input_file)
    chain = find_alpha_chain(atoms)
    print("Total number of alpha-carbon atoms in the chain:", len(chain))
    print("Alpha-carbon chain order:")
    for a in chain:
        print(a)

if __name__ == "__main__":
    main()
