"""
Author: William Carlsson

How to run:
- Execute the script using `python task1.py reactions.csv`.
"""

import csv
import sys


def read_reactions(file_path):
    """
    Reads the reactions from a CSV file and extracts the start and end metabolites with the reaction details.
    
    Parameters:
        file_path: Path to the CSV file
    Returns:
        List of tuples containing (start_metabolite, end_metabolite, enzyme)
    """
    reactions = []
    try:
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                if len(row) == 3:
                    start_metabolite, end_metabolite, enzyme = row
                    reactions.append((start_metabolite, end_metabolite, enzyme))
                else:
                    print(f"Skipping malformed row: {row}")
        return reactions
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        sys.exit(1)


def get_start_and_end_metabolites(reactions):
    """
    Extracts the first and last metabolites from the reactions list.
    
    Parameters:
        reactions: List of reactions (start_metabolite, end_metabolite, enzyme)
    Returns:
        A tuple of (start_metabolite, end_metabolite)
    """
    if not reactions:
        print("Error: The reactions list is empty.")
        sys.exit(1)
    
    start_metabolite = reactions[0][0]
    end_metabolite = reactions[-1][1]
    
    return start_metabolite, end_metabolite



def build_graph(reactions):
    """
    Build a graph from the given reactions.

    Parameters:
        reactions (list): List of tuples (start_metabolite, end_metabolite, enzyme).

    Returns:
        dict: A dictionary where keys are metabolites and values are lists of (neighbor, enzyme).
    """
    graph = {}
    for start, end, enzyme in reactions:
        if start not in graph:
            graph[start] = []
        graph[start].append((end, enzyme))
    return graph


def has_path(graph, start, end, visited):
    """
    Check if there is a path from start to end in the graph.

    Parameters:
        graph (dict): Graph representation.
        start (str): Starting node.
        end (str): Target node.
        visited (set): Set of visited nodes.

    Returns:
        bool: True if a path exists, False otherwise.
    """
    if start == end:
        return True
    visited.add(start)
    for neighbor, _ in graph.get(start, []):
        if neighbor not in visited:
            if has_path(graph, neighbor, end, visited):
                return True
    return False


def identify_non_essential_enzymes(reactions, start, end):
    """
    Identifies non-essential enzymes in a metabolic pathway.

    Parameters:
        reactions (list): List of tuples (start_metabolite, end_metabolite, enzyme).
        start (str): Starting metabolite (in case of task 1 its 'glucose').
        end (str): Target metabolite (in case of task 1 its 'pyruvate').

    Returns:
        list: List of non-essential enzymes.
    """
    
    graph = build_graph(reactions)

    # Identify all enzymes in the pathway
    enzymes = {enzyme for _, _, enzyme in reactions}
    non_essential = []

    # Test each enzyme's essentiality
    for enzyme in enzymes:
        # Create a copy of the graph with the enzyme's reactions removed
        modified_graph = {
            node: [(neighbor, enz) for neighbor, enz in edges if enz != enzyme]
            for node, edges in graph.items()
        }

        # Check if a path still exists from start to end
        if has_path(modified_graph, start, end, set()):
            non_essential.append(enzyme)

    return non_essential


if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: python task1.py <reactions.csv>")
        sys.exit(1)

    file_path = sys.argv[1]

    reactions = read_reactions(file_path)
    start_metabolite, end_metabolite = get_start_and_end_metabolites(reactions)

    non_essential_enzymes = identify_non_essential_enzymes(reactions, start_metabolite, end_metabolite)

    print("Non-essential enzymes:", non_essential_enzymes)
