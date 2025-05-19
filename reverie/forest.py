import re

# Define a simple tree node.
class TreeNode:
    def __init__(self, name):
        self.name = name
        # Use a list of children to preserve insertion order.
        self.children = []
        
    # Utility method to find a child by name.
    def get_child(self, child_name):
        for child in self.children:
            if child.name == child_name:
                return child
        return None

# Helper to add (or retrieve) a child from a parent node.
def add_child(parent, child_name):
    child = parent.get_child(child_name)
    if child is None:
        child = TreeNode(child_name)
        parent.children.append(child)
    return child

# Rebuild the forest from the list of leaf node strings.
def rebuild_forest(leaf_paths):
    forest = {}  # key: root name, value: TreeNode
    for path in leaf_paths:
        path = path.strip()
        tokens = []
        if '(' in path:
            # The part before the first '(' is the root.
            idx = path.index('(')
            root_token = path[:idx].strip()
            tokens.append(root_token)
            # Use regex to capture text within parentheses.
            inside_tokens = re.findall(r'\(([^)]+)\)', path)
            tokens.extend(token.strip() for token in inside_tokens)
        else:
            tokens = [path]
        
        # Build (or update) the tree path.
        root_name = tokens[0]
        if root_name not in forest:
            forest[root_name] = TreeNode(root_name)
        current = forest[root_name]
        for token in tokens[1:]:
            current = add_child(current, token)
            
    return forest

# Recursively format a tree node into a list of strings with numbering.
# - 'numbering' is a list of integers that track the hierarchy position.
# - 'level' is used for indentation (each level indented 2 spaces).
def format_tree(node, numbering=[], level=0):
    lines = []
    indent = "  " * level
    # The root node (level 0) is printed without numbering.
    if level == 0:
        # If the node has children, add a colon.
        line = f"{indent}{node.name}:"
        lines.append(line)
    else:
        # For non-root levels, create the numbering prefix.
        prefix_str = "-".join(str(num) for num in numbering)
        # Append a colon if the node has children.
        if node.children:
            line = f"{indent}{prefix_str}) {node.name}:"
        else:
            line = f"{indent}{prefix_str}) {node.name}"
        lines.append(line)
    
    # For each child, determine its numbering by appending its order.
    for i, child in enumerate(node.children, start=1):
        child_numbering = numbering + [i]
        lines.extend(format_tree(child, child_numbering, level + 1))
    return lines

# Format the entire forest into a list of tree strings.
def format_forest(forest):
    trees = []
    # Iterate in the order the roots were added.
    for root_name, root_node in forest.items():
        tree_lines = format_tree(root_node)
        tree_str = "\n".join(tree_lines)
        trees.append(tree_str)
    return trees

# Example usage:
if __name__ == "__main__":
    leaf_strings = [
        "Animal (Mammal) (Dog) (Beagle)",
        "Animal (Mammal) (Dog) (Bulldog)",
        "Animal (Mammal) (Cat) (Siamese)",
        "Plant (Tree) (Oak)"
    ]
    
    # Rebuild the forest from the list of leaf strings.
    forest = rebuild_forest(leaf_strings)
    # Format the forest as a list of multi-line strings.
    forest_strings = format_forest(forest)

    print(forest_strings)