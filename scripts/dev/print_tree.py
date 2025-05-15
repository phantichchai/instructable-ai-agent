import os


def print_tree(root, prefix="", level=2):
        if level < 0:
            return
        files = sorted(os.listdir(root))
        for f in files:
            if f in ['.git', 'dataset', "__pycache__"]:
                continue
            
            path = os.path.join(root, f)
            print(prefix + "├── " + f)
            if os.path.isdir(path):
                print_tree(path, prefix + "│   ", level - 1)

if __name__ == "__main__":
    print_tree(".", level=3)