import os


def print_tree(root, ignore_dir: list[str], prefix="", level=2):
        if level < 0:
            return
        files = sorted(os.listdir(root))
        for f in files:
            if f in ignore_dir:
                continue
            
            path = os.path.join(root, f)
            print(prefix + "├── " + f)
            if os.path.isdir(path):
                print_tree(path, ignore_dir, prefix + "│   ", level - 1)

if __name__ == "__main__":
    print_tree(
        root=".",
        ignore_dir=['.git', "__pycache__"],
        prefix="",
        level=1)