import os


def create_project_structure():
    folders = ['data', 'model', 'src', 'notebooks']

    # Create subfolders
    for folder in folders:
        folder_path = os.path.join('.', folder)
        os.makedirs(folder_path, exist_ok=True)

    # Create other essential files
    open(os.path.join('.', 'requirements.txt'), 'a').close()
    open(os.path.join('.', 'README.md'), 'a').close()
    open(os.path.join('.', 'Dockerfile'), 'a').close


if __name__ == "__main__":
    project_root = 'instadeep-test'
    create_project_structure()
