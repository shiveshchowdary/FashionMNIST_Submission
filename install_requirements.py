import subprocess

def install_dependencies():
    dependencies = ["tensorflow", "matplotlib", "seaborn", "numpy", "pandas", "scikit-learn"]

    for package in dependencies:
        subprocess.run(["pip", "install", package])

if __name__ == "__main__":
    install_dependencies()
