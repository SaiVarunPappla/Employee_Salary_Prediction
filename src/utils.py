import dill

def load_model(path):
    with open(path, "rb") as f:
        return dill.load(f)
