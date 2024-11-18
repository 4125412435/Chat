import os

def load_text(fname):
    with open(os.path.join('resources', 'text', fname), 'r') as f:
        return f.read()

def get_path(*args):
    return os.path.join('resources', *args)
