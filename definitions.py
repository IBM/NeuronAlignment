import os


def get_project_root():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    return dname
