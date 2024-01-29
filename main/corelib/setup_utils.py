import os


def get_project_name():
    """
    :return: the name of the project. This is done by finding the directories in './src/'
    """
    possible_names = []
    for d in os.listdir('./src/'):
        if '.' not in d:
            possible_names.append(d)
    assert len(possible_names) == 1, 'Only one name allowed. Found possible names={}'.format(possible_names)
    return possible_names[0]
