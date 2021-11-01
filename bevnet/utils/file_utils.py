import os
import yaml
import numpy as np

def parse_poses(filename):
    """ read poses file with per-scan poses from given filename
        Returns
        -------
        list
            list of poses as 4x4 numpy arrays.
    """
    file = open(filename)

    poses = []

    for line in file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(pose)

    return np.stack(poses)

def resolve_yaml_includes(content, base_dir):
    if content.startswith('includes:'):
        end_of_include_line = content.index('\n') + 1
        dt = yaml.load(content[:end_of_include_line], Loader=yaml.FullLoader)
        content = content[end_of_include_line:]

        inc_contents = []
        for include in dt['includes']:
            with open(os.path.join(base_dir, include), 'r') as fstream:
                inc_content = fstream.read()
            inc_content = resolve_yaml_includes(inc_content, base_dir)
            inc_contents.append(inc_content)
        
        inc_contents.append(content)

        return '\n'.join(inc_contents)
    else:
        return content

def load_yaml(fn, resolve_includes=True):

    with open(fn, 'r') as fstream:
        content = fstream.read()

    if resolve_includes:
        base_dir = os.path.dirname(fn)
        content = resolve_yaml_includes(content, base_dir)

    return yaml.load(content, Loader=yaml.FullLoader)
    
    
def listdir(dir, suffix=''):
    '''
        List all the files in a directory that their names ends with suffix
    '''
    return [os.path.join(dir, name)
            for name in sorted(os.listdir(dir))
            if name.endswith(suffix)]

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in sorted(filenames)
        if filename.endswith(suffix)
    ]

