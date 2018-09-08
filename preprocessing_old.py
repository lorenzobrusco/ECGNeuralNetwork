import numpy as np
import scipy.io as sio

"""The range is ± 16.384 mV"""
range_value = 16.384


def create_simple_file(format):
    """
        Create a simple txt file to allow us to see
        the form of patient001/s0010_re
        :param format: is the file format
        :return: no return
    """
    example_patient001 = np.fromfile('original_dataset/patient001/s0010_re.%s' % format, dtype=float)
    output = open('dataset_example_txt/patient001_%s.txt' % format, 'w')
    for i in range(len(example_patient001)):
        output.write("".join(str(example_patient001[i])) + "\n")
    output.close()


if __name__:
    # create_simple_file('dat')
    # create_simple_file('xyz')
    pass
