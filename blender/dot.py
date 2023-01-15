import os
import sys
import traceback

import numpy


def run_np_dot_standard():
    np_dir = os.path.dirname(__file__)
    try:

        result_file = os.path.join(np_dir, "dot.npy")
        print(result_file)
        a = numpy.load(os.path.join(np_dir, "a.npy"))
        b = numpy.load(os.path.join(np_dir, "b.npy"))
        dot = numpy.dot(a, b)
        numpy.save(result_file, dot)
    except:
        tb = open(os.path.join(np_dir, "tb.log")).write(traceback.format_exc())
        return -1

    return 0


if __name__ == "__main__":
    sys.exit(run_np_dot_standard())
