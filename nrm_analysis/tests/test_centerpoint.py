import unittest, os, glob
import sys
import numpy as np

"""
In order for pytest to run this unit test code you must be using a 'development'
version installed from the directory where setup.py exists.
Here that directory is  >>/whatever<</nrm_analysis
    cd >>directory_containing_setup.py<<
    pip install -e .  (yes, there is a dot at the end) 
so that, for instance, "from ..misctools import utils" imports OK.

Also note that the next directory down,  >>/whatever<</nrm_analysis/nrm_analysis,
contains an __init__.py file that reads
    from .version import *
    __all__ = ['fringefitting','misctools','modeling','tests']

It appears that the tests run with or without the 'from .version import *' line
The test is not to be run with python but with pytest:
    (astroconda) user: pytest test_centerpoint.py
"""
from ..misctools import utils


"""
    Write  a test to check centerpoint() in utils.py.
    anand@stsci.edu 2018.04.05
"""
def distance(x,y):
    """ returns float Euclidean distance between two 1-d tuples or arrays x and y
    """
    return np.sqrt( np.power(np.array(x) - np.array(y), 2).sum() )

class CenterpointTestCase(unittest.TestCase):
    def setUp(self):
        self.odd = 5
        self.evn = 4

    def test_centerpoint_oddarray(self):
        #         perform the actual test
        self.assertTrue(distance(utils.centerpoint((self.odd,self.odd)), (2,2)) < 1e-15, 
                        'misctools.utils.centerpoint ((5,5)) incorrect)')

    def test_centerpoint_evenarray(self):
        #         perform the actual test
        self.assertTrue(distance(utils.centerpoint((self.evn,self.evn)), (1.5,1.5)) < 1e-15, 
                        'misctools.utils.centerpoint ((4,4)) incorrect)')


if __name__ == "__main__":
    unittest.main()
