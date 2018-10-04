import unittest, os, glob
import sys
import numpy as np

from nrm_analysis.misctools.utils import Affine2d

"""
    Test pitch algebra for Affine2d class
    anand@stsci.edu 2018.04.12

    run with pytest -s _moi_.py to see stdout on screen
"""

def distance(a,b):
    """ returns float Euclidean distance between two 1-d tuples or arrays a and b
    """
    return np.sqrt( np.power(np.array(a) - np.array(b), 2).sum() )


def corecalc(aff, testpoints, fmt):
    """ input: aff Affine2d object, tuple ot test points to transform.
        returns: tuple of points (a point is np array of len 2)
    """
    print()
    aff.show()
    ql = []
    for p in testpoints:
        q = aff.forward(p)
        # store transformed points if future tests are done by calling code
        ql.append(q)
        print(fmt.format(p[0],p[1],q[0],q[1]))
    return tuple(ql)
    


class Affine2dTestCase(unittest.TestCase):

    def setUp(self):

        self.fmt = "    ({0:+.3f}, {1:+.3f}) -> ({2:+.3f}, {3:+.3f})"
        self.testpoints = (np.array((0,0)), np.array((1,0)), np.array((0,1)), np.array((1,1)))

        mx, my = 1.0, 1.0
        sx, sy= 0.0, 0.0
        xo, yo= 0.0, 0.0
        self.affine_ideal = Affine2d(mx=mx,my=my, 
                                     sx=sx,sy=sy, 
                                     xo=xo,yo=yo, name="Ideal")
        mx, my = -1.0, 1.0
        sx, sy= 0.0, 0.0
        xo, yo= 0.0, 0.0
        self.affine_xrev = Affine2d(mx=mx,my=my, 
                                    sx=sx,sy=sy, 
                                    xo=xo,yo=yo, name="Xreverse")
        mx, my = 2.0, 1.0
        sx, sy= 0.0, 0.0
        xo, yo= 0.0, 0.0
        self.affine_xmag = Affine2d(mx=mx,my=my, 
                                    sx=sx,sy=sy, 
                                    xo=xo,yo=yo, name="Xmag")
        mx, my = 1.0, 1.0
        sx, sy= 0.5, 0.0
        xo, yo= 0.0, 0.0
        self.affine_sigx = Affine2d(mx=mx,my=my, 
                                    sx=sx,sy=sy, 
                                    xo=xo,yo=yo, name="Xshear")

        self.affine_45 = Affine2d(rotradccw=np.pi/4.0, name="45deg")

        self.affine_90 = Affine2d(rotradccw=np.pi/2.0, name="90deg")

        self.affine_180 = Affine2d(rotradccw=np.pi, name="180deg")

        self.affine_360 = Affine2d(rotradccw=2.0*np.pi, name="360deg")

    def test_affine_ideal(self):
        transformedpoints = corecalc(self.affine_ideal, self.testpoints, self.fmt)
        self.assertTrue(abs(distance(self.testpoints, transformedpoints)) < 1e-15, \
                        'error: affine identity tfm failed')

    def test_affine_xrev(self):
        corecalc(self.affine_xrev, self.testpoints, self.fmt)
        self.assertTrue((self.affine_xrev.determinant+1) < 1e-15, \
                        'error: xrev determinant != -1')

    def test_affine_xmag(self):
        corecalc(self.affine_xmag, self.testpoints, self.fmt)
        self.assertTrue((self.affine_xmag.determinant-2) < 1e-15,\
                        'error: xmag determinant != 2')

    def test_affine_sigx(self):
        corecalc(self.affine_sigx, self.testpoints, self.fmt)
        self.assertTrue((self.affine_sigx.determinant-1) < 1e-15,\
                        'error: sigx 1 determinant != 1')

    def test_affine_45(self):
        corecalc(self.affine_45, self.testpoints, self.fmt)
        self.assertTrue((self.affine_45.determinant-1) < 1e-15,\
                        'error: 45 1 determinant != 1')

    def test_affine_90(self):
        corecalc(self.affine_90, self.testpoints, self.fmt)
        self.assertTrue((self.affine_90.determinant-1) < 1e-15,\
                        'error: 90 1 determinant != 1')

    def test_affine_180(self):
        corecalc(self.affine_180, self.testpoints, self.fmt)
        self.assertTrue((self.affine_180.determinant-1) < 1e-15,\
                        'error: 180 1 determinant != 1')

    def test_affine_360(self):
        corecalc(self.affine_360, self.testpoints, self.fmt)
        self.assertTrue((self.affine_360.determinant-1) < 1e-15,\
                        'error: 360 1 determinant != 1')


if __name__ == "__main__":
    unittest.main()
