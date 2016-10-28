#!/usr/bin/env python

import functions as F
import numpy as N
import unittest

class TestFunctions(unittest.TestCase):
    def testApproxJacobian1(self):
        slope = 3.0
        def f(x):
            return slope * x + 5.0
        x0 = 2.0
        dx = 1.e-3
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (1,1))
        self.assertAlmostEqual(Df_x, slope)

    def testApproxJacobian2(self):
        A = N.matrix("1. 2.; 3. 4.")
        def f(x):
            return A * x
        x0 = N.matrix("5; 6")
        dx = 1.e-6
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (2,2))
        N.testing.assert_array_almost_equal(Df_x, A)

    def testPolynomial(self):
        # p(x) = x^2 + 2x + 3
        p = F.Polynomial([1, 2, 3])
        for x in N.linspace(-2,2,11):
            self.assertEqual(p(x), x**2 + 2*x + 3)

    def testBivariateQuadratic2D(self):
        q=F.BivariateQuadratic2D([1,2,3,4,5,6],[7,8,9,10,11,12])
        for x in N.linspace(-2,2,11):
            for y in N.linspace(-2,2,11):
                xs=N.matrix([[x],[y]])
                fs=q(xs)
                self.assertEqual(fs.shape, xs.shape)
                self.assertEqual(fs.item(0), x**2+2*y**2+3*x+4*y+5*x*y+6)
                self.assertEqual(fs.item(1), 7*x**2+8*y**2+9*x+10*y+11*x*y+12)

    def testBivariateQuadratic2DJacobian(self):
        q=F.BivariateQuadratic2D([1,0,0,1,0,1],[0,1,0,0,3,0])
        xs1=N.matrix("5;3")
        xs2=N.matrix("0;0")
        xs3=N.matrix("0;1")

        #actual Jacobian at xs1, xs2 calculated analytically by hand
        J1=N.matrix("10 1; 9 21")
        J2=N.matrix("0 1; 0 0")
        J3=N.matrix("0 1; 3 2")

        D1=F.ApproximateJacobian(q, xs1)
        D2=F.ApproximateJacobian(q, xs2)
        D3=F.ApproximateJacobian(q, xs3)

        self.assertEqual(D1.shape, (2,2))
        self.assertEqual(D2.shape, (2,2))
        self.assertEqual(D3.shape, (2,2))

        N.testing.assert_array_almost_equal(D1, J1)
        N.testing.assert_array_almost_equal(D2, J2)
        N.testing.assert_array_almost_equal(D3, J3)



if __name__ == '__main__':
    unittest.main()
