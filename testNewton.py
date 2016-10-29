#!/usr/bin/env python

import newton
import unittest
import numpy as N
import functions as F
#from mock import patch - I was going to use this to test that analytic derivative was called but then realised that mock
#isn't necessarily part of a standard python install so I took it out


class TestNewton(unittest.TestCase):
    def testLinear(self):
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(2.0)
        self.assertEqual(x, -2.0)

    def testQuadratic(self):
        p=F.Polynomial([1,-1,-6])    #x^2-x+6 has roots 3,-2
        solver = newton.Newton(p, tol=1.e-15, maxiter=20)
        x1=solver.solve(3.5)
        x2=solver.solve(-1.5)
        self.assertEqual(x1, 3.0)
        self.assertEqual(x2, -2.0)

    def testQuadratic2D(self):
        q1=F.BivariateQuadratic2D([2,0,0,0,0,0],[0,2,0,0,0,0]) #x^2+y^2
        guess=N.matrix("0.1;-0.1")
        solver = newton.Newton(q1, tol=1.e-13, maxiter=50)
        x=solver.solve(guess)
        self.assertEqual(x.shape, (2,1))
        N.testing.assert_array_almost_equal(x, N.matrix("0;0"))

    def testCubic(self):
        p = F.Polynomial([1,0,1,0]) #root is x=0
        solver = newton.Newton(p, tol=1.e-15, maxiter=50)
        x1=solver.solve(0.1)
        x2=solver.solve(-0.1)
        self.assertAlmostEqual(x1, 0.)
        self.assertAlmostEqual(x2, 0.)

    def testMaxIterException(self):
        p = F.Polynomial([1,0,0,0])
        solver = newton.Newton(p, tol=1.e-15, maxiter=2)
        guess=5.0   #would take more than 2 iterations to reach 0 from here
        self.assertRaises(RuntimeError, solver.solve, guess)

    def testLinearAnalyticJac(self):
        f = lambda x : 3.0 * x + 6.0
        df= lambda x: N.matrix("3.0")
        solver = newton.Newton(f, tol=1.e-15, maxiter=2, Df=df)
        x = solver.solve(2.0)
        self.assertEqual(x, -2.0)

    def testQuadraticAnalyticJac(self):
        p=F.Polynomial([1,-1,-6])    #x^2-x+6 has roots 3,-2
        solver = newton.Newton(p, tol=1.e-15, maxiter=20, Df=p.Jacobian)
        x1=solver.solve(3.5)
        x2=solver.solve(-1.5)
        self.assertEqual(x1, 3.0)
        self.assertEqual(x2, -2.0)

    def testQuadratic2DAnalyticJac(self):
        q=F.BivariateQuadratic2D([2,0,0,0,0,0],[0,2,0,0,0,0]) #x^2+y^2
        guess=N.matrix("0.1;-0.1")
        solver = newton.Newton(q, tol=1.e-13, maxiter=50, Df=q.Jacobian)
        x=solver.solve(guess)
        self.assertEqual(x.shape, (2,1))
        N.testing.assert_array_almost_equal(x, N.matrix("0;0"))

    def testCubicAnalyticJac(self):
        p = F.Polynomial([1,0,1,0]) #root is x=0
        solver = newton.Newton(p, tol=1.e-15, maxiter=50, Df=p.Jacobian)
        x1=solver.solve(0.1)
        x2=solver.solve(-0.1)
        self.assertAlmostEqual(x1, 0.)
        self.assertAlmostEqual(x2, 0.)

    def testSinusoidalAnalyticJac(self):
        s=F.Sinusoid(2., 5., 0.)
        solver = newton.Newton(s, tol=1.e-15, maxiter=50, Df=s.Jacobian)
        x1=solver.solve(0.1)
        x2=solver.solve(-0.1)
        self.assertAlmostEqual(x1, 0.)
        self.assertAlmostEqual(x2, 0.)

    def testStep(self):
        f=lambda x: 1/2.*x**2
        solver = newton.Newton(f, tol=1.e-15, maxiter=20, Df=lambda x:1.*x)
        x_new=solver.step(10.)
        self.assertAlmostEqual(x_new, 5.)

    def testMaxRadius(self):
        s=F.Sinusoid(1., 3., N.pi/2.)  #cosine, roots at +-pi/2
        r=N.pi/8.
        guess=N.pi/4.
        solver = newton.Newton(s, tol=1.e-15, maxiter=50, Df=s.Jacobian)
        x=solver.solve(guess)
        self.assertAlmostEqual(x, N.pi/2)    #check that it reaches within maxiter if no radius imposed

        solver = newton.Newton(s, tol=1.e-15, maxiter=50, Df=s.Jacobian, maxrad=r)  #included max radius
        self.assertRaises(RuntimeError, solver.solve, guess)   #should not reach solution within maxrad





if __name__ == "__main__":
    unittest.main()
