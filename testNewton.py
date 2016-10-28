#!/usr/bin/env python

import newton
import unittest
import numpy as N
import functions as F

class TestNewton(unittest.TestCase):
    def testLinear(self):
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(2.0)
        self.assertEqual(x, -2.0)

    def testMaxIterException(self):
        p = F.Polynomial([1,0,0,0])
        solver = newton.Newton(p, tol=1.e-15, maxiter=2)
        guess=5.0   #would take more than 2 iterations to reach 0 from here
        self.assertRaises(RuntimeError, solver.solve, guess)

if __name__ == "__main__":
    unittest.main()
