import numpy as N

def ApproximateJacobian(f, x, dx=1e-6):
    """Return an approximation of the Jacobian Df(x) as a numpy matrix"""
    try:
        n = len(x)
    except TypeError:
        n = 1
    fx = f(x)
    Df_x = N.matrix(N.zeros((n,n)))
    for i in range(n):
        v = N.matrix(N.zeros((n,1)))
        v[i,0] = dx
        Df_x[:,i] = (f(x + v) - fx)/dx
    return Df_x

class BivariateQuadratic2D(object):
    """creates the 2D function f(x,y) = (u(x,y), v(x,y)) where
    coeffs1=[a, b, c, d, e, f]
    coeffs2=[A, B, C, D, E, F]
    and the function returned is f(x,y) = (ax^2 + by^2 + cx + dy + exy + f, Ax^2 + By^2 + Cx + Dy + Exy + F)
    f is returned as a 2D numpy matrix

    initialise an object using q=BivariateQuadratic2D(coeffs1, coeffs2)
    evaluate using q((x,y)) where (x,y) is a two element numpy column matrix
    """
    def __init__(self, coeffs1, coeffs2):
        self._coeffs1 = coeffs1
        self._coeffs2 = coeffs2

    def __repr__(self):
        return "2D Bivariate Quadratic(%s)(%s)" % (", ".join([str(x) for x in self._coeffs1]),(", ".join([str(x) for x in self._coeffs2])))

    def f(self,xs):
        x=xs.item(0)
        y=xs.item(1)

        u = self._coeffs1[0]*x**2 + self._coeffs1[1]*y**2 + self._coeffs1[2]*x + self._coeffs1[3]*y + self._coeffs1[4]*x*y+self._coeffs1[5]
        v = self._coeffs2[0]*x**2 + self._coeffs2[1]*y**2 + self._coeffs2[2]*x + self._coeffs2[3]*y + self._coeffs2[4]*x*y+self._coeffs2[5]

        f=N.matrix([[u],[v]])
        return f

    def __call__(self, xs):
        return self.f(xs)

    def Jacobian(self, xs):
        x=xs.item(0)
        y=xs.item(1)

        dudx= 2*self._coeffs1[0]*x+self._coeffs1[2]+self._coeffs1[4]*y
        dudy= 2*self._coeffs1[1]*y+self._coeffs1[3]+self._coeffs1[4]*x
        dvdx= 2*self._coeffs2[0]*x+self._coeffs2[2]+self._coeffs2[4]*y
        dvdy= 2*self._coeffs2[1]*y+self._coeffs2[3]+self._coeffs2[4]*x

        df=N.matrix([[dudx, dudy], [dvdx, dvdy]])

        return df


class Polynomial(object):
    """Callable polynomial object.

    Example usage: to construct the polynomial p(x) = x^2 + 2x + 3,
    and evaluate p(5):

    p = Polynomial([1, 2, 3])
    p(5)"""

    def __init__(self, coeffs):
        self._coeffs = coeffs

    def __repr__(self):
        return "Polynomial(%s)" % (", ".join([str(x) for x in self._coeffs]))

    def f(self,x):
        """
        I updated this to try get my cubic root to be more accurate, it was getting f(x)=0.0 for x~1e-6
        ans = self._coeffs[0]
        for c in self._coeffs[1:]:
            ans = x*ans + c
        return ans"""
        ans=0
        for n, c in enumerate(self._coeffs[::-1]):
            ans+=c*x**n
        return ans

    def __call__(self, x):
        return self.f(x)

    def Jacobian(self, x):
        ans=0
        for n, c in enumerate((self._coeffs[::-1])[1:]):
            ans+=(n+1)*c*x**n
        return ans
