def differentiate(func, x) :
    for i in range(8) :
        dx = 10**(-i)
        res = (func(x + dx) - func(x - dx)) / (2 * dx)
        print("dx = {0} | f`({1}) = {2}".format(dx, x, res))
        
def quadratic(x) :
    return 3*x**4+2*x**3-5*x**2+7*x-10

differentiate(quadratic, 4)