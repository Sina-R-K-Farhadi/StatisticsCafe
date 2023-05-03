
class Car:
    x = 5

c1 = Car()
print(c1.x)


#%%
n = 7
class Car:
    def __init__(h, v= 99999):
        h.x = v
        h.n  = 18
    def P(h):
        print(h.n)
        print(n)


c1 = Car()
print(c1.x)
c1.P()

#%%

class Car:
    def __init__(self, v):
        self.x = v

    def __PrintSmt(self):
        return "Hi"

    def DoSmt(self):
        return self.__PrintSmt()



c2 = Car(34)
#print(c2.x)
c2.DoSmt()

#%%

class GeometricalShape():

    def __init__(self, name):
        self.name = name


class Rectangle():
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def Area(self):

        return self.width * self.height

r1 = Rectangle(8, 10)
print(r1.Area())

#%%
class Square(Rectangle,GeometricalShape):
    def __init__(self, side):
        Rectangle.__init__(self, side, side)
        GeometricalShape.__init__(self, "square")


    def Area(self):
        return self.width * 4


s1 = Square(10)
print(s1.Area())

