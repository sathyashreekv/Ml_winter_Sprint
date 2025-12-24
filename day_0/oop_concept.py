"""
Docstring for day_0 23-12-25 .oop_concept
Object Oriented programming conecpt helps in maintaining codebae foter the teamcollaboration,modular coding and resuabality.
we orgainize tthe code uisng classes and objects
object is a intance of a class and class is a blueprint to create objects.
classes ->represent real world entities
class-> attributes(properties,data) and methods(behaviors,actions performed on data)

features
Organizes code into classes and objects
Supports encapsulation to group data and methods together
Enables inheritance for reusability and hierarchy
Allows polymorphism for flexible method implementation
Improves modularity, scalability, and maintainability
"""
''' self is a reference to the current instance of the class .It allow us to access the attribute and method of the objects.
'''
# class Car:
#     engine='petrol'
#     def __init__(self,color,model,year):
#        self.color=color
#        self.model=model
#        self.year=year
#     def start(self):
#         print(f"car {self.model} started")
# maruti_suzuki=Car('red','swift',2020)
# maruti_suzuki.start()
# print(maruti_suzuki.engine)

# bmw=Car('black','X5',2025)
# bmw.start()
# print(bmw.engine)

'''
inheitance -ability to acquire the proprties from the parent class
single inheritance->a child inherit fromingle parent class
multiple inheritance->a child class inherits from multiple parent class
multi level->a child class inherit from parent class which in turn inherits from another class
hierarchical->multiple child inherit from a sinngle parent class
hybrid-combination of  2 or more type of inheritance

'''
'''
class Car:
    def __init__(self,model,color,year,engine):
        self.model=model
        self.color=color
        self.year=year
        self.engine=engine

    def start(self):
        print(f"starting .... {self.model}")
    def check_engine(self,model,year):
        print(f"{self.model} Model engine is {self.engine}and we are checking its condition")
class Bmw(Car): #single inheritance
    def horn(self):
        print(f"wroooommmmmmm")
class X5(Bmw):
    def __init__(self,capacity,price):
        self.capacity=capacity
        self.price=price
    def about_feature(self):
        print(f"Features of {self.model} are :")
        print(f"Model name :{self.model}")
        print(f" Model color:{self.color}")
        print(f" Model year:{self.year}")
        print(f" Model capacity(no of seats):{self.capacity}")
        print(f" Model cost:{self.price}")

class EV:
    def __init__(self,battery):
        self.battery=battery
    def get_battery_name(self):
        print(f"Battery:{self.battery}")

class TataNexonEv(Car,EV):
    def smart_meter(self):
        print(f" Model :{self.model} and its battery{self.battery}")

b1=Car('swift','white',2021,'petrol')
b1.start()
b1.check_engine(b1.model,b1.year)

b2=Bmw('x6','black',2023,'petrol')
b2.horn()

b3=X5(5,'$15000')
b3.about_feature()

b4=TataNexonEv('Thar','black',2024,'none','electric')
b4.get_battery_name()
b4.smart_meter()
print(b4.battery)
'''

"""
here we got a runtime error because we are calling parent  class constructor directly without using super() which cannot call this from child during muliple class inheritance
using super() and **kwargs we can resolve this issue prevent the break in method resolution order (MRO)
"""

class Car:
    def __init__(self,model,color,year,engine,**kwargs):
        super().__init__(**kwargs)
        self.model=model
        self.color=color
        self.year=year
        self.engine=engine

    def start(self):
        print(f"starting .... {self.model}")
    def check_engine(self,model,year):
        print(f"{self.model} Model engine is {self.engine}and we are checking its condition")
class Bmw(Car): #single inheritance
    def __init__(self, model, color, year, engine, brand, **kwargs):
        super().__init__(model=model, color=color, year=year, engine=engine, **kwargs)
        self.brand=brand
    def horn(self):
        print(f"wroooommmmmmm")
class X5(Bmw):
    def __init__(self, model, color, year, engine, brand, capacity, price, **kwargs):
        super().__init__(model=model, color=color, year=year, engine=engine, brand=brand, **kwargs)
        self.capacity=capacity
        self.price=price
    def about_feature(self):
        print(f"Features of {self.model} are :")
        print(f"Model name :{self.model}")
        print(f" Model color:{self.color}")
        print(f" Model year:{self.year}")
        print(f" Model capacity(no of seats):{self.capacity}")
        print(f" Model cost:{self.price}")

class EV:
    def __init__(self,battery,**kwargs):
        super().__init__(**kwargs)
        self.battery=battery
    def get_battery_name(self):
        print(f"Battery:{self.battery}")

class TataNexonEv(Car,EV):
    def __init__(self,model,color,year,engine,battery):
        super().__init__(model=model,color=color,year=year,engine=engine,battery=battery)

    def smart_meter(self):
        print(f" Model :{self.model} and its battery{self.battery}")

b1=Car('swift','white',2021,'petrol')
b1.start()
b1.check_engine(b1.model,b1.year)

b2=Bmw('x6','black',2023,'petrol','BMW')
b2.horn()

b3=X5('X5','silver',2024,'petrol','BMW',5,'$15000')
b3.about_feature()

b4=TataNexonEv('Thar','black',2024,'none','electric')
b4.get_battery_name()
b4.smart_meter()
print(b4.battery)

'''polymorphism - ability to take many forms
method overloading -same method name with different parameters in the same class
method overriding - child class redefines parent class method
'''

# METHOD OVERLOADING - Python doesn't support traditional overloading
# Last defined function overwrites previous ones
# Use default parameters or *args instead

class Calculator:
    # Method overloading using default parameters
    def add(self, a, b, c=0):
        return a + b + c
    
    # Method overloading using *args
    def multiply(self, *args):
        result = 1
        for num in args:
            result *= num
        return result

# METHOD OVERRIDING - Child class redefines parent method
class Vehicle:
    def start(self):
        print("Vehicle starting...")
    
    def sound(self):
        print("Vehicle makes sound")

class Car(Vehicle):
    # Overriding parent's start method
    def start(self):
        print("Car engine starting with key...")
    
    # Overriding parent's sound method
    def sound(self):
        print("Car goes vroom vroom")

class Bike(Vehicle):
    # Overriding parent's start method
    def start(self):
        print("Bike starting with kick...")
    
    # Overriding parent's sound method
    def sound(self):
        print("Bike goes putt putt")

# Testing method overloading
calc = Calculator()
print(f"Add 2 numbers: {calc.add(2, 3)}")
print(f"Add 3 numbers: {calc.add(2, 3, 4)}")
print(f"Multiply 2 numbers: {calc.multiply(2, 3)}")
print(f"Multiply 4 numbers: {calc.multiply(2, 3, 4, 5)}")

# Testing method overriding (polymorphism)
vehicles = [Vehicle(), Car(), Bike()]

for vehicle in vehicles:
    vehicle.start()  # Different behavior for each class
    vehicle.sound()  # Different behavior for each class
    print("---")

"""
ABSTRACT CLASSES AND INTERFACES
Abstract class - cannot be instantiated, contains abstract methods that must be implemented by child classes
Interface - defines a contract that classes must follow (Python uses ABC for this)
"""

from abc import ABC, abstractmethod

# ABSTRACT CLASS
class Animal(ABC):
    def __init__(self, name):
        self.name = name
    
    # Concrete method (has implementation)
    def sleep(self):
        print(f"{self.name} is sleeping")
    
    # Abstract method (must be implemented by child classes)
    @abstractmethod
    def make_sound(self):
        pass
    
    @abstractmethod
    def move(self):
        pass

# INTERFACE (Pure abstract class - only abstract methods)
class Flyable(ABC):
    @abstractmethod
    def fly(self):
        pass
    
    @abstractmethod
    def land(self):
        pass

# Concrete classes implementing abstract class
class Dog(Animal):
    def make_sound(self):
        print(f"{self.name} barks: Woof!")
    
    def move(self):
        print(f"{self.name} runs on four legs")

class Bird(Animal, Flyable):  # Multiple inheritance from abstract class and interface
    def make_sound(self):
        print(f"{self.name} chirps: Tweet!")
    
    def move(self):
        print(f"{self.name} walks and flies")
    
    # Implementing interface methods
    def fly(self):
        print(f"{self.name} is flying high")
    
    def land(self):
        print(f"{self.name} landed safely")

# Testing abstract classes and interfaces
# animal = Animal("Generic")  # ERROR: Cannot instantiate abstract class

dog = Dog("Buddy")
dog.make_sound()
dog.move()
dog.sleep()

bird = Bird("Sparrow")
bird.make_sound()
bird.move()
bird.fly()
bird.land()
bird.sleep()

print("---")

# Polymorphism with abstract classes
animals = [Dog("Rex"), Bird("Eagle")]
for animal in animals:
    animal.make_sound()  # Different implementations
    animal.move()        # Different implementations
    print("---")

