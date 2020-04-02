class Animal:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.sound = None

    def call(self):
        print(self.sound)


class Cat(Animal):
    def __init__(self, name, age):
        # Animal.__init__(self, name, age)
        self.sound = 'MEOW'


frank = Cat('Frank', 3)
print(type(frank) is Cat,
      type(frank) is Animal,
      isinstance(frank, Cat),
      isinstance(frank, Animal), sep='\n'
      )
