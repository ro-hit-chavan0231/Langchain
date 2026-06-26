from typing import TypedDict

class Person(TypedDict):

    name:str
    age:int

new_person: Person = {"name" :'rohit', 'age' : 45}

print(new_person)