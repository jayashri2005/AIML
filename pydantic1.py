from typing import Dict, List, Optional
from typing_extensions import Annotated
from pydantic import AnyUrl, BaseModel, EmailStr, Field
class person2(BaseModel):
    name: str 
    age: int
    city: str
    weight: float
    email: EmailStr
    myurl:AnyUrl


r=person2(name="Alice",age=30,city="New York",weight=65.5,email='alice@test.com',myurl='http://example.com')
print(r.name, r.age, r.city, r.weight, r.email, r.myurl)

d={'name': 'Bob', 'age': 25, 'city': 'Los Angeles','weight':70.5,'email':'bob@test.com','myurl':'http://google.com'}
r1=person2(**d)
print(r1.name,r1.age,r1.city,r1.weight,r1.email,r1.myurl)


class Patient1(BaseModel):
    name: Annotated[str, Field(max_length=50, title="Patient Name", description="Name of the patient")]
    age: int = Field(gt=0, lt=120)
    email: EmailStr
    myurl: AnyUrl
    weight: Annotated[float, Field(gt=0, strict=True)]
    married: Optional[bool] = Field(default=None, title="Marital Status", description="Is the patient married?")
    allergies: Annotated[List[str], Field(default_factory=list, title="Allergies", description="List of allergies")]
    contact_details: Dict[str, str]

patient_info = {
    'name': 'John Doe',
    'age': 45,
    'email': 'abc@gmail.com',
    'myurl': 'http://example.com',
    'weight': 80.5,
    'married': True,
    'allergies': ['penicillin', 'peanuts'],
    'contact_details': {'phone': '123-456-7890'},
}

patient = Patient1(**patient_info)
print(patient)

class Person:
    def __init__(self, name: str, age: int, city: str):
        self.name = name
        self.age = age
        self.city = city
    
    @classmethod
    def from_string(cls, data_str):
        name, age, city = data_str.split(',')
        return cls(name, int(age), city)
p = Person.from_string("Charlie,28,Chicago")
print(p.name, p.age, p.city)


from pydantic import field_validator
class Patient2(BaseModel):
    name: str
    age: int
    city: str
    email: EmailStr
    weigth

    @classmethod
    def from_string(cls, data_str: str) -> "Patient2":
        name, age, city = data_str.split(',')
        return cls(name=name, age=int(age), city=city)