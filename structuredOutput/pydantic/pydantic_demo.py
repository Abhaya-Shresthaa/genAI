from pydantic import BaseModel, EmailStr, Field
from typing import Optional

#class ma j define vaxa teii nai use hunu parxa pachiii panii
#typedDict ma compulsory thena
class Student(BaseModel):
    name: str
    roll_no: int = 6 ##putting default values
    gpa: float = Field(gt=2, lt= 4.0001, description="This is the marks in gpa highest is 4", default= 3.5)
    faculty: Optional[str] = None
    email: EmailStr
    
new_student = {
    # "name": 32, ##it won't let it to be integer 
    "name" : "Abhaya",
    "roll_no": 6,
    "gpa": 4,
    "faculty" : "Computer",
    "email": "hello@gmail.com"
}

student = Student(**new_student)


## changing it to dictionary
student_dict = dict(student)
student_json = student.model_dump_json()
print(type(student_dict))
print(student_json)
