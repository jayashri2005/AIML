# to run "  uvicorn fastapi1:appn --reload  "
# http://127.0.0.1:8000/ ->manually enter call/start = http://127.0.0.1:8000/call/start here 8000 is default port
 # 1st return aprm entha return nope 
from fastapi import FastAPI
from pydantic import BaseModel


appn=FastAPI()
@appn.get("/")
def read_root():
    return {"first":"One","second":"Two"}

@appn.get("/call/start")


def start_call():
    return {"message":"API is working"}

@appn.post("/status")
def status():
    return {"status":"API is running fine",
            "update": "4th June 2024"}


@appn.get("/items/{item_id}")
def read_item(item_id):
    return {"item_id": item_id}
    
class Item(BaseModel):
    name: str
    email: str

@appn.post("/create-items/")
def create_item(item: Item):
    #item_dict = item.dict()
    return item.name + " -  " + item.email 
    