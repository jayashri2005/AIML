import json
import requests

payload={
    "jsonrpc":"2.0",
    "id":1,
    "method":"add",
    "params":[10,20]
}

response=requests.post("http://localhost:5000",json=payload)

print("Raw response:",response.text)
print("Parsed response:",response.json()["result"])