import json
import requests

payload={
    "jsonrpc":"2.0",
    "id":1,
    "method":"add",
    "params":[2,3]
}

response=requests.post("http://localhost:4000",json=payload)

print("Raw response:",response.text)
print("Parsed response:",response.json()["result"])