import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from urllib3 import response

class JSONRPCHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h1>JSON-RPC Server</h1><p>Use POST requests to call JSON-RPC methods</p>")

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        request = json.loads(body)
        response={
            "jsonrpc": "2.0",
            "id": request.get("id")
        }
        try:
            method=request["method"]
            params=request.get("params",[])
            if method =="add":
                result=params[0]+params[1]
                response["result"]=result
            else:
                raise Exception("Method not found")
            

        except Exception as e:
            response["error"]={
                "code": -32603,
                "message": str(e)
            }
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

def run():
    server=HTTPServer(("localhost",4000),JSONRPCHandler)
    print("Server started on http://localhost:4000")
    server.serve_forever()

if __name__=="__main__":
    run()    


    #rcp - json