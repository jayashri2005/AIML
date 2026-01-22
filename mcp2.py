#json to json rpc then read in client and back jsonrpc to json 

from fastmcp import Client
import asyncio

async def call_tool(name: str):
    client = Client("http://localhost:8000/mcp")
    async with client:
        result = await client.call_tool("greet",{"name": name})
        print(result)

if __name__ == "__main__":
    asyncio.run(call_tool("John"))