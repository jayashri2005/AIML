from fastmcp import FastMCP
import wikipedia
import asyncio


mcp=FastMCP("wikipedia",debug=True)
@mcp.tool
def search(query: str)-> str:
    summary = wikipedia.summary(query, sentences=3)
    return summary

if __name__=="__main__":
    mcp.run(transport="http",port=8000,path="/mcp")