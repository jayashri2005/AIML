
# streamlit_app.py
import os
from dotenv import load_dotenv

import streamlit as st
from langgraph.graph import StateGraph
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()



def build_llm(repo_id: str, provider_free: bool = True) -> ChatHuggingFace:
    """
    Build a ChatHuggingFace LLM. If provider_free=True, use a model that runs on the
    regular HF Inference API (no 'Inference Providers' needed).
    """
    if provider_free:
        repo_id = repo_id or "mistralai/Mistral-7B-Instruct-v0.2"

    hf_llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",  
        max_new_tokens=128,
        temperature=0.0,         
    )
    return ChatHuggingFace(llm=hf_llm)



def build_graph(llm: ChatHuggingFace):
    def manager_node(state: dict) -> dict:
        task_input = state.get("task", "")
        user_input = state.get("input", "")

        prompt = """
You are a task manager. Based on the user request below, decide whether it is a:
- translate
- calculate
- summarize

Respond with only one word (translate, summarize, calculate).
Task: %s
""" % task_input

        decision = llm.invoke(prompt).content.strip().lower()
        if decision not in {"translate", "summarize", "calculate"}:
            decision = "default"
        return {"agent": decision, "input": user_input}

    def translate_node(state: dict) -> dict:
        text = state.get("input", "")
        prompt = f"Act like a translator. Only respond with the English translation of the text below:\n\n{text}"
        result = llm.invoke(prompt).content.strip()
        return {"result": result}

    def summarize_node(state: dict) -> dict:
        text = state.get("input", "")
        prompt = f"Summarize the following in 1-2 lines:\n\n{text}"
        result = llm.invoke(prompt).content.strip()
        return {"result": result}

    def calculate_node(state: dict) -> dict:
        expression = state.get("input", "")
        prompt = f"Please calculate and return the result of:\n\n{expression}"
        result = llm.invoke(prompt).content.strip()
        return {"result": result}

    def default_node(state: dict) -> dict:
        return {"result": "Sorry, I could not understand the task."}

    def route_by_agent(state: dict) -> str:
        decision = state.get("agent", "").strip().lower()
        mapping = {
            "translate": "translate",
            "summarize": "summarize",
            "calculate": "calculate",
            "default": "default",
        }
        return mapping.get(decision, "default")

    g = StateGraph(dict)
    g.add_node("manager", manager_node)
    g.add_node("translate", translate_node)
    g.add_node("summarize", summarize_node)
    g.add_node("calculate", calculate_node)
    g.add_node("default", default_node)

    g.set_entry_point("manager")
    g.add_conditional_edges("manager", route_by_agent)

    g.set_finish_point("translate")
    g.set_finish_point("summarize")
    g.set_finish_point("calculate")
    g.set_finish_point("default")

    return g.compile()
    





@st.cache_resource(show_spinner=False)
def get_graph(repo_id: str, provider_free: bool):
    llm = build_llm(repo_id=repo_id, provider_free=provider_free)
    return build_graph(llm)



def main():
    st.set_page_config(page_title="LangGraph Task Manager", layout="centered")
    st.title("🧭 LangGraph Task Manager (Translate / Summarize / Calculate)")

    with st.sidebar:
        st.subheader("LLM Settings")
        provider_free = st.toggle(
            "Use provider‑free HF model (recommended)",
            value=True,
            help="Avoids Hugging Face 'Inference Providers' (403).",
        )
        repo_id = st.text_input(
            "Hugging Face model repo_id",
            value="mistralai/Mistral-7B-Instruct-v0.2" if provider_free else "openai/gpt-oss-120b",
            help=(
                "Examples:\n"
                "- mistralai/Mistral-7B-Instruct-v0.2 (no providers)\n"
                "- HuggingFaceH4/zephyr-7b-beta (no providers)\n"
                "- openai/gpt-oss-120b (needs 'Allow Inference Providers')"
            ),
        )

        token_present = bool(os.getenv("HUGGINGFACEHUB_API_TOKEN"))
        st.caption(f"HF Token loaded: {'✅' if token_present else '❌'} (from .env)")

        if not token_present:
            st.warning("No HUGGINGFACEHUB_API_TOKEN found. Add it to your .env.")

        if not provider_free:
            st.info(
                "Using provider-backed models may require:\n"
                "• Paid/credit plan\n"
                "• Token with 'Allow Inference Providers'\n"
            )

        if st.button("Rebuild graph (clear cache)"):
            get_graph.clear()
            st.success("Cleared. Graph will rebuild on next run.")

    st.markdown("### Enter your task & input")
    c1, c2 = st.columns(2)
    task = c1.text_input("Task", value="Can you translate this?")
    user_input = c2.text_input("Input", value="Bonjour le monde")

    run = st.button("▶ Run")

    if run:
        graph = get_graph(repo_id, provider_free)
        with st.spinner("Running the graph..."):
            try:
                output = graph.invoke({"task": task, "input": user_input})
                st.success("Done")
                st.write("**State output**:")
                st.json(output)
                if "result" in output:
                    st.markdown("**Result**")
                    st.code(output["result"])
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")

    st.divider()
    st.subheader("Workflow Diagram")

    graph = get_graph(repo_id, provider_free)
    with st.spinner("Rendering graph..."):
        try:
            png_bytes = graph.get_graph().draw_mermaid_png()
            st.image(png_bytes, caption="LangGraph (Mermaid PNG)", use_column_width=True)
        except Exception:
            try:
                mermaid_src = graph.get_graph().draw_mermaid()
                st.markdown("Mermaid source (copy to a Mermaid renderer):")
                st.code(mermaid_src, language="mermaid")
            except Exception as e:
                st.info("Could not render Mermaid. Check graph rendering dependencies.")
                st.exception(e)

    st.divider()
    st.markdown("#### Quick tests")
    col_a, col_b, col_c, col_d = st.columns(4)
    if col_a.button("Translate sample"):
        graph = get_graph(repo_id, provider_free)
        out = graph.invoke({"task": "Can you translate this?", "input": "Bonjour le monde"})
        st.code(out.get("result", out))
    if col_b.button("Summarize sample"):
        graph = get_graph(repo_id, provider_free)
        out = graph.invoke({
            "task": "Please summarize the following",
            "input": "LangGraph helps you build flexible multi-agent workflows in python..",
        })
        st.code(out.get("result", out))
    if col_c.button("Calculate sample"):
        graph = get_graph(repo_id, provider_free)
        out = graph.invoke({"task": "what is 12 * 8 + 5?", "input": "12 * 8 + 5"})
        st.code(out.get("result", out))
    if col_d.button("Unknown sample"):
        graph = get_graph(repo_id, provider_free)
        out = graph.invoke({"task": "Can you dance?", "input": "foo"})
        st.code(out.get("result", out))


if __name__ == "__main__":
    main()
