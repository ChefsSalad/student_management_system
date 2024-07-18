### src/utils/__init__.py
from langchain_community.llms.tongyi import Tongyi
import streamlit as st
def init_llm(model_name, temperature=0.3):
    # st.session_state是一个变量寄存器，可储存当前session的变量
    # 由于streamlit是动态加载的，灵活使用st.session_state可以避免一些数据多次加载的情况
    # 若st.session_state中不存在llm或传入的模型名称与保存的不一致时创建或更新
    if not "llm" in st.session_state or st.session_state.llm.model_name!=model_name:
        st.session_state["llm"] = Tongyi(model_name=model_name,
                                model_kwargs={"temperature": temperature})

### main.py
llm = init_llm("qwen-plus", temperature=0.3)