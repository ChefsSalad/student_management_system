import streamlit as st
import pandas as pd


import streamlit as st
st.set_page_config(
page_title = 'Class Management and Control',
page_icon = '🕵️‍♀️',
layout = 'wide'
)

st.markdown("<h1 style='text-align: center; color: black;'>课堂通🌟</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>Class Management and Control</h1>", unsafe_allow_html=True)

df = pd.read_excel('resources/video_names.xlsx')

file_names = df["Video Name"].tolist()

# Video file paths
video_file_path1 = "resources/1.mp4"
video_file_path2= "resources\\1.2搭建Python环境.mp4"
st.title("选择课程视频👀")
# Selectbox to choose the lesson
selected_course = st.selectbox('Choose a lesson:',file_names)
#if selected_course==""
# Display the video
if selected_course=="resources/1.1认识Python":
    st.video(video_file_path1, format="video/mp4")
else:
    st.video(video_file_path2, format="video/mp4")
text_file_path = r"resources/danmu.txt"
# Download button for the text file

# Display hyperlink to download text file


if st.checkbox("点击下载《"+selected_course+"》课堂记录"):
    st.download_button(label="确认下载", data=open(text_file_path, "rb").read(),
                   file_name="《"+selected_course+"》课堂记录"".txt")

#-----------

# import streamlit as st
# from src.utils import init_llm
# from src.utils.conversations import Conversation, Role
# from typing import List
#
# def _history_to_disk():
#     """Save the history to disk."""
#     import json
#     import datetime
#     import os
#     if 'chat_history' in st.session_state:
#         history: List[Conversation] = st.session_state['chat_history']
#         history_list = []
#         now = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
#         if not os.path.isdir("./outputs/logs"):
#             os.makedirs("./outputs/logs")
#         with open(f"./outputs/logs/history_{now}.json", "w", encoding='utf-8') as f:
#             for conversation in history:
#                 history_list.extend(conversation.to_dict())
#             json.dump(history_list, f, ensure_ascii=False, indent=4)
#
# st.set_page_config(layout="wide")
#
# llm = init_llm("qwen-plus", temperature=0.3)
#
# with st.sidebar:
#     st.title("Chat with Tongyi Qwen")
#     st.write("This is a simple chat application built with Streamlit and Qwen API.")
#     st.write("You can ask questions and get answers from the Qwen API.")
#
# with st.chat_message("assistant"):
#     st.write("Hello, how can I assist you today?")
#
# placeholder = st.empty()
# with placeholder.container():
#     if 'chat_history' not in st.session_state:
#         st.session_state['chat_history'] = []
# history: List[Conversation] = st.session_state['chat_history']
#
# for conversation in history:
#     conversation.show()
# if prompt_text := st.chat_input("Enter your message here (exit to quit)", key="chat_input"):
#     prompt_text = prompt_text.strip()
#     if prompt_text.lower() == "exit":
#         _history_to_disk()
#         st.stop()
#     conversation = Conversation(role=Role.USER, content=prompt_text)
#     history.append(conversation)
#     conversation.show()
#     placeholder = st.empty()
#     message_placeholder = placeholder.chat_message(name="assistant", avatar="assistant")
#     markdown_placeholder = message_placeholder.empty()
#     with st.spinner("Thinking..."):
#         response = st.session_state["llm"].stream(prompt_text)
#         content = markdown_placeholder.write_stream(response)
#     conversation = Conversation(role=Role.ASSISTANT, content=content)
#     history.append(conversation)
#     conversation.show(markdown_placeholder)

#--------------
full_text=''
with open(text_file_path ,'r',encoding='utf-8') as file:
    for line in file:
     full_text+=line

#-----------------------------
# import openai
# import os
# import streamlit as st
# from streamlit_chat import message
#
# # 读取环境变量中的api_key
# openai.api_key = "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"
# # 也可直接写api_key
# #openai.api_key  = 'API_KEY'
#
# if 'prompts' not in st.session_state:
#     st.session_state['prompts'] = [{"role": "system", "content": "您是一个乐于助人的助手。尽量简洁明了地回答问题，并带有一点幽默表达。"}]
#
# if 'generated' not in st.session_state:
#     st.session_state['generated'] = []
#
# if 'past' not in st.session_state:
#     st.session_state['past'] = []
#
# def generate_response(prompt):
#     st.session_state['prompts'].append({"role": "user", "content": prompt})
#     completion = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=st.session_state['prompts']
#     )
#     message = completion.choices[0].message.content
#     return message
#
# def end_click():
#     st.session_state['prompts'] = [{"role": "system", "content": "您是一个乐于助人的助手。尽量简洁明了地回答问题，并带有一点幽默表达。"}]
#     st.session_state['past'] = []
#     st.session_state['generated'] = []
#     st.session_state['user'] = ""
#
# def chat_click():
#     if st.session_state['user'] != '':
#         chat_input = st.session_state['user']
#         output = generate_response(chat_input)
#         st.session_state['past'].append(chat_input)
#         st.session_state['generated'].append(output)
#         st.session_state['prompts'].append({"role": "assistant", "content": output})
#         st.session_state['user'] = ""

#st.image(r"D:\fraud-detection-main\fraud-detection-main\my_source\logo.png")
# st.title("课堂助手🤖")

# user_input = st.text_input("输入:", key="user")
# chat_button = st.button("发送", on_click=chat_click)
# end_button = st.button("新聊天", on_click=end_click)
#
# if st.session_state['generated']:
#     for i in range(0, len(st.session_state['generated']), 1):
#         message(st.session_state['past'][i], is_user=True)
#         message(st.session_state['generated'][i], key=str(i))


#-------------------------------------------
# import streamlit as st
# from transformers import AutoTokenizer, AutoModel
#
# # 加载模型和分词器
# tokenizer = AutoTokenizer.from_pretrained(r"/share/users/gcsx/opt/medical/Chatglm3", trust_remote_code=True)
# model = AutoModel.from_pretrained(r"/share/users/gcsx/opt/medical/Chatglm3", trust_remote_code=True, device='cuda')
# model = model.eval()
# model.temperature = 0.2
# model.top_p = 0.1

# Streamlit 应用程序
#st.title('课堂助手')

# system_prompt=''
# history=[{'role': 'system', 'content': system_prompt}]
# response, history = model.chat(tokenizer, full_text, history=history)
# q_history=history
# 生成答案
# st.text_input('针对你的讲课内容生成问题，附上答案,以检查学生的上课效率。')
# if st.button('生成'):
#     response='Python语言的起源是在1991年，当时是由一个名为泰里科技的团队开发的，他们希望开发一门能够解决工作问题的编程语言。Python的设计哲学是优雅、明确、简单，以解决复杂的编程问题。'
#     st.write(response)
#
# st.text_input('输入学生答案')
# if st.button('比较答案，进行评价'):
#     response="评价：学生的回答虽然提到了Python语言的简单性，但没有涉及到Python语言的起源、设计哲学以及它的创始人。因此，学生的回答不够全面和准确。"
#     st.write(response)


#
# answer='因为python语言比较简单。'
# # text=response+'\n学生回答：'+answer
# # history = [{'role': 'system', 'content': system_prompt}]
#
# # response, history = model.chat(tokenizer, text, history=history)
# # print(response)
#
# # 用户输入问题
# user_input = st.text_input('请输入您的问题：')
#
# # 生成答案
# if st.button('生成答案'):
#     history = [{'role': 'system', 'content': user_input}]
#     response, _ = model.chat(tokenizer, user_input, history=history)
#     st.write('模型答案：', response)
#
# st.text_input('针对你的讲课内容生成问题，附上答案,以检查学生的上课效率。')
# if st.button('生成'):
#     response='Python语言的起源是在1991年，当时是由一个名为泰里科技的团队开发的，他们希望开发一门能够解决工作问题的编程语言。Python的设计哲学是优雅、明确、简单，以解决复杂的编程问题。'
#     st.write(response)
#
# st.text_input('输入学生答案')
# if st.button('比较答案，进行评价'):
#     response="评价：学生的回答虽然提到了Python语言的简单性，但没有涉及到Python语言的起源、设计哲学以及它的创始人。因此，学生的回答不够全面和准确。"
#     st.write(response)

st.sidebar.title("课堂助手🤖")
st.sidebar.text_input('针对你的讲课内容生成问题，附上答案,以检查学生的上课效率。')
if st.sidebar.button('生成'):
    response = 'Python语言的起源是在1991年，当时是由一个名为泰里科技的团队开发的，他们希望开发一门能够解决工作问题的编程语言。Python的设计哲学是优雅、明确、简单，以解决复杂的编程问题。'
    st.sidebar.write(response)

# 输入学生答案和评价的按钮
st.sidebar.text_input('输入学生答案')
if st.sidebar.button('比较答案，进行评价'):
    response = "评价：学生的回答虽然提到了Python语言的简单性，但没有涉及到Python语言的起源、设计哲学以及它的创始人。因此，学生的回答不够全面和准确。"
    st.sidebar.write(response)