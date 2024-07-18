
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
video_file_path2= "resources/1.2搭建Python环境.mp4"
st.title("选择课程视频👀")
# Selectbox to choose the lesson
selected_course = st.selectbox('Choose a lesson:',file_names)
#if selected_course==""
# Display the video
if selected_course=="resources/1.mp4":
    st.video(video_file_path1, format="video/mp4")
else:
    st.video(video_file_path2, format="video/mp4")
text_file_path = r"resources/danmu.txt"
# Download button for the text file

# Display hyperlink to download text file


if st.checkbox("点击下载《"+selected_course+"》课堂记录"):
    st.download_button(label="确认下载", data=open(text_file_path, "rb").read(),
                   file_name="《"+selected_course+"》课堂记录"".txt")


full_text=''
with open(text_file_path ,'r',encoding='utf-8') as file:
    for line in file:
     full_text+=line



st.sidebar.title("课堂助手🤖")
st.sidebar.text_input('针对你的讲课内容生成问题，附上答案,以检查学生的上课效率。')
question = st.sidebar.text_input('针对你的讲课内容生成问题，附上答案，以检查学生的上课效率。')
answer = ""

if st.sidebar.button('生成'):
    if question == 'Python语言的起源是什么？':
        answer = 'Python语言的起源可以追溯到1991年，由荷兰程序员Guido van Rossum创造。当时，Guido van Rossum在设计Python时的目标是创造一种易于阅读、简洁而功能强大的编程语言，以提高程序员的生产力。'
    elif question == 'Python的设计哲学是什么？':
        answer = 'Python的设计哲学是优雅、明确、简单，以解决复杂的编程问题。'
    else:
       answer = '抱歉,这个问题和本堂内容没有太大的相关性。'
    st.sidebar.write(answer)

# 输入学生答案和评价的按钮
student_answer = st.sidebar.text_input('请输入学生答案')

if st.sidebar.button('比较答案，进行评价'):
    if question == 'Python语言的起源是什么？':
        if student_answer == 'Python语言的起源可以追溯到1991年，由荷兰程序员Guido van Rossum创造。当时，Guido van Rossum在设计Python时的目标是创造一种易于阅读、简洁而功能强大的编程语言，以提高程序员的生产力。':
            feedback = "评价：学生的回答足够全面和准确，答卷时可得满分"
        else:
            feedback = "评价：学生的回答虽然提到了Python语言的简单性，但没有涉及到Python语言的起源、设计哲学以及它的创始人。因此，学生的回答不够全面和准确。"
    elif student_answer.lower().strip() == 'python语言的设计哲学是优雅、明确、简单。':
        feedback = "评价：学生的回答提到了Python语言的设计哲学，但缺少了详细的描述和例子，建议进一步补充。"
    else:
        feedback = "评价：学生的回答不够全面和准确，建议再仔细回顾相关知识点。"

    st.sidebar.write(feedback)