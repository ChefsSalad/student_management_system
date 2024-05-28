from PIL import Image
# from streamlit_shap import st_shap
import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.express as px
import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error,confusion_matrix,accuracy_score,recall_score,precision_score,classification_report,roc_auc_score
import shap
import catboost
from catboost import CatBoostClassifier
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

plt.style.use('default')


feature_mapping = {
    'learned Lessons Num': '学习课程数',
    'finished Task Num': '完成任务数',
    'Study Duration': '学习时长',
    'Note Num': '笔记数',
    'Discussion Participation': '讨论参与度',
    'joined CourseSet Num': '加入课程集数',
    'joined Course Num': '加入课程数',
    'Gender': '性别',
    'Role': '角色'
}

st.set_page_config(
    page_title='学生成绩管控系统',
    page_icon='🕵️‍♀️',
    layout='wide'
)

# dashboard title
# st.title("Real-Time Fraud Detection Dashboard")
st.markdown("<h1 style='text-align: center; color: black;'>学生成绩管控系统</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>Student Performance Management and Control</h1>",
            unsafe_allow_html=True)



df = pd.read_excel(r'D:\fraud-detection-main\fraud-detection-main\fraud_updated1.xlsx')
df_education = pd.read_excel(r'D:\fraud-detection-main\fraud-detection-main\final.xlsx')
ID=0
text_content = ""

def user_input_features():
    global ID
    global text_content
    st.sidebar.header('Catboost预测')
    st.sidebar.write('学生学习情况和个人信息如下⬇️')

    user_id = int(st.sidebar.text_input('输入学生学号'))


    ID=user_id
    #user_id=int(user_id)
    user_data = df_education[df_education['userId'] == user_id].head(1)
    for col, val in user_data.iloc[0].items():
       text_content += f"{col}: {val}\n"
    if not user_data.empty:
        st.sidebar.write(f'查询到学生信息{user_id}！')
        user_data = user_data.iloc[0]

        a1 = st.sidebar.slider('完成课程量', 0.0, 450.0, float(user_data['learned Lessons Num']))
        a2 = st.sidebar.slider('完成任务量', 0.0, 1000.0, float(user_data['finished Task Num']))
        a3 = st.sidebar.slider('学习时长', 2000.0, 222000.0, float(user_data['Study Duration']))
        a4 = st.sidebar.slider('笔记数量', 0.0, 10.0, float(user_data['Note Num']))
        a5 = st.sidebar.slider('讨论参与度', 0.0, 10.0, float(user_data['Discussion Participation']))
        a6 = st.sidebar.slider('课程集数量', 0.0, 8.0, float(user_data['joined CourseSet Num']))
        a7 = st.sidebar.slider('课程数量', 0.0, 80.0, float(user_data['joined Course Num']))
        a8 = st.sidebar.selectbox("性别？", ('Male', 'Female'), index=0 if user_data['Gender'] == 'Male' else 1)
        a9 = st.sidebar.selectbox("角色？", ('student', 'assistant|student'), index=0 if user_data['Role'] == 'student' else 1)

    else:
        st.sidebar.write('未查询到该学生，使用默认数值')
        a1 = st.sidebar.slider('完成课程量', 0.0, 450.0, 0.0)
        a2 = st.sidebar.slider('完成任务量', 0.0, 1000.0, 500.0)
        a3 = st.sidebar.slider('学习时长', 2000.0, 222000.0, 150000.0)
        a4 = st.sidebar.slider('笔记数量', 0.0, 10.0, 2.0)
        a5 = st.sidebar.slider('讨论参与度', 0.0, 10.0, 6.0)
        a6 = st.sidebar.slider('课程集数量', 0.0, 8.0, 4.0)
        a7 = st.sidebar.slider('课程数量', 0.0, 80.0, 40.0)
        a8 = st.sidebar.selectbox("性别？", ('Male', 'Female'))
        a9 = st.sidebar.selectbox("角色？", ('student', 'assistant|student'))


    output = [a1, a2, a3, a4, a5, a6, a7, a8, a9]
    return output

# 添加一个按钮用于跳转至指定链接
if st.sidebar.button("全体学生数据管控大屏"):
    st.sidebar.markdown("https://datav.aliyun.com/share/page/fba125d9121091a8ca42f495fba34571")

outputdf = user_input_features()


# df = pd.read_excel(r'D:\fraud-detection-main\fraud-detection-main\final.xlsx')
file_path = r'D:\fraud-detection-main\fraud-detection-main\final.xlsx'
# df = pd.read_excel(file_path, nrows=8000)

st.title('数据集	🧑‍🎓')
if st.button('查看随机学生数据'):
    st.write(df_education.sample(10))

# st.write(f'The dataset is trained on Catboost with totally length of: 50868. 0️⃣ means passing student, 1️⃣ failing studen. data is unbalanced (not⚖️)')
st.write(f'该数据集有50868个样例，使用Catboost的机器学习模型. 0️⃣ 代表不合格, 1️⃣ 代表合格. ')

unbalancedf = pd.DataFrame(df_education.Grade.value_counts())
st.write(unbalancedf)

# 需要一个count plot
placeholder = st.empty()
placeholder2 = st.empty()
placeholder3 = st.empty()

with placeholder.container():
    f1, f2, f3 = st.columns(3)

    with f1:
        a11 = df[df['Class'] == 1]['learned Lessons Num']
        a10 = df[df['Class'] == 0]['learned Lessons Num']
        hist_data = [a11, a10]
        # group_labels = ['Real', 'Fake']
        fig = ff.create_distplot(hist_data, group_labels=['不合格', '合格'])
        fig.update_layout(title_text='完成课程数量')
        st.plotly_chart(fig, use_container_width=True)
    with f2:
        a21 = df[df['Class'] == 0]['finished Task Num']
        a20 = df[df['Class'] == 1]['finished Task Num']
        hist_data = [a21, a20]
        # group_labels = ['Real', 'Fake']
        fig = ff.create_distplot(hist_data, group_labels=['不合格', '合格'])
        fig.update_layout(title_text='完成任务量')
        st.plotly_chart(fig, use_container_width=True)
    with f3:
        a31 = df[df['Class'] == 1]['Study Duration']
        a30 = df[df['Class'] == 0]['Study Duration']
        hist_data = [a31, a30]
        # group_labels = []
        fig = ff.create_distplot(hist_data, group_labels=['不合格', '合格'])
        fig.update_layout(title_text='学习时长')
        st.plotly_chart(fig, use_container_width=True)

with placeholder2.container():
    f1, f2, f3 = st.columns(3)

    with f1:
        a41 = df[df['Class'] == 1]['Note Num']
        a40 = df[df['Class'] == 0]['Note Num']
        hist_data = [a41, a40]
        # group_labels = ['Real', 'Fake']
        fig = ff.create_distplot(hist_data, group_labels=['不合格', '合格'])
        fig.update_layout(title_text='笔记数量')
        st.plotly_chart(fig, use_container_width=True)
    with f2:
        a51 = df[df['Class'] == 1]['joined CourseSet Num']
        a50 = df[df['Class'] == 0]['joined CourseSet Num']
        hist_data = [a51, a50]
        # group_labels = ['Real', 'Fake']
        fig = ff.create_distplot(hist_data, group_labels=['不合格', '合格'])
        fig.update_layout(title_text='joined CourseSet Num')
        st.plotly_chart(fig, use_container_width=True)
    with f3:
        a61 = df[df['Class'] == 1]['joined CourseSet Num']
        a60 = df[df['Class'] == 0]['joined CourseSet Num']
        hist_data = [a61, a60]
        # group_labels = ['Real', 'Fake']
        fig = ff.create_distplot(hist_data, group_labels=['不合格', '合格'])
        fig.update_layout(title_text='joined CourseSet')
        st.plotly_chart(fig, use_container_width=True)

as1 = df[df['Class'] == 1]['Discussion Participation']
as0 = df[df['Class'] == 0]['Discussion Participation']
hist_data = [as1, as0]
colors = [px.colors.qualitative.Plotly[0], px.colors.qualitative.Plotly[1]]
fig = ff.create_distplot(hist_data, colors=colors, group_labels=['不合格', '合格'])
fig.update_layout(title_text='Discussion Participation')
st.plotly_chart(fig, use_container_width=True)

# df2 = df[['Class','Gender']].value_counts().reset_index()
#
# df3 = df[['Class','role']].value_counts().reset_index()
# class_mapping = {0: 1, 1: 0}
# # 使用 map 方法，将 Class 列中的值根据映射进行转换
# df2['Class'] = df2['Class'].map(class_mapping)
# df3['Class'] = df3['Class'].map(class_mapping)
df2 = pd.read_excel(r'D:\fraud-detection-main\fraud-detection-main\df2.xlsx')
df3 = pd.read_excel(r'D:\fraud-detection-main\fraud-detection-main\df3.xlsx')

with placeholder3.container():
    f2, f3 = st.columns(2)

    with f2:
        # fig = plt.figure()
        fig = px.bar(df2, x='Class', y='count', color='Gender', color_continuous_scale=px.colors.qualitative.Plotly,
                     title=" Gender: 🔴Female; 🔵Male")
        st.write(fig)

    with f3:
        fig = px.bar(df3, x='Class', y='count', color="role", title="role:  ❤️Student; 💙Student|Assistant")
        st.write(fig)

st.title('SHAP 值📈')

# image4 = Image.open(r'D:\fraud-detection-main\fraud-detection-main\summary.png')
shapdatadf = pd.read_excel(r'D:\fraud-detection-main\fraud-detection-main\shapdatadf1.xlsx')
shapvaluedf = pd.read_excel(r'D:\fraud-detection-main\fraud-detection-main\shapvaluedf1.xlsx')

placeholder5 = st.empty()
f2 = placeholder5.container()

# with f1:s
#     st.subheader('Summary plot')
#     st.write('👈 class 0: Passed')
#     st.write('👉 class 1: Failed')
#     st.write(' ')
#     st.write(' ')
#     st.write(' ')
#     st.image(image4)
with f2:
    st.subheader('学生各项学习指标与课程成绩依赖关系图')
    cf = st.selectbox("选择查看的指标", (shapdatadf.columns))

    fig = px.scatter(x=shapdatadf[cf],
                     y=shapvaluedf[cf],
                     color=shapdatadf[cf],
                     color_continuous_scale=['blue', 'red'],
                     labels={'x': 'Original value', 'y': 'shap value'})
    st.write(fig)

features = pd.read_excel(r'D:\fraud-detection-main\fraud-detection-main\features.xlsx')
catmodel = CatBoostClassifier()
catmodel.load_model(r"D:\fraud-detection-main\fraud-detection-main\CatBoost_model")

st.title('执行预测	🏫')
#outputdf = user_input_features1()
outputdf = pd.DataFrame([outputdf], columns=features.columns)
# st.write('学生学习情况如下⬇️')
# st.write(outputdf)



cat_columns = ['Role', 'Gender']
for col in cat_columns:
    if outputdf[col].dtype == 'object':
        outputdf[col] = LabelEncoder().fit_transform(outputdf[col])

p1 = catmodel.predict(outputdf)[0]
p2 = catmodel.predict_proba(outputdf)

# 判断 p1 的值，并生成相应的文本
predicted_class = "不合格" if p1 == 0 else "合格"


explainer = shap.Explainer(catmodel)
shap_values = explainer(outputdf)
print(type(shap_values))
shap_values.feature_names = [feature_mapping.get(name, name) for name in shap_values.feature_names]
placeholder6 = st.empty()
with placeholder6.container():
    f1, f2 = st.columns(2)
    with f1:
        st.markdown(
            """
            <h3>学生学习情况如下⬇️</h3>

            """,
            unsafe_allow_html=True
        )
        #st.write('学生学习情况如下⬇️')
        #st.write(outputdf)
        new_outputdf = outputdf.copy()
        columns = new_outputdf.columns.tolist()
        columns[2], columns[4] = columns[4], columns[2]
        new_outputdf = new_outputdf[columns]
        outputdf=new_outputdf
        st.write(outputdf.iloc[:, :len(outputdf.columns) // 2])
        st.write(outputdf.iloc[:, len(outputdf.columns) // 2:])

        # st.write('User input parameters below ⬇️')
        # st.write(outputdf)
        #st.write(f'预测结果: {predicted_class}')
        # st.write('预测概率：')
        # st.write('0️⃣ means failing student, 1️⃣ passing student')
        # st.write(p2)
        #     st.markdown(
        #         """
        #         <div style="text-align: center;">
        #             <h3>预测概率：</h3>
        #             <p>0️⃣ 代表不合格学生, 1️⃣ 代表合格学生</p>
        #
        #         </div>
        #         """,
        #         unsafe_allow_html=True
        #     )
        st.markdown(
            """
            <h3>预测概率：</h3>
            <p>0️⃣ 代表不合格学生, 1️⃣ 代表合格学生</p>
            """,
            unsafe_allow_html=True
        )
             # st.write('预测概率：', text_align='center')
             # st.write('0️⃣ 表示不及格，1️⃣ 表示及格', text_align='center')

        st.write(p2, text_align='center')


    with f2:
        st.markdown(
            """
            <h3>学生学习指标管控图</h3>
            """,
            unsafe_allow_html=True
        )
        # st_shap(shap.plots.waterfall(shap_values[0]),  height=500, width=1700)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.plots.waterfall(shap_values[0])
        st.pyplot(bbox_inches='tight')


top_indices = np.argsort(shap_values.values.sum(0))[-2:]
bottom_indices = np.argsort(shap_values.values.sum(0))[:2]
top_features = [shap_values.feature_names[i] for i in top_indices]
bottom_features = [shap_values.feature_names[i] for i in bottom_indices]
placeholder = st.empty()  # 占位容器


#st.title("分析助手🤖")
st.markdown(
            """
            <h3>分析助手🤖</h3>

            """,
            unsafe_allow_html=True
        )
st.markdown("""
    <center>
    <blockquote>
        分析数据可得：<br>
        以下两项任务完成较好✌️:<br>
        - <strong>{}</strong> & <strong>{}</strong><br>
        仍需多做努力的是🙌：<br>
        - <strong>{}</strong> & <strong>{}</strong>
    </blockquote>
    </center>
""".format(top_features[0], top_features[1], bottom_features[0], bottom_features[1]), unsafe_allow_html=True)



text_content = f"""
学生学习情况如下：
{text_content}

预测结果: {predicted_class}
预测概率: {p2}

分析助手🤖
学生学习指标分析：
以下两项任务完成较好✌️:
- {top_features[0]} & {top_features[1]}
仍需多做努力的是🙌:
- {bottom_features[0]} & {bottom_features[1]}
"""
# text_content = f"""
#           学生学习情况如下：
#           {outputdf.to_string(index=False)}
#
#           预测结果: {predicted_class}
#           预测概率: {p2}
#
#           分析助手🤖
#           分析数据可得：
#           以下两项任务完成较好✌️:
#           - {top_features[0]} & {top_features[1]}
#           仍需多做努力的是🙌:
#           - {bottom_features[0]} & {bottom_features[1]}
#           """

# 将内容写入txt文件
with open('output.txt', 'w', encoding='utf-8') as file:
    file.write(text_content)

st.download_button(
                label=f"下载学生 {ID} 的报告",
                data=text_content,
                file_name='output.txt',
                mime='text/plain',
            )