import streamlit as st
import datetime
from utils import ReadData
import pandas as pd
from Globals import PHONE_NUM, PASSWORD
from Classifier import PowerClassifier
from pyecharts.charts import Scatter
from pyecharts import options as opts
from streamlit_echarts import st_pyecharts as st_echarts
import json


def Activity_Recognition():
    """
    活动识别页面
    """
    # 用户信息
    PhoneNum = PHONE_NUM
    password = PASSWORD
    BeeID = '86200001187'
    mac = 'Mt3-M1-84f703120b64'
    
    # 获取今天的日期
    
    # 创建日期选择器，默认为今天
    selected_date = st.date_input(
        "请选择日期",
        help="请选择今天及之前的日期"
    )
    
    # 生成选择日期前一天的日期字符串
    previous_date = selected_date - datetime.timedelta(days=1)
    previous_date_str = previous_date.strftime("%Y-%m-%d")
    selected_date_str = selected_date.strftime("%Y-%m-%d")
    

    # 加载选择日期的数据
    try:
        df_selected = ReadData.ReadData_Day(
            beeId=BeeID, 
            mac=mac, 
            time=selected_date_str, 
            PhoneNum=PhoneNum, 
            password=password, 
            DataType='P'
        )
    except Exception as e:
        st.error(f"加载 {selected_date_str} 数据失败：{str(e)}")
        return
    
    # 加载前一天的数据
    try:
        df_previous = ReadData.ReadData_Day(
            beeId=BeeID, 
            mac=mac, 
            time=previous_date_str, 
            PhoneNum=PhoneNum, 
            password=password, 
            DataType='P'
        )
    except Exception as e:
        st.error(f"加载 {previous_date_str} 数据失败：{str(e)}")
        return
    
    # 合并两个dataframe
    df_combined = pd.concat([df_previous, df_selected], ignore_index=True)
    df_combined = df_combined.sort_values('Time').reset_index(drop=True)
    
    st.success(f"数据加载成功")

    classifier = PowerClassifier(
        model_path='final_power_classification_model.pth',
        data = df_combined,
    )
    results = classifier.classify_daily_data(str(selected_date))
    

    # 从results中提取结果
    power = []
    labels = []
    labels_index = []
    time_list = []
    for result in results:
        power.append(result['power'])
        labels.append(label_mapping[result['predicted_mode']])
        labels_index.append(activity_labels[result['predicted_mode']])
        time_list.append(result['time_str'])

    # 创建活动识别结果的折线图
    create_activity_chart(time_list, power, labels, labels_index)

def create_activity_chart(time_list, power, labels, labels_index):
    """
    创建活动识别结果的散点图 - 显示功率散点，鼠标悬停时显示状态信息
    """
    # 创建散点图
    scatter_chart = Scatter(init_opts=opts.InitOpts(width='1200px', height='600px'))
    
    # 设置x轴为时间轴
    scatter_chart.add_xaxis(time_list)
    
    # 创建包含状态信息的数据点
    # 将功率和状态信息组合成自定义数据格式
    power_with_state = []
    for i, (pwr, label) in enumerate(zip(power, labels)):
        power_with_state.append({
            'value': [i, pwr],  # x轴索引，y轴功率值
            'name': f"{time_list[i]} | {label}",  # 显示时间和状态
            'itemStyle': {'color': '#1f77b4'}
        })
    
    # 添加功率散点
    scatter_chart.add_yaxis(
        series_name="功率",
        y_axis=power_with_state,
        symbol_size=8,
        label_opts=opts.LabelOpts(is_show=False)
    )
    
    # 设置图表全局选项
    scatter_chart.set_global_opts(
        title_opts=opts.TitleOpts(
            title="活动识别结果",
            subtitle="鼠标悬停查看各时间点的活动状态"
        ),
        tooltip_opts=opts.TooltipOpts(
            trigger="item",
            axis_pointer_type="cross"
        ),
        xaxis_opts=opts.AxisOpts(
            name="时间",
            type_="category",
            axislabel_opts=opts.LabelOpts(
                interval=20,  # 每20个点显示一个标签
                rotate=45     # 旋转45度
            )
        ),
        yaxis_opts=opts.AxisOpts(
            name="功率 (W)",
            type_="value"
        ),
        legend_opts=opts.LegendOpts(
            pos_top="10%",
            pos_left="center",
            orient="horizontal"
        ),
        datazoom_opts=[
            opts.DataZoomOpts(
                is_show=True,
                type_="slider",
                pos_bottom="5%",
                range_start=0,
                range_end=100
            )
        ]
    )
    
    # 在Streamlit中显示图表
    st_echarts(scatter_chart, height=600)
    
if __name__=='__page__':
    st.title('活动识别')

    activity_labels = {
        'mode_1': 1,
        'mode_2': 2,
        'mode_3': 3,
        'mode_4': 4,
        'mode_5': 5,
        'mode_6': 6,
        'mode_7': 7,
        'mode_8': 8,
    }
    label_mapping = {
        'mode_1': '无人活动',
        'mode_2': '轻度办公', 
        'mode_3': '餐饮休息',
        'mode_4': '标准办公',
        'mode_5': '夜间办公',
        'mode_6': '深夜空闲',
        'mode_7': '周末活跃',
        'mode_8': '周末空闲'
    }


    with st.container():
        Activity_Recognition()

