from datetime import datetime
import streamlit as st
import psutil
import GPUtil
import os



def br(number=1):
    for _ in range(number):
        st.markdown("<br>", unsafe_allow_html=True)

def dividerBr():
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)


def memoryUsage():

    info_container = st.empty()
    text  = f""" """
    gpus = GPUtil.getGPUs()

    if gpus:
        first_gpu = gpus[0]
        text += f"""GPU:  {first_gpu.name} \n\n"""
        text += f"""VRAM: {first_gpu.memoryUsed} MB\n\n"""
    else:
        info_container.markdown("No GPU")

    pid = os.getpid()

    process = psutil.Process(pid)
    memory_info = process.memory_full_info()

    text += f"""RAM:  {memory_info.uss / (1024 * 1024):.2f} MB\n"""

    st.code(text)




# to be able to use chat history for programming code
def replace_brackets(text):
    modified_text = ""
    for char in text:
        if char == "{":
            modified_text += "{{"
        elif char == "}":
            modified_text += "}}"
        else:
            modified_text += char
    return modified_text






def get_current_date_and_time():
    now = datetime.now()
    current_date = now.strftime("%d/%m/%Y")
    current_time = now.strftime("%H:%M:%S")
    return current_date, current_time



def get_current_weekday():
    return datetime.today().strftime('%A')








def errorMsg(error, info):

    st.error(error, icon='❌')
    st.info(info, icon='ℹ️')







def displayHeader(text):
    br(1)
    st.header(text)
    br(1)






def get_max_memory():
    #  Maximum RAM size in megabytes
    max_ram = psutil.virtual_memory().free
    max_ram_gb = max_ram / (1024 ** 3)  # 1024^3 = 1 GB
    return max_ram_gb




def get_max_vram():

    #  Maximum VRAM size in megabytes
    gpu_list = GPUtil.getGPUs()
    if gpu_list:
        max_vram = max([gpu.memoryTotal for gpu in gpu_list])
        max_vram_gb = max_vram / 1024        # 1024^2 = 1 MB
    else:
        return 0.0
    return max_vram_gb




def get_color_based_on_memory(size, max_ram, max_vram):

    float_numberSize = float(size)
    maxMem = max_ram + max_vram

    if maxMem < float_numberSize:
        return "#000000"  # Schwarz
    elif max_ram < float_numberSize and max_vram < float_numberSize:
        return "#FF7F7F"  # Hellrot
    elif max_ram < float_numberSize or max_vram < float_numberSize:
        return "#FFFF00"  # Gelb
    else:
        return "#7FFF7F"  # Hellgrün