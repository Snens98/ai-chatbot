from datetime import datetime
import streamlit as st
import threading
import psutil
import GPUtil
import ctypes
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
def replaceBracketThatCodeCanStored(text):
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
    




    
def exit():
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == 'python':
            for child_proc in psutil.Process(proc.info['pid']).children(recursive=True):
                if child_proc.name() == 'cmd.exe':
                    child_proc.terminate()
                    break



def show_error_message():
    st.error("The language model is too large. The program is closed to avoid complications", icon='❌')




def show_error_message(message):
    MB_YESNO = 0x00000004  # Yes and No buttons
    MB_ICONQUESTION = 0x00000020  # Question mark icon

    # Display the MessageBox with Yes/No buttons
    ctypes.windll.user32.MessageBeep(0xFFFFFFFF)  # Plays the error sound
    result = ctypes.windll.user32.MessageBoxW(0, message, "Modell to large", MB_YESNO | MB_ICONQUESTION)
    return result







def monitor_memory():
    timer = 0

    try:
        while True:

            timer+=1      
            total_memory = psutil.virtual_memory()
            total_memory_usage = total_memory.used / (1024 * 1024 * 1024)

            max_memory = get_max_memory()
            max_vram = get_max_vram()

            gpus = GPUtil.getGPUs()

            if gpus:
                first_gpu = gpus[0]
                usedvram = (first_gpu.memoryUsed / 1024.0)
            else:
                usedvram = 0.0
                    
            usedvram = (float("{:.2f}".format(usedvram))) 
            max_vram = (float("{:.2f}".format(max_vram)))

            total_memory_usage = (float("{:.2f}".format(total_memory_usage)))
            max_memory = (float("{:.2f}".format(max_memory)))

            vram = max_vram/1.03 # puffer
            ram = max_memory/1.03 # puffer

            if usedvram >=(vram) and total_memory_usage >= ram:
                if show_error_message("The language model is too large. Close the program to avoid complications?") == 6:
                    os._exit(0)            

            # After 1h shutdown App
            if timer > 3600:
                os._exit(0)

            threading.Event().wait(1)
    except KeyboardInterrupt:
        print("Stopped.")




init = st.session_state
def startThread_monitorMemory():
    
    if not init.memUsageThread:
        init.memUsageThread = True

    monitor_thread = threading.Thread(target=monitor_memory)
    monitor_thread.daemon = True
    monitor_thread.start()




def checkGPU():
    try:
        import torch
        if not torch.cuda.is_available():
            st.info("GPU is not available. Model run now with CPU-Mode!")
    except Exception as e:
        st.info("GPU is not available. Model run now with CPU-Mode!")




