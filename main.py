import handleEmbeddings as embeddings
import manageNewModels as manageLLM
import writeResponseInFile as wrf
import streamlit as st
import init as vars
import sidebar
import helper
import model
import sys
import threading
import psutil
import GPUtil
import os
import ctypes
import requests



init = st.session_state


st.set_page_config(page_title="Local-AI-Chat")
user_prompt = " "



def Sidebar():
    st.markdown("<h5>Local-AI-Chat v. 0.8<br>Author: Snens98</h5>", unsafe_allow_html=True)
    st.link_button("Go to Gitgub", "https://github.com/snens98")
    st.divider()

    sidebar.handle_model()
    sidebar.handle_toggle_Buttons()
    sidebar.imageFileUpload()
    sidebar.handle_Prompts()
    sidebar.handle_LLM_Settings()

    # Center button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        sidebar.handle_ChatHistory_Button()






def saveChat():
    init.saveChat = """ """

    if init.usechatMemory:
        num_entries = len(st.session_state.messages)
        num_to_save = min(num_entries, 7)
        last_entries = st.session_state.messages[-num_to_save:]

        for entry in last_entries:

            if entry['role'] == 'user':
                init.saveChat += init.historyTemplateUSER.format(entry['content'])

            elif entry['role'] == 'assistant':
                init.saveChat +=  init.historyTemplateBOT.format(entry['content'])
            
        init.saveChat = helper.replace_brackets(init.saveChat)






def AI_Chat():

    helper.displayHeader("AI-Chatbot")

    chat_history = st.expander("Chat history")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, how can I help you today? 🙃"}]

    if init.model_loaded:
        model.display_Chathistory()

    with chat_history:
        for message in st.session_state.messages[:-1]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])     

    if user_prompt and init.model_loaded and not model.init.error:
        model.process_user_prompt()
        wrf.write_the_language_model_answer_to_file('LLM_Answers.docx', user_prompt, init.fullResponse)

    if not init.model_loaded:
        st.error("A language model must be activated in order to use the application.")
        st.info("To select a language model use the list at the top left and press 'update language model'", icon="ℹ️")

    saveChat()




def exit():

    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == 'python':
            # Finde den untergeordneten Prozess von python (cmd.exe) und beende ihn
            for child_proc in psutil.Process(proc.info['pid']).children(recursive=True):
                if child_proc.name() == 'cmd.exe':
                    child_proc.terminate()
                    break



def show_error_message():
    st.error("The language model is too large. The program is closed to avoid complications", icon='❌')




def show_error_message(message):
    MB_YESNO = 0x00000004  # Schaltflächen Ja und Nein
    MB_ICONQUESTION = 0x00000020  # Fragezeichen-Icon

    # Anzeigen der MessageBox mit Ja/Nein-Schaltflächen
    ctypes.windll.user32.MessageBeep(0xFFFFFFFF)  # Spielt den Fehlerton ab
    result = ctypes.windll.user32.MessageBoxW(0, message, "Modell to large", MB_YESNO | MB_ICONQUESTION)
    return result












def monitor_memory():

    try:
        while True:
                        
            total_memory = psutil.virtual_memory()
            total_memory_usage = total_memory.used / (1024 * 1024 * 1024)

            max_memory = helper.get_max_memory()
            max_vram = helper.get_max_vram()

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


            threading.Event().wait(1)
    except KeyboardInterrupt:
        print("Stopped.")

























def RAG():

    st.markdown("<br><h3>Select dataset and load Embedding-Model</h3>", unsafe_allow_html=True)
    border = st.container(border=True)
    with border:
        embeddings.handle_Datasets()
        with st.spinner("load embeddings..."):
            pass
            embeddings.handle_EmbeddingLLM()
    helper.br(2)
    embeddings.create_Embedding_from_new_Dataset()
    





def Extracted_text():
    st.markdown("<br><br><h3>Extracted document text</h3>", unsafe_allow_html=True)
    st.code(f"{init.vartext}")
    st.divider()
    st.markdown("<br><br><h3>Extracted text segments with distance score</h3>", unsafe_allow_html=True)
    st.write(init.results)







def Infos():
    helper.memoryUsage()







#############
#   Start   #
#############
if user_prompt := st.chat_input("Write here a Question...", key="user_input"):
    init.user_prompt = user_prompt
    model.LLm_Response(user_prompt)










def main():

    vars.initVars()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([" 💬 AI-Chat ", " 🗃️ Retrieval Augmented Generation (RAG)", " 📃 Extracted text", " ℹ️ Infos ", " ✚ Add Language-Model"])

    with tab1:
        AI_Chat()

    with tab2:
        RAG()

    with tab3:
        Extracted_text()

    with tab4:
        Infos()

    with tab5:
        helper.br()
        tab6, tab7, tab8 = st.tabs([" 🔍 Search language model ", " 📂 Available language models ", " 🌐 Trending models "])

        with tab6:
            manageLLM.searchModelsAndRelatedQuants()

        with tab7:
            with st.container(border=True):
                manageLLM.show_Installed_LLMS()
        with tab8:
            manageLLM.trendingModels(20)
            pass

    with st.sidebar:
        Sidebar()


    if not init.memUsageThread:
        init.memUsageThread = True
        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.start()



if __name__ == '__main__':
    main()




