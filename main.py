import handleEmbeddings as embeddings
import manageNewModels as manageLLM
import writeResponseInFile as wrf
import streamlit as st
import init as vars
import sidebar
import helper
import model

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

def checkGPU():
    try:
        import torch
        if not torch.cuda.is_available():
            st.info("GPU is not available. Model run now with CPU-Mode!")
    except Exception as e:
        st.info("GPU is not available. Model run now with CPU-Mode!")



def AI_Chat():

    helper.displayHeader("AI-Chatbot")

    if model.isModelLoaded(init.model_loaded):
        model.setFirstMessage("Hi, how can I help you today? 🙃")
        model.displayChatHistory(last_answer_count = 2)
        model.displayLastChatMessages(numberOfInputResponsePairs = 2)

        if user_prompt:
            model.process_user_prompt(user_prompt)
            wrf.writeLLMAnswerToFileIfEnabled('LLM_Answers.docx', user_prompt, init.fullResponse, enabled=init.writeInDocs)
    else:
        st.error("A language model must be activated in order to use the application.")
        st.info("To select a language model use the list at the top left and press 'Update language model'", icon="ℹ️")




def RAG():

    st.markdown("<br><h3>Select dataset and load Embedding-Model</h3>", unsafe_allow_html=True)

    with st.container(border=True):
        embeddings.handle_Datasets()
        with st.spinner("load embeddings..."):
            pass
            embeddings.handle_EmbeddingLLM()
    helper.br(2)
    embeddings.uploadNewDatasetProcess()
    


def Extracted_text():
    st.markdown("<br><br><h3>Extracted document text</h3>", unsafe_allow_html=True)
    st.markdown(f"{init.vartext}", unsafe_allow_html=True)
    st.divider()
    st.markdown("<br><br><h3>Extracted text segments with distance score</h3>", unsafe_allow_html=True)
    st.write(init.results)



def Infos():
    helper.memoryUsage()



def other():
    helper.br(2)
    init.dayDateInfo = st.toggle("Give model information about time, day, date & Username")
    helper.br()
    wholePrompt = st.text_area(label="Whole prompt", value=init.prompt, height=800)




def storeUserInputInGlobalVar(init_user_prompt):
    init.user_prompt = init_user_prompt



#############
#   Start   #
#############
if user_prompt := st.chat_input("Write here a Question...", key="user_input"):
    storeUserInputInGlobalVar(init_user_prompt = user_prompt)





def main():

    checkGPU()
    vars.initVars()
    
    tab1, tab2, tab3, tab4, tab5, tabOther = st.tabs(
        [" 💬 AI-Chat ", " 🗃️ Retrieval Augmented Generation (RAG)", " 📃 Extracted text", " ℹ️ Infos ", " ✚ Add Language-Model", " Other "])

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
        tab6, tab7, tab8 = st.tabs(
            [" 🔍 Search language model ", " 📂 Available language models ", " 🌐 Trending models "])

        with tab6:
            manageLLM.searchModelsAndRelatedQuants(NumberOfSearchResults = 25)

        with tab7:
            with st.container(border=True):
                manageLLM.show_Installed_LLMS()

        with tab8:
            manageLLM.trendingModels(maximalModels=20)
    with tabOther:
        other()

    with st.sidebar:
        Sidebar()


if __name__ == '__main__':
    main()




