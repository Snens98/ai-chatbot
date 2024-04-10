from huggingface_hub import hf_hub_download
from PIL import UnidentifiedImageError
import manageNewModels as manageLLM
import promptTemplates as pt
import streamlit as st
import prompt as pr
import helper
import model
import json
import os


init = st.session_state


# The function searches for an entry with a specific index in a JSON file and returns the corresponding template. 
def searchedForAnEntryWithASpecificIndex(selection_index):
    with open("model_options.json", "r") as f:
        data = json.load(f)

    for key, value in data.items():
        if value.get("index") == selection_index:
            return value.get("template", "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user{}<|im_end|>\n<|im_start|>assistant")
    
    return "Template not found"




def isTheModelCurrentlyLoaded():

    if init.llm == None:
        return False
    return init.llm



def getPathToModelOrDownload():
    return hf_hub_download(repo_id=init.repo_id, filename=init.model_file_name, repo_type='model')




# This function updates a language model selection box based on the option chosen. It retrieves model information from a 
# JSON file, sets the model index accordingly, and updates the selected language model in memory upon button click. 
# It handles model loading and updates flags accordingly.
def update_LLM_In_Selectbox(option):

    model_info = manageLLM.read_model_options_from_file('model_options.json').get(option, {})

    if model_info:

        init.model = model_info['index']

        if st.button("Update language model", help=f"Load or update the selected language model in the memory"):
            
            init.selectedLLM = model_info['model_file_name']

            with st.spinner("Load model..."):

                if isTheModelCurrentlyLoaded():
                    model.remove_llm()

                init.repo_id = model_info['repo_id']
                init.model_file_name = model_info['model_file_name']

                with st.spinner(f"Download {init.model_file_name}"):
                    init.model_path = getPathToModelOrDownload()

                model.create_llm()

                if init.llm is None:
                    return
                
                init.model_loaded = True
                init.model_updated = True
                st.rerun()
            





# This function display_model_load_success_message(option, condition) displays a success message if a certain condition is met. 
# It retrieves model information from a JSON file based on the provided option and displays a success message indicating 
# that the language model specified by the option was loaded successfully.
def display_model_load_success_message(option, condition):
    if condition:
        model_info = manageLLM.read_model_options_from_file('model_options.json').get(option, {})
        st.success(f"The language model {model_info['model_file_name']} was loaded successfully!")




# This function is responsible for managing the language model selection. It displays a header indicating the purpose of the selection, 
# provides a select box for choosing a language model, arranges buttons for updating and removing the language model, and displays a success message 
# if a language model was successfully loaded.
def handle_model():

    st.markdown("<h3><Center>Select Language-Model</Center></h3>", unsafe_allow_html=True)

    # remove mmproj-Files from selectbox (for vision-models)
    options = manageLLM.read_model_options_from_file('model_options.json').keys()
    filtered_options = [option for option in options if 'mmproj' not in option]
    option = st.selectbox('Sprachmodell', filtered_options, label_visibility="collapsed")
    
    col_update_LLM_In_Selectbox, col2_Unload_language_model = st.sidebar.columns(2)

    with col_update_LLM_In_Selectbox:
        update_LLM_In_Selectbox(option)

    with col2_Unload_language_model:
        if st.button("Unload language model", type="primary", help=f"Unload the currently selected language model from the memory (but not from the disk)"):
            model.remove_llm()
            st.info(f"Successfully removed!")
            st.rerun()
    display_model_load_success_message(option, init.model_updated and init.model_loaded)

    st.divider()
    helper.br()







# To manage toggle buttons for various settings. It presents toggle buttons for options such as RAG, Chat-Memory usage, and writing language model answers in a .docx file.
# Additionally, it visually indicates the status of the RAG toggle button with different colored text depending on its state.
def handle_toggle_Buttons():
    
    with st.container(border=True):

        col1_initRAG, col2_RAG_status = st.columns(2)

        with col1_initRAG:
            init.rag = st.toggle('RAG', label_visibility="visible")

        init.usechatMemory = st.toggle('Use Chat-Memory', label_visibility="visible")
        init.writeInDocs = st.toggle('Write language model answers in .docx', label_visibility="visible")

        with col2_RAG_status:
            if init.rag:
                st.markdown("<p style='color:#dffde9; background-color:#173928; border-radius:5px; text-align:center;'>RAG active!</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:#f5d5d5; background-color:#3e2428; border-radius:5px; text-align:center;'>RAG not active!</p>", unsafe_allow_html=True)
    helper.br(2)






# The primary function of systemPromptExpander(height) is to display a text area with system prompts based on the current state of the RAG toggle button. 
# If the RAG toggle is active (init.rag is True), it displays the prompt text defined in pr.promptText. Otherwise, it displays the prompt text defined in pr.nonRAG_Prompt. 
def systemPromptExpander(height):

    promptText = pr.promptText().replace("{", "").replace("}", "")
    nonRAG_Prompt = pr.nonRAG_Prompt.replace("{", "").replace("}", "")

    if init.rag:
        system_prompt = st.text_area(label=" ", value=promptText, height=height)
    else:
        system_prompt = st.text_area(label=" ", value=nonRAG_Prompt, height=height)

    return system_prompt






def createTempImage(image):

    if image != None:

        temp_folder = "tempImages"
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        file_path = os.path.join(temp_folder, image.name)

        with open(file_path, "wb") as f:
            f.write(image.getbuffer())

        init.file_path = file_path
        return image
    
        


def imageFileUpload():
    init.vision = st.toggle("Activate vision funktion")
    image = st.file_uploader(label=" Upload image for vision model", accept_multiple_files=False, type=["png", "jpg"])
    init.imageUpload = createTempImage(image)

    try:
        if image:
            st.image(init.file_path)
    except UnidentifiedImageError:
        helper.errorMsg(error="The image is corrupted or has an unsupported format. It cannot be analyzed", info="Upload a other Image (.PNG or .JPG) and try agan!")





# The basic task of the function handle_Prompts() is to manage and display application settings related to prompting. 
# This includes displaying system prompts, prompt templates, additional instructions, and responses if no suitable information can be retrieved.
def handle_Prompts():

    helper.br()
    st.divider()
    st.markdown("<h1><Center> ⚙️ Application Settings ⚙️ </Center></h1>", unsafe_allow_html=True)
    helper.br()
    st.markdown("<h3><Center>Prompting</Center></h3>", unsafe_allow_html=True)

    expander_SystemPrompt = st.expander("System-Prompt")
    with expander_SystemPrompt: 
        pr.init.system_prompt = systemPromptExpander(height=400)


    expander_PromptTemplate = st.expander("Prompt-Template")
    with expander_PromptTemplate:
        template = st.selectbox(label=" ", options=pt.getTemplateList(), index=0)
        init.template = pt.getTemplate(template, searchedForAnEntryWithASpecificIndex(init.model))


    expander_AdditionalInstructions = st.expander("Additional instructions (at the end)")
    expander_ChatHistory = st.expander("Chat-History-Template")
    expander_noSuitableInfo = st.expander("Response if no suitable info can be taken from the file (RAG)")


    helper.dividerBr()

    with expander_AdditionalInstructions:
        _end_Instruction = st.text_area(label=" ", value=pr.end_Instruction, height=200)
        init.endInstruction = _end_Instruction

    with expander_ChatHistory:
        init.historyTemplateUSER = st.text_area("user:", "<|im_start|>user{}<|im_end|>", height=50)
        init.historyTemplateBOT = st.text_area("AI:", "<|im_start|>assistant{}<|im_end|>", height=50)
        
    with expander_noSuitableInfo:
        text = "Important: Do not answer the following question! Answer only with: I don't know. Google yourself."
        init.notInfo = st.text_area(label=" ", value=text, height=200)
    
    

        
    


# To manage and display settings related to language models and embeddings. 
# It presents sliders and select sliders for adjusting various parameters related to language model generation and embedding processing. 
def handle_LLM_Settings():

    st.markdown("<h3><Center>Language-Model Settings</Center></h3>", unsafe_allow_html=True)

    border = st.container(border=True)
    with border:
        init.gpu_layer = st.slider('Offloaded layers to GPU:', -1, 100, 20, disabled=init.download_Model)
        init.temperature = st.slider('Temperature:', 0.0, 1.0, 0.2, disabled=init.download_Model)
        init.max_Token, init.n_ctx = st.select_slider('Maximum Output Tokens and n_ctx:', options = list(range(8, 4097)), value=(512, 2048))
        init.top_p = st.slider('top_p:', 0.0, 1.0, 0.9)
        init.min_p = st.slider('min_p:', 0.00, 0.50, 0.05)
        init.top_k = st.slider('top_k:', 1, 100, 30)
        init.repeat_penalty = st.slider('repeat_penalty:', 0.0, 5.0, 1.0)

    helper.dividerBr()

    st.markdown("<h3><Center>Embedding Settings</Center></h3>", unsafe_allow_html=True)

    border = st.container(border=True)
    with border:
        init.topk = st.slider('top-k:', 1, 10, 4)
        init.chunk_size = st.slider('Chunk_Size:', 50, 1024, 600)
        init.overlap = st.slider('Chunk_Overlap:', 10, 512, 50)
        init.Euklidischer_Abstand = st.slider('Max. Euclidean distance:', 0.0, 1.0, 0.38)

    helper.br()
    helper.dividerBr()






# Provision of a button labeled "Delete chat history". When this button is clicked (if st.button("Delete chat history"):), 
# the chat history is deleted by clearing the list of messages saved in the session state (st.session_state.messages = []).
def handle_ChatHistory_Button():
    if st.button("Delete chat history"):
        st.session_state.messages = []
        st.rerun()



