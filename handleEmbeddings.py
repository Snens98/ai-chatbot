from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from llama_index.core import SimpleDirectoryReader
from langchain_community.vectorstores import FAISS
import streamlit as st
import send2trash
import tempfile
import shutil
import torch
import os
import re


init = st.session_state


embeddingModels = ["mixedbread-ai/mxbai-embed-large-v1", "WhereIsAI/UAE-Large-V1",
                    "avsolatorio/GIST-large-Embedding-v0", "BAAI/bge-large-en-v1.5", 
                    "BAAI/bge-base-en-v1.5", "thenlper/gte-large", "thenlper/gte-base", 
                    "intfloat/e5-large-v2", "BAAI/bge-small-en-v1.5"]






######################
# prepare embeddings #
######################

def load_EmbeddingModel(type='cpu'): # type = cpu oder gpu (CUDA)
    
    if torch.cuda.is_available():
        init.embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large", model_kwargs={'device': type})
    else:
        st.info("No Cuda GPU is available. Run in CPU-Mode")
        init.embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large", model_kwargs={'device': type})









def prepare_Document(uploaded_text):

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as temp_file:
        temp_file.write(uploaded_text)
        temp_file_path = temp_file.name

    loader = TextLoader(file_path=temp_file_path)
    init.documents = loader.load()
    
    os.remove(temp_file_path)








def get_unique_filename(base_name, extension, start=1):
    index = start
    while True:
        if index == 1:
            filename = f"{base_name}.{extension}"
        else:
            filename = f"{base_name}_{index}.{extension}"
        if not os.path.exists(filename):
            return filename
        index += 1








def create_embeddings_From_Dokument():

    try:
        # Divide the text into several small blocks (chunks)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=init.chunk_size, chunk_overlap=init.overlap)

        # text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=20)                 # other variant
        docs = text_splitter.split_documents(init.documents)

        # Create vector database from the document with the embedding model
        init.db = FAISS.from_documents(docs, init.embeddings)           # The embedding model must be loaded first

    
        unique_filename = get_unique_filename(f"Datasets/{init.embedding_name}", "db")
        init.db.save_local(unique_filename)

        st.success("The vector representations from the data set were successfully created.")
        st.experimental_rerun()
    except (OSError, RuntimeError) as e:
        st.error(f"Unsupported character in '{unique_filename}'")
        st.info("Change the name. Avoid characters like 'Ä, ö, ?, ', ^' and try again!")






def updateSelectbox():
    datasets_path = 'Datasets'
    dataset_folders = [folder for folder in os.listdir(datasets_path) if os.path.isdir(os.path.join(datasets_path, folder))]
    return dataset_folders





def removeSelectedDataset():
    index = init.selected_index
    datasets_path = 'Datasets'
    dataset_folders = [folder for folder in os.listdir(datasets_path) if os.path.isdir(os.path.join(datasets_path, folder))]

    if index < len(dataset_folders):
        folder_to_remove = dataset_folders[index]
        folder_path = os.path.join(datasets_path, folder_to_remove)

        try:
            shutil.rmtree(folder_path)
            st.success(f"Folder '{folder_path}' removed successfully.")
        except OSError as e:
            st.error(f"Error: {e.strerror}")

        # To automatically remove the index from the select box
        st.rerun()
    else:
        st.error("Invalid index")





########################################
# create new embeddings based on file #
########################################

def handle_Datasets():

    DB_option = None
    datasets_path = 'Datasets'

    if not os.path.exists(datasets_path):
        os.makedirs(datasets_path)

    # List of folders in the datasets directory
    dataset_folders = [folder for folder in os.listdir(datasets_path) if os.path.isdir(os.path.join(datasets_path, folder))]

    try:
        # Streamlit widget for selecting the data set
        DB_option = st.selectbox(
            'Select Dataset: ',
            updateSelectbox()
        )
        topic = DB_option[:-3] if DB_option.endswith('.db') else DB_option
        init.RAGTopic = topic

        if dataset_folders != []:
        # Get the index of the selected option
            init.selected_index = dataset_folders.index(DB_option)

            if st.button("Remove this file", type="primary"):
                removeSelectedDataset()


        # Load the FAISS index file based on the selected option
            index_path = os.path.join(datasets_path, f"{DB_option}")
            init.RAGFilePath = index_path
            init.db = FAISS.load_local(index_path, init.embeddings, allow_dangerous_deserialization=True)
    except (RuntimeError, Exception) as e:
        if dataset_folders != []:
            st.error(f"The file is corrupt or Empty.")
        try:
            st.info(index_path)
            send2trash.send2trash(index_path)
            updateSelectbox()
            st.experimental_rerun()
        except Exception as e:
            if dataset_folders != []:
                st.info("Choose a other file!")

    if DB_option != None:
        init.selectedFile = DB_option
    st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)








def selectedFile():
    return init.selectedFile[:-3]





def handle_EmbeddingLLM():

    option = st.selectbox(label='Selected Embedding-model', options=["intfloat/multilingual-e5-large [1.8 GB]"], label_visibility="visible")
    st.text("")

    col1, col2 = st.columns(([2, 3]))

    with col1:
        embeddingModell_Button = st.button("Load Embedding-Model")

    with col2:
        init.embedding_Mode = st.select_slider('Embedding-Modus', options=['CPU (RAM)', 'GPU [Cuda] (VRAM)'], label_visibility="collapsed")

    st.text("")
    if(embeddingModell_Button):

        if init.embedding_loaded is False:

            if init.embedding_Mode == "CPU (RAM)":
                init.embedding = load_EmbeddingModel('cpu')
            else:
                init.embedding = load_EmbeddingModel('cuda')
            
        init.embedding_loaded = True

    if(init.embedding_loaded):
        st.success(f"""Embedding-Modell: intfloat/multilingual-e5""")
    else:
        st.error("No Embedding-Model loaded!")

    if not init.rag:
        st.info("""To enable the language model to extract information from the file, activate the "RAG" option in the settings on the left-hand side.""", icon="ℹ️")

        








def save_uploaded_file(uploaded_file: bytes, save_dir: str):
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    except Exception as e:
        pass





#Upload new data set, convert to vectors and save in vector database
def create_Embedding_from_new_Dataset():
    st.markdown("<h3>Upload new File and save in vector database</h3>", unsafe_allow_html=True)

    border = st.container(border=True)
    with border:
        st.info("""If you want to use a large file with a lot of text, use the 'GPU [Cuda]' setting for the embedding model. This speeds up the process a lot! But you need free graphics memory and an Nvidea graphic card.""", icon="ℹ️")
        st.text("")
        init.embedding_name = st.text_input(
            "Write Topic of the file here 👇",
            placeholder="Language model",
            value="Language model"
        )
        st.text("")

        # Upload-widget to upload .txt, docx and .pdf files
        uploaded_file = st.file_uploader("Create vector representations from the file", type=["txt", "pdf", "docx"], disabled=not init.embedding_loaded, accept_multiple_files=True)
        save_dir = os.getcwd() + "/data"

        if uploaded_file:

            if st.button("Create vector representation"):

                for uploaded in uploaded_file:
                    save_uploaded_file(uploaded, save_dir)

                reader = SimpleDirectoryReader(input_dir=save_dir)
                documents = reader.load_data()
                combined_text = ""

                if documents:
                    for doc in documents:
                        combined_text += f"{doc.text}\n\n\n"
                
                cleaned_text, removed_chars = remove_non_utf8(combined_text)

                if len(removed_chars) > 0:
                    st.info(f"Removed unsupported characters: {removed_chars}")

                prepare_Document(cleaned_text)
                create_embeddings_From_Dokument()

            st.text("")

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)


def remove_non_utf8(text):
    removed_chars = []
    
    #cleaned_text = text.encode('utf-8', 'ignore').decode('utf-8')
    #printable_chars = ''.join(char for char in cleaned_text if char.isprintable())
    #removed_chars.extend(set(cleaned_text) - set(printable_chars))
    #cleaned_text = printable_chars
    
    # Find all characters that are not in the ASCII range and are not umlauts (ö,ü,ä, for German texts)
    removed_non_utf8 = re.findall(r'(?![öäüÖÄÜ-])[^A-Za-z0-9\x00-\x7F]+', text)
    removed_chars = []
    cleaned_text = re.sub(r'(?![öäüÖÄÜ-])[^A-Za-z0-9\x00-\x7F]+', '', text)
    removed_chars.extend(removed_non_utf8)
    
    return cleaned_text, removed_chars




##########################
# handle with embeddings #
##########################

def get_Embedding_Text_From_Input(results):

    init.vartext = ""

    for doc in results:
        if doc[1] <= init.Euklidischer_Abstand:
            init.vartext += doc[0].page_content + "\n"
    if init.vartext.isspace() or len(init.vartext) == 0:      
        init.vartext = init.notInfo






def search_similarity_embeddings_From_Input(user_question):

    try:
        # Search all relevant vectors that match the input vector
        # results = init.db.similarity_search(user_question)                   

        # The returned distance value is the L2 distance. Therefore a lower score is better. Best results sorted in ascending order
        results = init.db.similarity_search_with_score(user_question, k=init.topk)      #k=x -> Search for the x best results: Vector similarity + Euclidean distance
        init.results = results

        # Cache and filter results
        get_Embedding_Text_From_Input(results)
    except AttributeError as e:
        if init.rag:
            st.info("No File Selected")
        init.results = ""




