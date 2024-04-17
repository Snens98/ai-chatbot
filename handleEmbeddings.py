from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from llama_index.core import SimpleDirectoryReader
from langchain_community.vectorstores import FAISS
import streamlit as st
import send2trash
import tempfile
import shutil
import helper
import time
import os
import re


init = st.session_state


embeddingModels = ["mixedbread-ai/mxbai-embed-large-v1", "WhereIsAI/UAE-Large-V1",
                    "avsolatorio/GIST-large-Embedding-v0", "BAAI/bge-large-en-v1.5", 
                    "BAAI/bge-base-en-v1.5", "thenlper/gte-large", "thenlper/gte-base", 
                    "intfloat/e5-large-v2", "BAAI/bge-small-en-v1.5"]


model_name="intfloat/multilingual-e5-small"







######################
# prepare embeddings #
######################

def load_EmbeddingModel_device(device='cpu'): # type = cpu oder gpu (CUDA)
    
    try:
        import torch

        if torch.cuda.is_available():
            init.embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': device})
        else:
            st.info("No Cuda GPU is available. Run in CPU-Mode")
            init.embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
    except ImportError:
        print("Torch not installed!")
        init.embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})








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








def create_embeddings_From_Dokument(text):

    try:
        # Divide the text into several small blocks (chunks)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=init.chunk_size, chunk_overlap=init.overlap)
        docs = text_splitter.split_documents(text)

        # Create vector database from the document with the embedding model
        init.db = FAISS.from_documents(docs, init.embeddings)           # The embedding model must be loaded first
    
        unique_filename = get_unique_filename(f"Datasets/{init.embedding_name}", "db")
        init.db.save_local(unique_filename)

        setLitteSpace()
        st.success("The vector representations from the data set were successfully created.")
        time.sleep(2)
        st.experimental_rerun()

    except (OSError, RuntimeError) as e:
        st.error(f"Unsupported character in '{unique_filename}'")
        st.info("Change the name. Avoid characters like 'Ä, ö, ?, ', ^' and try again!")






def currentListOfFoldersOFSavedFiles(savedFiles_path):
    ListOfFoldersOFSavedFiles = [folder for folder in os.listdir(savedFiles_path) if os.path.isdir(os.path.join(savedFiles_path, folder))]
    return ListOfFoldersOFSavedFiles





def removeSelectedDataset(index, savedFiles_path):

    ListOfFoldersOFSavedFiles = currentListOfFoldersOFSavedFiles(savedFiles_path)

    if index < len(ListOfFoldersOFSavedFiles):
        folder_to_remove = ListOfFoldersOFSavedFiles[index]
        folder_path = os.path.join(savedFiles_path, folder_to_remove)

        try:
            #shutil.rmtree(folder_path)
            send2trash.send2trash(folder_path)

            st.success(f"Folder '{folder_path}' removed successfully.")
        except OSError as e:
            st.error(f"Error: {e.strerror}")

        # To automatically remove the index from the select box
        st.rerun()
    else:
        st.error("Invalid index")




def createFolderForSavedFilesWhenNotExist(savedFiles_path):
    if not os.path.exists(savedFiles_path):
        os.makedirs(savedFiles_path)



def setTopicOfUploadedFile(savedFilesForRAG):
    topic = savedFilesForRAG[:-3] if savedFilesForRAG.endswith('.db') else savedFilesForRAG
    init.RAGTopic = topic



def loadFAISSIndexFileBasedOnSelectedOption(savedFiles_path, savedFilesForRAG):

    index_path = os.path.join(savedFiles_path, f"{savedFilesForRAG}")
    init.RAGFilePath = index_path
    init.db = FAISS.load_local(index_path, init.embeddings, allow_dangerous_deserialization=True)
    return index_path





########################################
# create new embeddings based on file #
########################################

def handle_Datasets():

    savedFilesForRAG = None
    savedFiles_path = 'Datasets'
    createFolderForSavedFilesWhenNotExist(savedFiles_path)

    # List of folders in the datasets directory
    ListOfFoldersOFSavedFiles = [folder for folder in os.listdir(savedFiles_path) if os.path.isdir(os.path.join(savedFiles_path, folder))]

    try:
        # Streamlit widget for selecting the dataset
        savedFilesForRAG = st.selectbox(
            'Select Dataset: ',
            currentListOfFoldersOFSavedFiles(savedFiles_path)
        )
        setTopicOfUploadedFile(savedFilesForRAG)

        if ListOfFoldersOFSavedFiles != []:
            selectedIndex = ListOfFoldersOFSavedFiles.index(savedFilesForRAG)
            if st.button("Remove this file", type="primary"):
                removeSelectedDataset(selectedIndex, savedFiles_path)

    except (RuntimeError, Exception):
        if ListOfFoldersOFSavedFiles != []:
            st.error(f"The file is corrupt or Empty.")

    if savedFilesForRAG != None:
        init.selectedFile = savedFilesForRAG
    helper.br()




def setLitteSpace():
    st.text("")



def isEmbeddingModelLoaded():
    loaded = init.embedding_loaded
    return loaded




def loadEmbeddingModel():
    if not isEmbeddingModelLoaded():

        if init.embedding_Mode == "CPU (RAM)":
            init.embedding = load_EmbeddingModel_device('cpu')
        else:
            init.embedding = load_EmbeddingModel_device('cuda')
        
        init.embedding_loaded = True

    if isEmbeddingModelLoaded():
        st.success(f"Embedding-Modell: {model_name}")
    else:
        st.error("No Embedding-Model loaded!")





def handle_EmbeddingLLM():

    # TODO: implement a selection of different embedding-models here
    option = st.selectbox(label='Selected Embedding-model', options=["intfloat/multilingual-e5-large [0.8 GB]"], label_visibility="visible")
    setLitteSpace()

    col1_embeddingModell_Button, col2_embedding_Mode = st.columns(([2, 3]))

    with col1_embeddingModell_Button:
        embeddingModell_Button = st.button("Load Embedding-Model")

    with col2_embedding_Mode:
        init.embedding_Mode = st.select_slider('Embedding-Modus', options=['CPU (RAM)', 'GPU [Cuda] (VRAM)'], label_visibility="collapsed")

    setLitteSpace()
    if(embeddingModell_Button):
        loadEmbeddingModel()

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






def extractTextFromDocuments(uploaded_file, save_dir):

    for uploaded in uploaded_file:
        save_uploaded_file(uploaded, save_dir)

    reader = SimpleDirectoryReader(input_dir=save_dir)
    documents = reader.load_data()
    combined_text = ""

    if documents:
        for doc in documents:
            combined_text += f"{doc.text}\n\n\n"
    
    cleaned_text, removed_chars = remove_non_Supported_Chars(combined_text)

    if len(removed_chars) > 0:
        st.info(f"Removed unsupported characters: {removed_chars}")

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as temp_file:
        temp_file.write(cleaned_text)
        temp_file_path = temp_file.name

    loader = TextLoader(file_path=temp_file_path)
    extractedText = loader.load()
    os.remove(temp_file_path)
    return extractedText





def createTextInputFieldForTopic(lable):
    setLitteSpace()
    init.embedding_name = st.text_input(
        label=lable,
        placeholder="Language model",
        value="Language model"
    )
    setLitteSpace()





#Upload new data set, convert to vectors and save in vector database
def uploadNewDatasetProcess():
    st.markdown("<h3>Upload new File and save in vector database</h3>", unsafe_allow_html=True)

    with st.container(border=True):
        st.info("""If you want to use a large file with a lot of text, use the 'GPU [Cuda]' setting for the embedding model. This speeds up the process a lot! But you need free graphics memory and an Nvidea graphic card.""", icon="ℹ️")

        createTextInputFieldForTopic(lable = "Write Topic of the file here 👇")

        # Upload-widget to upload .txt, docx and .pdf files
        uploaded_file = st.file_uploader("Create vector representations from the file", type=["txt", "pdf", "docx"], disabled=not init.embedding_loaded, accept_multiple_files=True)
        save_dir = os.getcwd() + "/data"

        if uploaded_file:
            if st.button("Create vector representation"):
                text = extractTextFromDocuments(uploaded_file, save_dir)
                create_embeddings_From_Dokument(text)
            setLitteSpace()


        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)




def clean_non_printable_characters(text, removed_chars):
    cleaned_text = text.encode('utf-8', 'ignore').decode('utf-8')
    printable_chars = ''.join(char for char in cleaned_text if char.isprintable())
    removed_chars.extend(set(cleaned_text) - set(printable_chars))
    cleaned_text = printable_chars
    return removed_chars, removed_chars




def remove_non_Supported_Chars(text, clean_non_printable_chars=False):
    removed_chars = []
    
    if clean_non_printable_chars:
        cleaned_text, removed_chars = clean_non_printable_characters()
    
    # Find all characters that are not in the ASCII range and are not umlauts (ö,ü,ä, for German texts)
    removed_non_utf8 = re.findall(r'(?![öäüÖÄÜ-ß“„–])[^A-Za-z0-9\x00-\x7F]+', text)
    removed_chars = []
    cleaned_text = re.sub(r'(?![öäüÖÄÜ-ß“„–])[^A-Za-z0-9\x00-\x7F]+', '', text)
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






def find_relevant_context(user_question):

    try:
        results = init.db.similarity_search_with_score(user_question, k=init.topk)      #k=x -> Search for the x best results: Vector similarity + Euclidean distance
        init.results = results

        get_Embedding_Text_From_Input(results)
    except AttributeError as e:
        if init.rag:
            st.info("No File Selected")
        init.results = ""
    except AssertionError:
        helper.errorMsg(error="The selected file does not match the current embedding model", info="Select the correct file or change it to the appropriate embedding model")




