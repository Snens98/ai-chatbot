from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
import streamlit as st
import tempfile
import PyPDF2
import torch
import os



init = st.session_state



######################
# prepare embeddings #
######################

def load_EmbeddingModel(type): # type = cpu oder gpu (CUDA)
    if torch.cuda.is_available():
        init.embeddings = HuggingFaceEmbeddings(model_name=init.EMBEDDING_MODEL_NAME, model_kwargs={'device': type})
    else:
        st.info("No Cuda GPU is available. Run in CPU-Mode")
        init.embeddings = HuggingFaceEmbeddings(model_name=init.EMBEDDING_MODEL_NAME, model_kwargs={'device': "cpu"})






def prepare_Document_Type(uploaded_file_PDF=None, uploaded_file_TXT=None):

    if uploaded_file_PDF is not None:
        uploaded_text = uploaded_file_PDF
        prepare_Document(uploaded_text)
    
    if uploaded_file_TXT is not None:
        uploaded_text = uploaded_file_TXT.read().decode('utf-8')
        prepare_Document(uploaded_text)







def prepare_Document(uploaded_text):
    # Cache extracted text from the document in a temp_file.txt file
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









########################################
# create new embeddings based on file #
########################################

def handle_Datasets():

    datasets_path = 'Datasets'

    # List of folders in the datasets directory
    dataset_folders = [folder for folder in os.listdir(datasets_path) if os.path.isdir(os.path.join(datasets_path, folder))]

    # Streamlit widget for selecting the data set
    DB_option = st.selectbox(
        'Select Dataset: ',
        dataset_folders, index=init.selected_index
    )

     # Get the index of the selected option
    init.selected_index = dataset_folders.index(DB_option)

    # Load the FAISS index file based on the selected option
    if DB_option:
        index_path = os.path.join(datasets_path, f"{DB_option}")
        init.db = FAISS.load_local(index_path, init.embeddings, allow_dangerous_deserialization=True)



    init.selectedFile = DB_option
    st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)








def selectedFile():
    return init.selectedFile[:-3]







def handle_EmbeddingLLM():

    col1, col2 = st.columns(2)

    with col1:
        embeddingModell_Button = st.button("Load Embedding-Model")
    with col2:
        init.embedding_Mode = st.select_slider('Embedding-Modus', options=['CPU (RAM)', 'GPU [Cuda] (VRAM)'], label_visibility="collapsed")

    if(embeddingModell_Button):

        if init.embedding_loaded is False:

            if init.embedding_Mode == "CPU (RAM)":
                load_EmbeddingModel('cpu')
            else:
                load_EmbeddingModel('cuda')

        init.embedding_loaded = True

    if(init.embedding_loaded):
        st.success(f"""Embedding-Modell: intfloat/multilingual-e5""")
    else:
        st.error("No Embedding-Model loaded!")

    if not init.rag:
        st.info("""To enable the language model to extract information from the file, activate the "RAG" option in the settings on the left-hand side.""", icon="ℹ️")

        






def getTextfromPDF(uploaded_file, filename):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Save text from the .PDf to a .txt file
    with open(f"{filename}.txt", "w") as txt_file:
        txt_file.write(text)
    st.success(f"The extracted text was saved in '{filename}.txt'.")
    return text









#Upload new data set, convert to vectors and save in vector database
def create_Embedding_from_new_Dataset():
    st.markdown("<h3>Upload new File and save in vector database</h3>", unsafe_allow_html=True)

    border = st.container(border=True)
    with border:
        st.info("""If you want to use a large file with a lot of text, use the 'GPU [Cuda]' setting for the embedding model. This speeds up the process a lot!""", icon="ℹ️")

        init.embedding_name = st.text_input(
            "Name 👇",
            placeholder="Wikipedia",
            value="Wikipedia"
        )
        # Upload-widget to upload .txt and .pdf files
        uploaded_file = st.file_uploader("Create vector representations from the file (supported formats: .txt and .pdf)", type=["txt", "pdf"], disabled=not init.embedding_loaded, accept_multiple_files=False)

        if uploaded_file is not None:

            if st.button("Create vector representation"):

               # Extract the file name without extension
                filename = uploaded_file.name
                filename_without_extension = os.path.splitext(filename)[0]

                #Für den Fall, wenn eine .pdf Datei hochgeladen wurde
                if uploaded_file.type == 'application/pdf':

                    text = getTextfromPDF(uploaded_file, filename_without_extension)
                    prepare_Document_Type(text, None)
                    create_embeddings_From_Dokument()

                #Für den Fall, wenn eine .txt Datei hochgeladen wurde
                else:
                    prepare_Document_Type(None, uploaded_file)
                    create_embeddings_From_Dokument()






##########################
# handle with embeddings #
##########################

def get_Embedding_Text_From_Input(results):

    init.vartext = ""

    for doc in results:
        if doc[1] <= init.Euklidischer_Abstand:
            init.vartext += doc[0].page_content + "\n"
            #init.vartext = init.vartext.replace(r'\n', '\n')

    # Wenn keine Informationen aus dem Datensatz gefunden wurden
    if init.vartext.isspace() or len(init.vartext) == 0:      
        init.vartext = init.notInfo






def search_similarity_embeddings_From_Input(user_question):

    #try:
    # Suche alle relevanten Vektoren, die zum Inputvektor passen
    # results = init.db.similarity_search(user_question)                     # variante ohne Abstands-Score

    # The returned distance value is the L2 distance. Therefore a lower score is better. Best results sorted in ascending order
    results = init.db.similarity_search_with_score(user_question, k=init.topk)      #k=x -> Search for the x best results: Vector similarity + Euclidean distance
    init.results = results

    # Ergebnisse zwischenspeichen und Filtern
    get_Embedding_Text_From_Input(results)
    #except TypeError as e:
    #    st.error(e)



