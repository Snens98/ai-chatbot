import streamlit as st
init = st.session_state



def initVars():

    # Prompt-Engineering
    init.model = st.session_state.get('model', 0)
    init.template = st.session_state.get('template', "")
    init.vartext = st.session_state.get('vartext', " ")
    init.notInfo = st.session_state.get('notInfo', "")

    init.endInstruction = st.session_state.get('endInstruction', "")
    init.prompt = st.session_state.get('prompt', "")
    init.system_prompt = st.session_state.get('system_prompt', " ")
    init.systemPrompt = st.session_state.get('systemPrompt', "")
    init.promptError = st.session_state.get('promptError', False)

    init.historyTemplateUSER = st.session_state.get('historyTemplateUSER', "")
    init.historyTemplateBOT = st.session_state.get('historyTemplateBOT', "")






    #RAG
    init.db = st.session_state.get('db', None) 
    init.file = st.session_state.get('file', True) 
    init.rag = st.session_state.get('rag', False)
    init.results = st.session_state.get('results', "")
    init.RAGTopic = st.session_state.get('RAGTopic', " ")
    init.RAGFilePath = st.session_state.get('RAGFilePath', " ")





    #Embedding
    init.embedding_name = st.session_state.get('embedding_name', " ")
    init.embedding_loaded = st.session_state.get('embedding_loaded', False)
    init.embedding_Mode = st.session_state.get('embedding_Mode', "")
    init.selectedFile = st.session_state.get('selectedFile', "")
    init.embeddings = st.session_state.get('embeddings', None)

    #Embedding-Modelle
    init.EMBEDDING_MODEL_NAME = st.session_state.get('EMBEDDING_MODEL_NAME', "intfloat/multilingual-e5-large")  # 2.4 GB of RAM
    #init.EMBEDDING_MODEL_NAME = st.session_state.get('EMBEDDING_MODEL_NAME', "BAAI/bge-large-en-v1.5")  # 1.4 GB of RAM
    
    # Embedding-LLM config
    init.topk = st.session_state.get('topk', 4)
    init.chunk_size = st.session_state.get('chunk_size', 600)
    init.overlap = st.session_state.get('overlap', 50)
    init.Euklidischer_Abstand = st.session_state.get('Euklidischer_Abstand', 3.5)






    # Vision LLM
    init.imageUpload = st.session_state.get('imageUpload', None)
    init.file_path = st.session_state.get('file_path', "")
    init.vision = st.session_state.get('vision', False)
    init.mmprojFileFound = st.session_state.get('mmprojFileFound', False)
    init.chat_handler = st.session_state.get('chat_handler', None)




    # LLM-Location
    init.repo_id = st.session_state.get('repo_id', 'TheBloke/SauerkrautLM-3B-v1-GGUF')
    init.model_file_name = st.session_state.get('model_file_name', 'sauerkrautlm-3b-v1.Q8_0.gguf')
    init.model_path = st.session_state.get('model_path', None)

    #LLM Handle
    init.model_loaded = st.session_state.get('model_loaded', False)
    init.user_prompt = st.session_state.get('user_prompt', "")
    init.fullResponse = st.session_state.get('fullResponse', " ")
    init.llm = st.session_state.get('llm', None)
    init.model_updated = st.session_state.get('model_updated', False)
    init.download_Model = st.session_state.get('download_Model', False)

    # LLM config
    init.gpu_layer = st.session_state.get('gpu_layer', 20)
    init.temperature = st.session_state.get('temperature', 0.25)
    init.max_Token = st.session_state.get('max_Token', 512)
    init.top_p = st.session_state.get('top_p', 0.8)
    init.min_p = st.session_state.get('min_p', 0.05)
    init.top_k = st.session_state.get('top_k:', 30)
    init.repeat_penalty = st.session_state.get('repeat_penalty:', 1.4)




    # Other Setttings
    init.dayDateInfo = st.session_state.get('dayDateInfo', False)
    init.numberOfHistory = st.session_state.get('numberOfHistory', 7)
    init.writeInDocs = st.session_state.get('writeInDocs', False)
    init.usechatMemory = st.session_state.get('usechatMemory', "") 





    init.memUsageThread = st.session_state.get('memUsageThread', False)
    init.downloadStart = st.session_state.get('downloadStart', False)