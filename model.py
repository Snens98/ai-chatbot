from llama_cpp.llama_chat_format import Llava15ChatHandler
import handleEmbeddings as embeddings
from urllib.error import URLError
from llama_cpp import Llama
import helper as helper
import streamlit as st
import prompt as pr
import base64
import torch
import gc
import re
import os
import time


init = st.session_state
        
def remove_llm(enforce=False):

    if init.llm != None or enforce:
            
        init.chat_handler = None
        del init.chat_handler

        init.llm.reset()
        init.llm.set_cache(None)
        init.llm = None
        del init.llm

        st.cache_data.clear()
        st.cache_resource.clear()
        gc.collect()
        torch.cuda.empty_cache()
        init.model_loaded = False




def ckeck_mmprojFileForVision():

    parent_directory = os.path.dirname(init.model_path)
    parent_directoryElements = os.listdir(parent_directory)
    mmprojFile = [False, ""]

    for element in parent_directoryElements:
        if 'mmproj' in element:
            mmprojFile_path = os.path.join(parent_directory, element)
            st.info(f"mmproj file found! Vision function is available!")
            init.mmprojFileFound = True
            mmprojFile = [init.mmprojFileFound, mmprojFile_path]
            break
        else:
            st.info(f"mmproj file NOT found! Vision function is NOT available!")
            init.mmprojFileFound = False
            mmprojFile = [init.mmprojFileFound, None]
            break

    return mmprojFile




def saveChatForLLM_Memory(numberOfUserAssistensPairsToBeStored = 6, enabled=False):
    saveChat = """ """

    if not enabled or init.numberOfHistory <= 0:
        return ""


    if enabled:
        num_chatMessages = len(st.session_state.messages)
        num_to_save_chatMessages = min(num_chatMessages, numberOfUserAssistensPairsToBeStored)
        last_entries_chatMessages = st.session_state.messages[-num_to_save_chatMessages:]

        for entry in last_entries_chatMessages:

            if entry['role'] == 'user':
                saveChat += init.historyTemplateUSER.format(entry['content'])

            elif entry['role'] == 'assistant':
                saveChat +=  init.historyTemplateBOT.format(entry['content'])
            
        return helper.replaceBracketThatCodeCanStored(saveChat)
    
         





def isMessageListEmpt():

    if "messages" in st.session_state:
        if len(st.session_state.messages) == 0:
            return True
        return False
    return False
    




def initChatMemory():
    if "messages" not in st.session_state:
        st.session_state.messages = []



def setFirstMessage(firstMessage: str):
    initChatMemory()

    if isMessageListEmpt():
        with st.chat_message("ai"):
            st.markdown(firstMessage)




def deleteChatHistoryAtLLMChance():
    if "messages" in st.session_state:
        st.session_state.messages = []   
        




def displayChatHistory(last_answer_count = 2, deleteChatHistoryAtLLM_Change = False):

    if deleteChatHistoryAtLLM_Change:
        deleteChatHistoryAtLLMChance()

    initChatMemory()

    if len(st.session_state.messages) >= last_answer_count+2:
        chat_history = st.expander("Chat history")

        with chat_history:
            for message in st.session_state.messages[:-last_answer_count]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])    





def isModelLoaded(model):
    return model



def disableModel(model):
    model = None
    init.model_loaded = False




def initialize_chat_handler_For_VisionLLM(chat_format = "llava-1-5"):

    chat_handler = None
    chat_format = None
    isVisionModel = False

    # Check if mmproj file for vision is found
    init.mmprojFileFound, path = ckeck_mmprojFileForVision()

    if init.mmprojFileFound:
        # Initialize chat handler with Llava15ChatHandler
        init.chat_handler = Llava15ChatHandler(clip_model_path=path)
        chat_handler = init.chat_handler
        chat_format = chat_format
        isVisionModel = True

    return chat_handler, chat_format, isVisionModel






#tokenizer: BaseLlamaTokenizer | None = None,
def createModelInstance():

    chat_handler, chat_format, isVisionModel = initialize_chat_handler_For_VisionLLM("llava-1-5")
    
    try:
        init.llm = Llama(
            model_path=init.model_path,
            chat_handler=chat_handler,
            chat_format=chat_format,
            n_gpu_layers=init.gpu_layer,
            verbose=True, 
            n_batch=512,
            n_ctx=init.n_ctx,
            use_mmap=False,
            use_mlock=False,
            logits_all=isVisionModel,
        )
    except OSError:
        helper.errorMsg(error="This model is not yet supported", info="Select a different model and try again.")
        disableModel(init.llm)
        
        




def additionalInfos():
    if init.dayDateInfo:
        date, time = helper.get_current_date_and_time()
        username = os.getlogin()
        info = "Date: " + date + "\nTime: "+ time + "\nWeekday: " + helper.get_current_weekday() + "\nUsername: " + username
        infos = f"Information on the current time, date, weekday and username can be found here:\n{info}\n\n"
        return infos
    return ""




def systemPrompt(markerForRAG_Context="###Context###"):
    if isRagActive():
        system_prompt = f"""{pr.init.system_prompt}{markerForRAG_Context}\n{init.vartext}\n{markerForRAG_Context}\n\n{additionalInfos()}{saveChatForLLM_Memory(init.numberOfHistory, init.usechatMemory)}\n"""
    else:
        system_prompt = f"{pr.init.system_prompt}\n\n{additionalInfos()}{saveChatForLLM_Memory(init.numberOfHistory, init.usechatMemory)}\n"

    init.systemPrompt = system_prompt
    return system_prompt




def updatePrompt():

    try:
        init.promptError = False

        system_prompt = systemPrompt()
        user_prompt = f"{init.user_prompt}\n{init.endInstruction}" 

        init.prompt = init.template.format(SystemPrompt=system_prompt, UserPrompt=user_prompt)
        return init.prompt
    
    except (ValueError, KeyError, IndexError) as e:
        helper.dividerBr()
        helper.errorMsg(error="Remove token sequences like {, {}, {text} from System-Prompt.", info=r"Only {SytemPrompt} and {UserPrompt} are allowed!")
        with st.expander("Error:"):
            st.error(e)





def imageType():
    if init.imageUpload != None:
        return str(init.imageUpload.type)
    return ""



def create_URL_Of_Local_Image(file_path):
    if file_path:
        type = imageType()

        with open(file_path, "rb") as img_file:
            base64_data = base64.b64encode(img_file.read()).decode('utf-8')
            return f"data:{type};base64,{base64_data}"




def getMessage(System, user, data_uri=None):

    if init.imageUpload and data_uri:       # For vision models --> https://llama-cpp-python.readthedocs.io/en/latest/server/#multimodal-models
        messages = [
                    {"role": "system", "content": System},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {"type" : "text", "text": user}
                        ]
                    }
                ]
        return messages
    else:                                   # For "normal" use  --> https://llama-cpp-python.readthedocs.io/en/latest/api-reference/
        messages = [
        {"role": "system", "content": System},
            {
                "role": "user",
                "content": user
            }
        ]
    return messages








def chatCompletionVision(System, user, data_uri=None):
    updatePrompt()
    return init.llm.create_chat_completion(
        stream=True,  
        temperature=init.temperature,
        max_tokens=init.max_Token,
        top_p=init.top_p,
        min_p=init.min_p,
        top_k=init.top_k,
        repeat_penalty=init.repeat_penalty,
        messages = getMessage(System, user, data_uri),
    )


def get_Language_Model():

    return init.llm.create_completion(
        
        prompt=updatePrompt(),
        stream=True, 
        temperature=init.temperature,
        max_tokens=init.max_Token,
        top_p=init.top_p,
        min_p=init.min_p,
        top_k=init.top_k,
        repeat_penalty=init.repeat_penalty,
        stop=["<|eot_id|>assistant", "<|eot_id|>", "<|im_end|>", "</s>", "<end_of_turn>", "<|END_OF_TURN_TOKEN|>"]
    )




def isVisionActive():
    return (init.imageUpload and init.vision)



def get_Vision_Model():

    data_uri = create_URL_Of_Local_Image(init.file_path)

    if isVisionActive():
        try: 
            return chatCompletionVision(
                System=init.systemPrompt,
                user=f"{init.user_prompt}\n{init.endInstruction}",
                data_uri=data_uri)

        except (URLError, OSError):
            helper.errorMsg(error="The image is corrupted or has an unsupported format. It cannot be analyzed", info="Upload a other Image (.PNG or .JPG) and try agan!")
            updatePrompt()
            return chatCompletionVision(
                System=f"{init.systemPrompt} Important! If the user tell questions about images or pictures let the user know that you have not received working / supported picture!", 
                user=f"{init.user_prompt}\n{init.endInstruction}")

    else:
        return chatCompletionVision(
            System=f"{init.systemPrompt} Important! Let the user know that you have not received a picture!", 
            user=f"{init.user_prompt}\n{init.endInstruction}" )






def iterateThroughVisionOutput(output):

    value = ""
    if 'choices' in output:
        
        choices = (output['choices'])     
        delta = choices[0]['delta']

        if 'content' in delta:
            value = (delta['content'])

            if init.promptError:
                return
            return value
    return value
        


def iterateThroughLLMOutput(output):
                      
    content_value = output['choices']
    value = content_value[0]['text']
    return value



def tooManyLineBreaks(text, numberOfLineBreaks):
    if text.count('\n') > numberOfLineBreaks: #Prevents language models from generating uncontrolled line breaks
        return True
    

def handle_n_ctx_Errors(_error):
    filterInvalidCTXNumber = re.search(r'\((\d+)\)', str(_error))
    if filterInvalidCTXNumber:
        number = filterInvalidCTXNumber.group(1)
    st.error(f"n_ctx must be set to at least {number}")



def displayCurrentToken(output_container, currentResult, Icon="🔘"):
    output_container.markdown(currentResult + Icon)




def displayFinishedOutput(cancelBtn, output_container, result, completedSymbol="🟢"):
    cancelBtn.empty()
    output_container.markdown(result + completedSymbol)
    st.session_state.messages.append({"role": "assistant", "content": init.fullResponse})






def getModel():

    if init.mmprojFileFound:

        with st.spinner("Analyze the image..."):
            model = get_Vision_Model()
    else:
        with st.spinner("Analyze inputs..."):
            model = get_Language_Model()
    return model




def displayCancelButton():

    button_container = st.empty()
    cancel_button = True

    if cancel_button == True:
        button_container.button("Cancel", type="primary")
    return button_container




def haveChatTemplateInvaledChars():
    updatePrompt()
    if init.promptError:
        if 'message' in st.session_state:
            if st.session_state.message:
                st.session_state.message.pop()
        return True
    return False



def displayUserInput(user_prompt):

    with st.chat_message("user"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})   
        st.markdown(user_prompt)




def displayModelResponse():

    with st.chat_message("ai"):

        output_container = st.empty()
        button_container = displayCancelButton()
        token = ""

        with st.spinner("Generate response"):
            
            try:
                for output in getModel():

                    if init.mmprojFileFound:
                        token += iterateThroughVisionOutput(output)   
                    else:
                        token += iterateThroughLLMOutput(output)   

                    if tooManyLineBreaks(token, 500):
                        break
                
                    init.fullResponse = token
                    displayCurrentToken(output_container, token, Icon="🔘")

            except ValueError as error:
                handle_n_ctx_Errors(error)

            displayFinishedOutput(button_container, output_container, token, completedSymbol="🟢")







def isRagPipelineActive():
    if init.rag and init.embedding_loaded and init.file:
        return True
    return False



def process_user_prompt(user_prompt):

    if haveChatTemplateInvaledChars():
        return
    
    if isRagPipelineActive():
        embeddings.find_relevant_context(user_prompt)

    displayUserInput(user_prompt)
    displayModelResponse()





def isRagActive()->bool:
    if init.rag:
        return True
    return False






# Displays the chat history in the browser window by iterating through the saved messages and displaying them in the chat window according to their type (user or assistant).
def displayLastChatMessages(numberOfInputResponsePairs=2):
    for message in st.session_state.messages[-numberOfInputResponsePairs:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
