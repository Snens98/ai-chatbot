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
    elements = os.listdir(parent_directory)
    result = [False, ""]

    for element in elements:
        if 'mmproj' in element:
            mmprojFile = os.path.join(parent_directory, element)
            st.info(f"mmproj file found! Vision function is available!")
            path = mmprojFile
            init.mmprojFileFound = True
            result = [init.mmprojFileFound, path]
            break
        else:
            st.info(f"mmproj file NOT found! Vision function is NOT available!")
            init.mmprojFileFound = False
            result = [init.mmprojFileFound, None]
            break

    return result









# Create a language model instance (LLM) with specific configuration parameters.
# If there is an unsupported model, an error message is displayed and the LLM instance is set to None, 
# indicating that the model is not loaded, and a flag is updated accordingly.
def create_llm():

    init.mmprojFileFound, path = ckeck_mmprojFileForVision()

    try:
        if init.mmprojFileFound:

            init.chat_handler = Llava15ChatHandler(clip_model_path=path)

            init.llm = Llama(
                model_path=init.model_path,
                chat_handler=init.chat_handler,
                chat_format="llava-1-5",
                n_gpu_layers=init.gpu_layer,
                verbose=True, 
                n_batch=512,
                n_ctx=init.n_ctx,
                use_mmap=False,
                use_mlock=False,
                logits_all=True) # needed to make llava work
        else:
            
            init.llm = Llama(
                model_path=init.model_path,
                n_gpu_layers=init.gpu_layer,
                verbose=True, 
                n_batch=512,
                n_ctx=init.n_ctx,
                use_mmap=False,
                use_mlock=False,                
                ) 
            
    except OSError:
        helper.errorMsg(error="This model is not yet supported", info="Select a different model and try again.")
        init.llm = None
        init.model_loaded = False





def day_and_time():
    date, time = helper.get_current_date_and_time()
    info = "Date: " + date + "\nTime: "+ time + "\nWeekday: " + helper.get_current_weekday()
    dayAndTimeInfo = f"Information on the current time, date and weekday can be found here: {info}\n"
    return dayAndTimeInfo





def systemPrompt(marker="###Context###"):
    if isRagActive():
        system_prompt = f"{pr.init.system_prompt}\n\n{marker}\n{init.vartext}\n{marker}\n\n{day_and_time()}\n\n{init.saveChat}\n\n"
    else:
        system_prompt = f"{pr.init.system_prompt}\n\n{day_and_time()}\n\n{init.saveChat}\n\n"

    init.systemPrompt = system_prompt
    return system_prompt





# To update the prompt used for the language model based on various parameters and configurations. 
# It ensures that the prompt is formatted correctly and contains relevant information. 
# It constructs the system prompt by combining system-related information, user prompts, and context if RAG is enabled. 
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

        init.prompt = "Hello!"
        init.promptError = True
        return init.prompt






# Get type of Image
def imageType():
    if init.imageUpload != None:
        return str(init.imageUpload.type)
    return ""




# Create url of local image
def image_to_base64_data_uri(file_path):

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




def get_Vision_Model():

    file_path = init.file_path
    data_uri = image_to_base64_data_uri(file_path)

    if init.imageUpload and init.vision:
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










# The function retrieves the language model instance.
def get_Language_Model():

    return init.llm.create_completion(
        
        prompt=updatePrompt(),
        stream=True, 
        temperature=init.temperature,
        max_tokens=init.max_Token,
        top_p=init.top_p,
        min_p=init.min_p,
        top_k=init.top_k,
        repeat_penalty=init.repeat_penalty
    )







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




# Process the user input and generate a response from the language model. It creates a user interface for the chat with an input field for the user
# input and an output field for the generated response. Within the function, the language model is retrieved using the get_model() function. 
# This function ensures the interaction between the user and the language model.
def process_user_prompt(user_prompt):

    try:
        if isRagPipelineActive():
            embeddings.search_similarity_embeddings_From_Input(user_prompt)  # Embeddings are created based on the input and similarities are searched for  
    except AssertionError:
        helper.errorMsg(error="The selected file does not match the current embedding model", info="Select the correct file or change it to the appropriate embedding model")


    # Display User-Imput
    with st.chat_message("user"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})   
        st.markdown(user_prompt)


    # Display LLM-Response
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



def isRagActive()->bool:
    if init.rag:
        return True
    return False






# Displays the chat history in the browser window by iterating through the saved messages and displaying them in the chat window according to their type (user or assistant).
def displayLastChatMessages(numberOfInputResponsePairs=2):
    for message in st.session_state.messages[-numberOfInputResponsePairs:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
