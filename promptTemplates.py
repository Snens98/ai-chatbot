import streamlit as st


init = st.session_state




def getTemplateList():
    list = [" Custom ", "ChatLM", "Mistral-Instruct", "Orca-Hashes", "Llama2-base", "Llama2-Chat", "Mixtral-Instruct", "User-Assistant", "User-Assistant2", "Sauerkraut", "EmGerman", "OpenChat", "OpenChat2", "DeepSeek", "CodeLlama", "Orca-Vicuna", "Command-r", "Zephyr", "Phi", "Gemma-it"]
    return list



def getTemplate(selected_option, get_template):



    if selected_option == " Custom ":
        init.template = st.text_area(" ",get_template, height=400)
    elif selected_option == "ChatLM":
        init.template = st.text_area(" ", "<|im_start|>system\n{SystemPrompt}<|im_end|>\n<|im_start|>user{UserPrompt}<|im_end|>\n<|im_start|>assistant", height=400)
        init.historyTemplateUSER = "<|im_start|>user{}<|im_end|>"
        init.historyTemplateBOT = "<|im_start|>assistant{}<|im_end|>"

    elif selected_option == "Mistral-Instruct":
        init.template = st.text_area(" ", "<s>[INST] {SystemPrompt}{UserPrompt}</s> [/INST]", height=400)
        init.historyTemplateUSER = "<s>[INST]{}</s> [/INST]"
        init.historyTemplateBOT = "<s>[INST]{}</s> [/INST]"

    elif selected_option == "Orca-Hashes":
        init.template = st.text_area(" ", "### System:\n{SystemPrompt}\n### User:\n{UserPrompt}\n### Assistant:", height=400)
        init.historyTemplateUSER = "### User:{}"
        init.historyTemplateBOT = "### Assistant:{}"

    elif selected_option == "Llama2-base":
        init.template = st.text_area(" ", "{SystemPrompt}\n{UserPrompt}", height=400)
        init.historyTemplateUSER = "<|im_start|>user{}<|im_end|>"
        init.historyTemplateBOT = "<|im_start|>assistant{}<|im_end|>"

    elif selected_option == "Llama2-Chat":
        init.template = st.text_area(" ", "[INST] <<SYS>>\n{SystemPrompt}\n<</SYS>>\n{UserPrompt}[/INST]", height=400)
        init.historyTemplateUSER = "<|im_start|>user{}<|im_end|>"
        init.historyTemplateBOT = "<|im_start|>assistant{}<|im_end|>"
        
    elif selected_option == "Mixtral-Instruct":
        init.template = st.text_area(" ", "[INST] {SystemPrompt}\n{UserPrompt} [/INST]", height=400)
    elif selected_option == "User-Assistant":
        init.template = st.text_area(" ", "{SystemPrompt}\n### User:\n{UserPrompt}\n### Assistant:", height=400)
    elif selected_option == "Sauerkraut":
        init.template = st.text_area(" ", "{SystemPrompt}\nUser: {UserPrompt}\nAssistant:", height=400)
    elif selected_option == "EmGerman":
        init.template = st.text_area(" ", "{SystemPrompt} USER: {UserPrompt} ASSISTANT:", height=400)
    elif selected_option == "OpenChat":
        init.template = st.text_area(" ", "{SystemPrompt} GPT4 User: {UserPrompt}<|end_of_turn|>GPT4 Assistant:", height=400)
    elif selected_option == "User-Assistant2":
        init.template = st.text_area(" ", "{SystemPrompt} USER: {UserPrompt}\nASSISTANT:", height=400)
    elif selected_option == "OpenChat2":
        init.template = st.text_area(" ", "{SystemPrompt} GPT4 Correct User: {UserPrompt}<|end_of_turn|>GPT4 Correct Assistant:", height=400)
    elif selected_option == "DeepSeek":
        init.template = st.text_area(" ", "{SystemPrompt}\n### Instruction:\n{UserPrompt}\n### Response:", height=400)
    elif selected_option == "CodeLlama":
        init.template = st.text_area(" ", "[INST] {SystemPrompt}\n{UserPrompt}\n[/INST]", height=400)
    elif selected_option == "Orca-Vicuna":
        init.template = st.text_area(" ", "SYSTEM: {SystemPrompt}\nUSER: {UserPrompt}\nASSISTANT:", height=400)
    elif selected_option == "Command-r":
        init.template = st.text_area(" ", "{SystemPrompt}## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{UserPrompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>", height=400)
    elif selected_option == "Zephyr":
        init.template = st.text_area(" ", "<|system|>{SystemPrompt}\n</s>\n<|user|>\n{UserPrompt}</s>\n<|assistant|>", height=400)
    elif selected_option == "Phi":
        init.template = st.text_area(" ", "{SystemPrompt}Instruct: {UserPrompt}\nOutput:", height=400)
    elif selected_option == "Gemma-it":
        init.template = st.text_area(" ", "{SystemPrompt}<bos><start_of_turn>user\n{UserPrompt}<end_of_turn>\n<start_of_turn>model", height=400)
    return init.template

