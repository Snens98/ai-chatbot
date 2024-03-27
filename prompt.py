import streamlit as st

init = st.session_state



# Mapping der Optionen zu den Modellinformationen
model_options = {

    "phi-2.Q6 [3.6 GB]": {
        "index": 1,
        "repo_id": "TheBloke/phi-2-GGUF",
        "model_file_name": "phi-2.Q6_K.gguf",
        "template": """Instruct: {}\n{}\nOutput:"""
    },
    "dolphin-2.2.1-mistral-7b.Q6 [7.4 GB]": {
        "index": 2,
        "repo_id": "TheBloke/dolphin-2.2.1-mistral-7B-GGUF",
        "model_file_name": "dolphin-2.2.1-mistral-7b.Q6_K.gguf",
        "template": """<|im_start|>system\n{}<|im_end|>\n<|im_start|>user{}<|im_end|>\n<|im_start|>assistant"""
    },
    "mistral-7b-instruct-v0.1.Q6 [7.4 GB]": {
        "index": 3,
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        "model_file_name": "mistral-7b-instruct-v0.1.Q6_K.gguf"
    },
    "neural-chat-7b-v3-3.Q6 [7.4 GB]": {
        "index": 4,
        "repo_id": "TheBloke/neural-chat-7B-v3-3-GGUF",
        "model_file_name": "neural-chat-7b-v3-3.Q6_k.gguf"
    },
    "solar-10.7b-instruct-v1.0.Q6 [11.31 GB]": {
        "index": 5,
        "repo_id": "TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF",
        "model_file_name": "solar-10.7b-instruct-v1.0.Q6_K.gguf"
    },
    "orca-2-13b.Q6 [13.2 GB]": {
        "index": 6,
        "repo_id": "TheBloke/Orca-2-13B-GGUF",
        "model_file_name": "orca-2-13b.Q6_K.gguf"
    },
    "llama-2-13b.Q6 [13.2 GB]": {
        "index": 7,
        "repo_id": "TheBloke/Llama-2-13B-GGUF",
        "model_file_name": "llama-2-13b.Q6_K.gguf"
    },
    "airoboros-l2-13b-3.1.1.Q6 [13.2 GB]": {
        "index": 8,
        "repo_id": "TheBloke/Airoboros-L2-13B-3.1.1-GGUF",
        "model_file_name": "airoboros-l2-13b-3.1.1.Q6_K.gguf"
    },
    "mixtral-8x7b-instruct-v0.1.Q6 [41.0 GB]": {
        "index": 9,
        "repo_id": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
        "model_file_name": "mixtral-8x7b-instruct-v0.1.Q6_K.gguf"
    },
    "sauerkrautlm-7b-hero.Q6 [7.4 GB]": {
        "index": 10,
        "repo_id": "TheBloke/SauerkrautLM-7B-HerO-GGUF",
        "model_file_name": "sauerkrautlm-7b-hero.Q6_K.gguf"
    },
    "sauerkrautlm-solar-instruct.Q6 [11.31 GB]": {
        "index": 11,
        "repo_id": "TheBloke/SauerkrautLM-SOLAR-Instruct-GGUF",
        "model_file_name": "sauerkrautlm-solar-instruct.Q6_k.gguf"
    },
    "sauerkrautlm-13b-v1.Q6 [13.2 GB]": {
        "index": 12,
        "repo_id": "TheBloke/SauerkrautLM-13B-v1-GGUF",
        "model_file_name": "sauerkrautlm-13b-v1.Q6_K.gguf"
    },
    "em_german_leo_mistral.Q6 [7.4 GB]": {
        "index": 13,
        "repo_id": "TheBloke/em_german_leo_mistral-GGUF",
        "model_file_name": "em_german_leo_mistral.Q6_K.gguf"
    },
    "em_german_13b_v01.Q6 [13.2 GB]": {
        "index": 14,
        "repo_id": "TheBloke/em_german_13b_v01-GGUF",
        "model_file_name": "em_german_13b_v01.Q6_K.gguf"
    },
    "discolm_german_7b_v1.Q6 [7.4 GB]": {
        "index": 15,
        "repo_id": "TheBloke/DiscoLM_German_7b_v1-GGUF",
        "model_file_name": "discolm_german_7b_v1.Q6_K.gguf"
    }
}




promptText_EN = """The following instructions are important and must be followed:
You are a helpful assistant. Your name is Diesel-Sören. Your task is to answer questions about Hamm-Lippstadt University of Applied Sciences and studying in general.
To answer the following question, only use the information between ###Context###. 
Answer the question clearly and briefly. Think step by step.
If you cannot answer the question with the information between ###Context###, answer with: "I have no information about this. Please visit the website: https://www.hshl.de".
only write links that are in the information between ###Context###.
Only answer questions about HSHL and the study program, do not answer any other questions!
If the information in the question does not match the information between ###Context###, correct the wrong information!"""



promptText = """Die Folgenden Anweisungen sind wichtig und müssen eingehalten werden:
Du bist ein hilfreicher Assistent. Dein Name ist Diesel-Sören. Deine Aufgabe ist es, Fragen zur Hochschule Hamm-Lippstadt und zum Studium im allgemeinen zu beantworten.
Um die folgende Frage zu beantworten, verwende nur die Informationen zwischen ###Context###. 
Beantworte die Frage klar und kurz. Denke Schritt für Schritt.
Wenn du die Frage mit den Informationen zwischen ###Context### nicht beantworten kannst, antworte mit: "Ich habe keine Informationen darüber. Bitte besuchen Sie die Website: https://www.hshl.de".
schreibe nur links die in den Informationen zwischen ###Context### stehen.
Beantworte nur Fragen zur HSHL und zum Studium, beantworte keine anderen Fragen!
Wenn die Informationen in der Frage nicht mit den Informationen zwischen ###Context### übereinstimmen, korrigiere die falschen Informationen!
Spreche Deutsch."""




end_Instruction="Antworte auf deutsch und kurz!"
end_Instruction_EN ="Answer precisely and briefly!"





nonRAG_Prompt_EN = "You are a helpful assistant called Diesel-Sören. Answer briefly and clearly. Reply with emojis too. Always in a good mood!"
nonRAG_Prompt = """Du bist ein hilfreicher Assistent namens Diesel-Sören. Antworte kurz und klar. Spreche Deutsch."""
