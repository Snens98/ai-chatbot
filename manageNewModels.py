from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub import hf_hub_download
from huggingface_hub import hf_hub_url
from huggingface_hub import HfApi
import concurrent.futures
import streamlit as st
import send2trash
import functools
import requests
import helper
import json
import time
import os



init = st.session_state
filename = 'model_options.json'



def get_model_url(model, sibling):
    return f"https://huggingface.co/{model.id}/resolve/main/{sibling.rfilename}"


def write_model_options_to_file(model_options, filename):
    with open(filename, 'w') as file:
        json.dump(model_options, file, indent=4)


def update_llm_Json(json_file_path, data):
    write_model_options_to_file(data, json_file_path)
    st.experimental_rerun()


def delete_llm_from_Json(data, model_file_name):
    for model_name, model_info in list(data.items()):
        if model_info["model_file_name"] == model_file_name:
            del data[model_name]


def getURL(model):                    
    url = f"https://huggingface.co/{model.id}"
    return url 




def display_model_expander(model):

    st.markdown(
        f"""
        <style>
        .button {{
            color: white !important;
            border: 1px solid grey;
            color: white;
            padding: 5px 10px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 14px; 
            margin: 0;
            cursor: pointer;
            border-radius: 10px;
            border-radius: 8px;
        }}
        </style>
        """
        , unsafe_allow_html=True
    )
    url = getURL(model)

    col1, col2, col3 = st.columns([4, 1, 1])     # quantization variant and download/delete
    with col1:
        st.markdown(f'<a href="{url}" class="button">Link to language model</a>', unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div title=" " style='
                background-color: #12161e;
                text-align: center;
                padding: 5px;
                border-radius: 10px;
                height: 30px; 
                display: flex;
                align-items: center; 
            '>
                <p style='font-size: 12px; color: #fafafa; margin: 0;'>Required (V)RAM</p>
            </div>
        """, unsafe_allow_html=True)





def model_search(searchtext, formatsupport, numberOFSearch):
# Possible Sort: downloads, likes, last_modified

    try:
        api = HfApi()
        searchtext += formatsupport # only GGUF-Format support
        models = api.list_models(direction=-1, sort= "likes", filter=[], search=searchtext, limit=numberOFSearch, token=False) # Filter models
        return models

    except requests.exceptions.RequestException as e:
        st.error("No internet connection. Please check your network connection and try again.")    
        with st.expander("Error:"):
            st.error(e)
            return
    except Exception as e:
        return
    



def getModelFile(file):
    modelFile = f"/resolve/main/{file.rfilename}"
    return modelFile



def ShowSizeOfFiles(text, color):
    st.markdown(f'<p style="font-size: 14px; color: {color}; background-color: #1a1c24; line-height: 280%; border-radius: 10px; padding-left: 25px;">{text}</p>', unsafe_allow_html=True)





def show_language_model_variants(variants):
    if "mmproj" in variants.rfilename: # mmproj is a vision adapter for vision models. mmproj must also be downloaded
        mmproj = "👁️ "
        tooltip_text = "ℹ One of the 'mmproj' files is required for the vision function"
    else:
        mmproj = ""
        tooltip_text = ""

    css_style = f'display: flex; align-items: center; font-size: 14px; color: #fafafa; background-color: #1a1c24; line-height: 280%; border-radius: 10px; padding-right: 25px;'
    element_html = f'<p title="{tooltip_text}" style="{css_style}"><span style="margin-left: 10px;">{variants.rfilename}</span><span style="margin-left: auto;"><span class="tooltip">{mmproj}<span class="tooltiptext"></span></span></span></p>'
    st.markdown(element_html, unsafe_allow_html=True)

    



def read_model_options_from_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data



def getModelDeletePath(model, sibling):
    deletePath = hf_hub_download(repo_id=model.id, filename=sibling.rfilename, repo_type='model')
    parentDirectory = os.path.dirname(os.path.dirname(os.path.dirname(deletePath)))
    return parentDirectory



def updateIndex(file='model_options.json', updateIndex=True):

    data = read_model_options_from_file(file)

    index = 1
    for key in data.keys():
        data[key]['index'] = index
        index += 1

    if updateIndex:
        write_model_options_to_file(data, file)





def add_new_model(name, repo_id, model_file_name, promptTemplate):
    updateIndex(filename)
    options = read_model_options_from_file(filename)
    num_elements = len(options)
    index = num_elements + 1
    options[name] = {
        'index': index,
        'repo_id': repo_id,
        'model_file_name': model_file_name,
        'template': promptTemplate,
        "download": True,  
        "recommended": False
    }
    write_model_options_to_file(options, 'model_options.json')







# Handles the actions that are performed when the "Delete" or "Download" button is clicked
def handle_button_action(model, sibling, delete_btn, download_btn, filename):

    if delete_btn:
        send2trash.send2trash(getModelDeletePath(model, sibling))
        st.info(f"'{filename}' has been removed! It has been moved to the Recycle Bin", icon='ℹ️')
        data = read_model_options_from_file(filename)
        delete_llm_from_Json(data, sibling.rfilename) 
        update_llm_Json(filename, data)

    if download_btn:
        try:
            init.downloadStart = True
            downloadlink = hf_hub_url(repo_id=model.id, filename=sibling.rfilename)
            st.markdown("download-Link:")
            st.code(f"{downloadlink}")
            helper.br()

            with st.spinner(f"Download '{sibling.rfilename}'"):
                init.model_path = hf_hub_download(repo_id=model.id, filename=sibling.rfilename, repo_type='model')
                st.success("Download was successful!")
                add_new_model(sibling.rfilename, model.id, sibling.rfilename, """<|im_start|>system\n{SystemPrompt}<|im_end|>\n<|im_start|>user\n{UserPrompt}<|im_end|>\n<|im_start|>assistant""")
                st.experimental_rerun()

        except requests.exceptions.ConnectionError as e:
            st.error("No internet connection. Please check your network connection and try again.")
            with st.expander("Error:"):
                st.error(e)
        except Exception as e:
            st.info("You must be authenticated to access it on hf.co.")



def isModelAvailable(sibling):
    return any(sibling.rfilename == model_info["model_file_name"] for model_info in read_model_options_from_file(filename).values())



@functools.lru_cache(maxsize=None)
def fetch_metadata(url):
    try:
        metadata = get_hf_file_metadata(url)
    except Exception as e:
        metadata = None

    return metadata




@functools.lru_cache(maxsize=None)
def fetch_metadata_Size(url):

    try:
        metadata = get_hf_file_metadata(url)
        metadata_Model_size = round((metadata.size / 1e+9)+1, 2)  # Bytes to GB

    except Exception as e:
        metadata_Model_size = 0.0

    metadata_Model_size = round(metadata_Model_size, 2)
    return metadata_Model_size




@functools.lru_cache(maxsize=None)
def query_file_size_parallel(models):

    resultSize = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        urls = []
        for model in models:
            for sibling in model.siblings:
                if sibling.rfilename.endswith('.gguf'):
                    url = get_model_url(model, sibling)
                    urls.append(url)

        results = executor.map(fetch_metadata_Size, urls, chunksize = 8, timeout=10)

        for gbStr in results:
            resultSize.append(gbStr)

    return resultSize



def clicked_Search_BTN():
    init.search = True



def model_interaction_interface(models, progress, no_results, iterate_through_GGUFSize, GGUFFileSize = [], max_ram = 32, max_vram= 32):
    for model in models:

        modelIDs = st.expander(f"{model.id}")
        with modelIDs:
            display_model_expander(model)

            for sibling in model.siblings:

                progressBar = iterate_through_GGUFSize+20
                if progressBar > 100:
                    progressBar = 100
                progress.progress(progressBar, text="List models...")

                download_btn = None
                delete_btn = None

                if not sibling.rfilename.endswith('.gguf'): # only GGUF-Format support
                    continue

                if sibling.rfilename.endswith('.gguf'):
                    col1_language_model_variants, col2_isModelAvailable, col3_ShowSizeOfFiles = st.columns([4, 1, 1])     # quantization variant and download/delete

                with col1_language_model_variants:
                    show_language_model_variants(sibling)

                with col2_isModelAvailable:
                    if isModelAvailable(sibling):
                        delete_btn = st.button(" Delete ", key=f"delete_{model.id}_{sibling.rfilename}", type="primary", help=f"Remove '{sibling.rfilename}' completely from disk", on_click = clicked_Search_BTN)
                    else:
                        download_btn = st.button("Download", key=f"delete_{model.id}_{sibling.rfilename}", help=f"Download language model from hf.co", on_click = clicked_Search_BTN)

                with col3_ShowSizeOfFiles:
                    if (not delete_btn or not download_btn) and GGUFFileSize != []:
                        ShowSizeOfFiles(str(GGUFFileSize[iterate_through_GGUFSize])+" GB", helper.get_color_based_on_memory(GGUFFileSize[iterate_through_GGUFSize], max_ram=max_ram, max_vram=max_vram))
                    iterate_through_GGUFSize += 1

                if delete_btn or download_btn:
                    init.download = True
                    handle_button_action(model, sibling, delete_btn, download_btn, filename)

    if iterate_through_GGUFSize == 0:
        no_results.info("No results!", icon="🔎")
    



def startSearchBtn(key):
    st.markdown("""
        <style>
            .stButton>button {

                height: 10px; !important;
                transform: translateY(+35%);
            }
        </style>
    """, unsafe_allow_html=True)

    startSearch = st.button(" 🔎 Start search ", key=key)
    return startSearch





def getAPI(url_API):
    try:
        url = url_API
        response = requests.get(url)
        return response
    except Exception as e:
        print("Connection error:", e)
        return



def trendingModels(maximalModels = 20):

    response = getAPI("https://huggingface.co/api/trending")

    if response.status_code == 200:

        data = response.json()         # Extract the JSON data of the response
        recently_trending = data.get("recentlyTrending", [])
        numberOfTendingModelsDisplayed  = 0

        for model in recently_trending:
            if numberOfTendingModelsDisplayed >= maximalModels:
                break  
            if model.get("repoType") == "model":
                repo_data = model.get("repoData", {})
                model_id = repo_data.get("id")
                fullname = repo_data.get("authorData", {}).get("fullname")

                if model_id:
                    url = f" {numberOfTendingModelsDisplayed+1}. {fullname} | {model_id} "
                    create_language_model_Link_Button(model_id, url)
                    numberOfTendingModelsDisplayed += 1
    else:
        st.info("Error retrieving the data. Status code: ")
        st.info(response.status_code)










def searchModelsAndRelatedQuants(NumberOfSearchResults = 25):

    iterate_through_GGUFSize = 0
    GGUFFileSize = []

    with st.spinner("Search models... 🔎"):
        
        progress = st.empty()

        with st.container(border=True):
            col1_searchtext, col2, col3_startSearch = st.columns([1.0, 0.1, 0.3])

            with col1_searchtext:
                searchtext = st.text_input(label="Search language Models ", placeholder="🔎")
            with col3_startSearch: 
                startSearch = startSearchBtn(key="startSearch")

        if startSearch or init.search:
            
            if searchtext == '' or searchtext is None:
                return
            
            models = model_search(searchtext=searchtext, formatsupport = " gguf", numberOFSearch=NumberOfSearchResults)
            progress.progress(2, text="Search GGUF quants...")

            try:
                if not init.search:
                    GGUFFileSize = query_file_size_parallel(models) # Create list of GGUFFileSize from models found

                progress.progress(20, text="Search GGUF quants...")

                max_ram = helper.get_max_memory()
                max_vram = helper.get_max_vram()
                no_results = st.empty()         # Display if no language model was found
                models = model_search(searchtext, " gguf", NumberOfSearchResults) 

                model_interaction_interface(models, progress, no_results, iterate_through_GGUFSize, GGUFFileSize, max_ram, max_vram)

            except TimeoutError as e:
                st.info("The search took too long and was aborted!")

            progress.progress(100, text="Complete!")
            time.sleep(1)
            progress.empty()

    init.search = False







def show_language_model_variants(variants):
    if "mmproj" in variants.rfilename:
        mmproj = "👁️ "
        tooltip_text = "ℹ One of the 'mmproj' files is required for the vision function"
    else:
        mmproj = ""
        tooltip_text = ""

    css_style = f'display: flex; align-items: center; font-size: 14px; color: #fafafa; background-color: #1a1c24; line-height: 280%; border-radius: 10px; padding-right: 25px;'
    element_html = f'<p title="{tooltip_text}" style="{css_style}"><span style="margin-left: 10px;">{variants.rfilename}</span><span style="margin-left: auto;"><span class="tooltip">{mmproj}<span class="tooltiptext"></span></span></span></p>'
    st.markdown(element_html, unsafe_allow_html=True)

    




    






def create_language_model_Link_Button(model_id, model_file_name):
    
    if "mmproj" in model_file_name:
        mmproj = "👁️ "
        tooltip_text = "ℹ This is a vision-adapter for vision funktion"
    else:
        mmproj = ""
        tooltip_text = ""

    link_url = f"https://huggingface.co/{model_id}"
    css_style = f'''
        display: flex;
        align-items: center;
        font-size: 14px;
        color: #b4b7be;
        background-color: #1a1c24;
        line-height: 275%;
        border-radius: 10px;
        padding-right: 25px;
        padding-left: 25px;
    '''
    
    html_content = f'''
    <style>
        .hover-blue:hover {{
            color: #fafafa !important;
           /* text-decoration: underline !important; */
            background-color: #131720 !important; 

        }}
    </style>

     <p class="hover-blue" title="{tooltip_text}" style="{css_style}">
        <a href="{link_url}" target="_blank" rel="noopener noreferrer" style="text-decoration: none; color: inherit; margin-right: 20px;">{model_file_name}</a>
        <span style="margin-left: auto;">{mmproj}</span>
    </p>
    '''

    st.markdown(html_content, unsafe_allow_html=True)







def show_Installed_LLMS():

    data = read_model_options_from_file(filename).values()

    col1_, col2_, col3_ = st.columns([0.7, 0.15, 0.25])

    for options in data:
        model_file_name = options["model_file_name"]
        model_id = options["repo_id"]
        download = options["download"]
        recommended = options["recommended"]

        with col1_:        
            create_language_model_Link_Button(model_id, model_file_name)
            
        with col2_:
            if st.button("Delete", key=f"{model_file_name}_Del", type="secondary", help=f"Remove '{model_file_name}' completely from disk"):

                data = read_model_options_from_file(filename)
                if not recommended:
                    deletePath = hf_hub_download(repo_id=model_id, filename=model_file_name, repo_type='model')
                    parentDirectory = os.path.dirname(os.path.dirname(os.path.dirname(deletePath)))
                    send2trash.send2trash(parentDirectory)
                    st.info("'{filename}' has been removed! It has been moved to the Recycle Bin", icon='ℹ️')

                delete_llm_from_Json(data, model_file_name)
                update_llm_Json(filename, data)   

        with col3_:
            if download:
                st.button("Is available", key=f"ha{model_file_name}_Del", type="secondary", help=f"Remove '{model_file_name}' completely from disk", disabled=True)
                
            else:
                downloadbtn = st.button("Download", key=f"hae{model_file_name}_Del", type="secondary", help=f"Remove '{model_file_name}' completely from disk")

                if downloadbtn:
                    with st.spinner(f"Download '{model_id}'"):
                        init.model_path = hf_hub_download(repo_id=model_id, filename=model_file_name, repo_type='model')
                        st.success("Download was successful!")

                        data = read_model_options_from_file(filename)

                        for key, value in data.items():
                            if value.get("repo_id") == model_id:
                                value["download"] = True
                        
                        update_llm_Json(filename, data)


