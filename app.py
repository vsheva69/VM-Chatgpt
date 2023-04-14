import openai
import os
import urllib.request
from PIL import Image
import streamlit as st
from streamlit_chat import message
from streamlit_option_menu import option_menu
from datetime import datetime
import random

from dotenv import load_dotenv
load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
response = False
prompt_tokens = 0
completion_tokes = 0
total_tokens_used = 0
cost_of_response = 0
translate_button= False

# Setting page title and header
st.set_page_config(page_title="VM ChatGPT API", page_icon=":robot_face:")

# Disable the submit button after it is clicked
def disable():
    st.session_state.disabled = True

# Initialize disabled for form_submit_button to False
if "disabled" not in st.session_state:
    st.session_state.disabled = False

#Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'format_type' not in st.session_state:
    st.session_state['format_type'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0


st.session_state['audio_file'] = []
with st.sidebar:
    selected = option_menu("VM ChatGPT API", ["Chat", "DALL E", "Whisper"],
                         icons=['book', 'camera fill', 'file-earmark-music-fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "#a614c7", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#5b91f0"},
    }
    )


def generate_image(resolusi,image_description):
  img_response = openai.Image.create(
    prompt = image_description,
    n=4,
    size=resolusi)
  img_url0 = img_response['data'][0]['url']
  img_url1 = img_response['data'][1]['url']
  img_url2 = img_response['data'][2]['url']
  img_url3 = img_response['data'][3]['url']

 
  urllib.request.urlretrieve(img_url0, 'img_result0.png')
  urllib.request.urlretrieve(img_url1, 'img_result1.png')
  urllib.request.urlretrieve(img_url2, 'img_result2.png')
  urllib.request.urlretrieve(img_url3, 'img_result3.png')
  img0 = Image.open("img_result0.png")
  img1 = Image.open("img_result1.png")
  img2 = Image.open("img_result2.png")
  img3 = Image.open("img_result3.png")

  return img0,img1,img2,img3


def make_request(models,question_input: str):
    st.session_state['messages'].append({"role": "user", "content": question_input})

    completion = openai.ChatCompletion.create(
        model=models,
        messages=st.session_state['messages']
    )
    response = completion.choices[0].message.content
    st.session_state['messages'].append({"role": "assistant", "content": response})

    # print(st.session_state['messages'])
    total_tokens = completion.usage.total_tokens
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    return response, total_tokens, prompt_tokens, completion_tokens

def make_request2(models,question_input: str):
    st.session_state['messages'].append({"role": "user", "content": question_input})

    completion = openai.Completion.create(
      model=models,
      prompt=question_input,
      max_tokens=150,
      temperature=0.7
    )
    response = completion["choices"][0]["text"] 
    st.session_state['messages'].append({"role": "assistant", "content": response})

     # print(st.session_state['messages'])
    total_tokens = completion.usage.total_tokens
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    return response, total_tokens, prompt_tokens, completion_tokens






def make_request3(audio_file):
    transkrip = openai.Audio.transcribe("whisper-1", audio_file)
    return transkrip

def make_request4(audio_file):
    
    translate = openai.Audio.translate("whisper-1", audio_file)
    return translate




if selected == "DALL E":
    # page title
    st.title('Image Generation - OpenAI')

    

    # text input box for image recognition
    img_description = st.text_input('Deskripsi Gambar')

    col0, col1 = st.columns(2)
    with col0:
        resolusi = st.radio(
        "Resolusi Gambar",
        ('256x256', '512x512', '1024x1024'))
    with col1:
        jumlah_gambar = st.radio(
        "Jumlah Gambar",
        (1, 2, 3, 4))


    if st.button('Buat Gambar dari deskripsi'):
        
        #img0,img1,img2,img3 = generate_image(resolusi,img_description) #2023-04-13 script lama
        if resolusi == "1024x1024":
            cost =  0.020  * jumlah_gambar
        elif resolusi == "512x512":
            cost = 0.018 * jumlah_gambar
        else:
            cost = 0.016 * jumlah_gambar
        st.markdown("""---""")
        st.write(f"Total biaya model gambar ini: ${cost:.5f}")

        img_response = openai.Image.create(
           prompt = img_description,
            n=jumlah_gambar,
            size=resolusi)
 
        cols = st.columns(jumlah_gambar)
        for x in range(jumlah_gambar):
            img_url = img_response['data'][x]['url'] 
            urllib.request.urlretrieve(img_url, 'img/'+img_description+'-'+resolusi+' '+str(x)+'.png')
            img = Image.open("img/"+img_description+'-'+resolusi+' '+str(x)+".png")
            with cols[x]:
                st.image(img)
        
   
elif selected == "Chat":
    # page title
    st.title('Chat - OpenAI')
    
    with st.sidebar:
        format_type = st.selectbox('Pilih Language Model OpenAI 😉',["gpt-3.5-turbo","gpt-3.5-turbo-0301","gpt-4","gpt-4-0314","gpt-4-32k","gpt-4-32k-0314","text-davinci-003"])
        counter_placeholder = st.sidebar.empty()
        counter_placeholder.write(f"Total biaya percakapan ini: ${st.session_state['total_cost']:.5f}")
        clear_button = st.sidebar.button("Hapus Percakapan", key="clear")
    st.markdown("""---""")
   

    # reset everything
    if clear_button:
        st.session_state['generated'] = []
        st.session_state['past'] = []
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        st.session_state['number_tokens'] = []
        st.session_state['format_type'] = []
        st.session_state['cost'] = []
        st.session_state['total_cost'] = 0.0
        st.session_state['total_tokens'] = []
        counter_placeholder.write(f"Total biaya percakapan ini: ${st.session_state['total_cost']:.5f}")
    else:
        pass

    # container for chat history
    response_container = st.container()
    # container for text box
    container = st.container()
    
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("VM Prompt:", key='input', height=100)
            submit_button = st.form_submit_button(label='Kirim')

        if submit_button and user_input:
            if format_type == "gpt-3.5-turbo" or format_type =="gpt-3.5-turbo-0301"or format_type =="gpt-4"or format_type =="gpt-4-0314"or format_type =="gpt-4-32k" or format_type =="gpt-4-32k-0314":
               output, total_tokens, prompt_tokens, completion_tokens = make_request(format_type,user_input)
            else:
               output, total_tokens, prompt_tokens, completion_tokens = make_request2(format_type,user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            st.session_state['format_type'].append(format_type)
            st.session_state['total_tokens'].append(total_tokens)

            # from https://openai.com/pricing#language-models
            if format_type == "gpt-3.5-turbo" or format_type =="gpt-3.5-turbo-0301":
                cost = total_tokens * 0.002 / 1000
            elif format_type =="gpt-4" or format_type =="gpt-4-0314"or format_type =="gpt-4-32k" or format_type =="gpt-4-32k-0314":
                cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000
            else:
                cost = total_tokens * 0.02 / 1000

            st.session_state['cost'].append(cost)
            st.session_state['total_cost'] += cost

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
                st.write(
                    f"Model used: {st.session_state['format_type'][i]}; Jumlah Token: {st.session_state['total_tokens'][i]}; Biaya: ${st.session_state['cost'][i]:.5f}")
                counter_placeholder.write(f"Total biaya percakapan ini: ${st.session_state['total_cost']:.5f}")

else:
    # page title
    st.title('Speech to text - OpenAI')
    audio_file = st.file_uploader("Pilih berkas audio", accept_multiple_files=False,type="mp3")
    col0, col1 = st.columns(2)
    with col0:
        rerun_button = st.button("Transkripsi Asli")
    with col1:
        translate_button = st.button("Terjemahkan Bahasa Inggris")
    st.markdown("""---""")

    if rerun_button:
      transkrip = make_request3(audio_file)
      st.session_state['transkrip'] = transkrip["text"]
      st.write("Transkripsi Asli: "+transkrip["text"]) 

    

    if translate_button:
      translate = make_request4(audio_file)
      st.write("Transkripsi Asli: "+st.session_state['transkrip']) 
      st.info("Terjemahan Bahasa Inggris : "+translate["text"]) 
    else:
        pass
    

    