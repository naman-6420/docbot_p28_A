import re
import os
import numpy as np
import time
import fitz

from PyPDF2 import PdfReader
from PIL import Image
import streamlit as st
import google.generativeai as genai

from inference_sdk import InferenceHTTPClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
from langchain.text_splitter import CharacterTextSplitter

def generate_image(req_pg_no, path_to_uploaded_pdf, query):
        embeddings_model = OpenAIEmbeddings()
        CLIENT = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="nwRZcJDi8wcF27Nk99Hd"
        )
        model = genai.GenerativeModel('gemini-pro-vision')
        genai.configure(api_key="AIzaSyBtAvrqOqnGqRsGzO6FU81vLw0NuHPthDY")
        uploads_dir = "./uploads"

        if not os.path.exists(uploads_dir):
              os.makedirs(uploads_dir)

        path_to_image = []
        captions_dic = {}
        
        def captions(path):
            img = Image.open(path)
            response = model.generate_content(["generate most relevant captions and description in short related to the image", img],stream=True)
            response.resolve()
            try:
                print((response.candidates)[0].content.parts[0].text)
                return (response.candidates)[0].content.parts[0].text
            except:
                return ""

        def find_capt():
            for i in path_to_image :
                print(f"Captions for image {i}")
                x = captions(i)
                captions_dic[i] = x            
          
        def get_embedding(text):
          embeddings = embeddings_model.embed_documents([text])
          return embeddings[0]
        
        def find_similarity_score(text1,text2) :
            em_1 = get_embedding(text1)
            em_2 = get_embedding(text2)
            cos_sim = np.dot(em_1, em_2) / (np.linalg.norm(em_1) * np.linalg.norm(em_2))
            return cos_sim
        
        def get_images_yolo(image):
            result = CLIENT.infer(image, model_id="file_2/2")
            out = []
            for bounding_box in result['predictions']:
                x_center = bounding_box['x']
                y_center = bounding_box['y']
                width = bounding_box['width']
                height = bounding_box['height']

                x1 = x_center - width / 2
                x2 = x_center + width / 2
                y1 = y_center - height / 2
                y2 = y_center + height / 2

                box = (x1, y1, x2, y2)
                cropped_image = image.crop(box)
                out.append(cropped_image)
            return out


        def convert_to_image(path):
            pdf_document = fitz.open(path)
            i = 0
            dpi = 300
            for page_number in range(req_pg_no-1,req_pg_no+2):
                page = pdf_document.load_page(page_number)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
                image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                out = get_images_yolo(image)
                for img in out:
                    p = f"./uploads/image{i}.jpg"
                    path_to_image.append(p)
                    img.save(f"uploads/image{i}.jpg")
                    i+=1
        
        convert_to_image(path_to_uploaded_pdf)
        find_capt()
        most_relevant_img_path = list(captions_dic.keys())[0]
        max_score = 0
        for i in captions_dic.keys():
                s = find_similarity_score(query,captions_dic[i])
                print("Cosine similarity:", s)
                if s > max_score :
                    max_score = s
                    most_relevant_img_path = i

        return most_relevant_img_path

openai_api_key = "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"
os.environ["OPENAI_API_KEY"] = "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"

llm = OpenAI()
embeddings_model = OpenAIEmbeddings()
path_to_image = []
captions_dic = {}

def save_uploaded_pdf(uploaded_file):
    try:
        save_path = r"uploads\file.pdf"
        with open(save_path, "wb") as f:  
            f.write(uploaded_file.getvalue())
        st.success("PDF saved successfully!")
    except Exception as e:
        st.error(f"An error occurred while saving the PDF: {e}")

def pg_nos(pdf_path):
    pdf_document = fitz.open(pdf_path)
    dict={}
    started = False
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        text = page.get_text()
        page_no = re.findall(r"\d+-\d+", text[:50]+text[-50:-1])
        # print(page_number, page_no)
        if len(page_no) > 0:
            if page_no[0]=='1-1': 
                started = True
            if started: dict[page_no[0]] = page_number
    return dict


def main():
    if "messages" not in st.session_state: 
        st.session_state.messages = []
    with st.sidebar :
        st.title("Upload your PDF here")
        pdf = st.file_uploader("Upload your PDF", type='pdf')
    st.header("📚 DocBot : \nYour Smart Documentation Assistant! 💬")
    chat_container1 = st.container()
    raw_text = ""
   
    if pdf is not None:
        uploads_dir = "uploads"

        if not os.path.exists(uploads_dir):
              os.makedirs(uploads_dir)
        # pdf_reader = PdfReader(pdf)
        save_uploaded_pdf(pdf)
        pdf_document = fitz.open(r"uploads\file.pdf")

        my_dict=pg_nos(r"uploads\file.pdf")
        raw_text = ''
        first_encountered = False
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            text = page.get_text()
            page_no = re.findall(r"\d+-\d+", text[:50]+text[-50:-1])
            # print(page_number, page_no)
            if len(page_no) > 0:
                if page_no[0]=='1-1': 
                    first_encountered = True
                if first_encountered:
                    raw_text += f"\n Page no:{page_no[0]},"
                    # print(page_no[0])
                    raw_text += f" text on this page: --PAGE STARTS  \n {text}  --PAGE ENDS "
            else:
                raw_text += f"\n Page no:{page_number}, text on this page: --PAGE STARTS  \n {text}  --PAGE ENDS "
                
                
            # if text:
            #     if (text[-5:-1].strip()=="1-1"):
            #         first_encountered = True
            #     if (first_encountered):
            #         raw_text+=f"\n Page no:{text[-5:-1]},"
            #     raw_text += f" text on this page: --PAGE STARTS  \n {text[:-5]}  --PAGE ENDS "

        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 800,
            chunk_overlap  = 200,
            length_function = len,
        )
        chunks = text_splitter.split_text(raw_text)
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        
        with chat_container1:
            if prompt := st.chat_input("Hey there! 👋, Ask me questions about your PDF file:"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    docs = VectorStore.similarity_search(query=prompt, k=3)
                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=prompt+" and what is the page number given in the pdf where this information is found?")
                    def stream_data():
                        for word in response.split(" "):
                            yield word + " "
                            time.sleep(0.2)
                    st.write_stream(stream_data)
                    # TEXT OUTPUT ENDS HERE 
                    
                    def extract_last_page_number(text):
                        page_number_pattern = r"\d+-\d+"
                        matches = re.findall(page_number_pattern, text)
                        if matches and matches[-1] in my_dict.keys():
                            return my_dict[matches[-1]]
                        try:
                            return my_dict[matches[-1][:-1]]
                        except:
                            return None
                    req_pg_no = extract_last_page_number(response)
                    if req_pg_no is None: return
                    images_path = generate_image(req_pg_no, r"uploads\file.pdf", prompt)

                    st.image(images_path)
                    # SHOW IMAGE
       
if __name__ == '__main__':
    main()