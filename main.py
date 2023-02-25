"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.llms import OpenAI
from ingest_data import embed_doc

from query_data import _template, CONDENSE_QUESTION_PROMPT, QA_PROMPT, get_chain
import pickle
import os
from langchain.callbacks import get_openai_callback
# PART 2 ADDED PIP INSTALL WIKIPEDIA API with "pip install Wikipedia-API"
import wikipediaapi
import os
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# PART 2 ADDED
wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
)

def wiki_search(topic):
    page_py = wiki_wiki.page(topic)
    title = page_py.title
    text = page_py.text
    title = title.lower()
    # !!!! IMPORTANT encode and decode to remove non-ascii characters 
    title = title.encode("ascii", "ignore").decode()
    text = text.encode("ascii", "ignore").decode()
    if title not in os.listdir("data"):
        with open(f"data/{title}.json", "w") as f:
            f.write(text)




# def load_chain():
#     """Logic for loading the chain you want to use should go here."""
#     llm = OpenAI(temperature=0)
#     chain = ConversationChain(llm=llm)
#     return chain



# From here down is all the StreamLit UI.
st.set_page_config(page_title="Q&A over custom documents Demo", page_icon=":robot:")
st.header("Q&A Custom  Demo")

uploaded_file = st.file_uploader("Upload a document you would like to chat about", type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

# check if file is uploaded and file does not exist in data folder
if uploaded_file is not None and uploaded_file.name not in os.listdir("data"):
    # write the file to data directory
    
    #if uploaded_file.name has pdf extension. extract text and store in a .txt file
    # Check if the file has a .pdf extension
    


    with open("data/" + uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.write("File uploaded successfully")


    if uploaded_file.name.endswith(".pdf"):
        # Call the function to convert PDF to text
        print('PDF detected')
        print(type(uploaded_file))
        pdf_filepath = os.path.splitext(uploaded_file)[0] + ".pdf"
        print(pdf_filepath)
        text = pdf_to_text(pdf_filepath)
        #print(text)

        # Create output file path by replacing .pdf extension with .txt
        output_filepath = os.path.splitext(uploaded_file)[0] + ".txt"

        # Write the text to the output file
        with open(output_filepath, "w") as f:
            f.write(text)
        print("Text saved to", output_filepath)
    else:
        print("Input file is not a PDF.")



    with st.spinner('Document is being vectorized...'):
        embed_doc()
# open vectorstore.pkl if it exists in current directory
if "vectorstore.pkl" in os.listdir("."):
    with open("vectorstore.pkl", "rb") as f:
        
        vectorstore = pickle.load(f)
        print("Loading vectorstore...")

    chain = get_chain(vectorstore)

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []




placeholder = st.empty()
def get_text():
    
    input_text = placeholder.text_input("You: ", value="",  key="input")
    return input_text


user_input = get_text()



if st.button("Submit Your Query"):
    # check 
    docs = vectorstore.similarity_search(user_input)
    # if checkbox is checked, print docs

    print(len(docs))
    # PART 2 ADDED: CALLBACK FOR TOKEN USAGE
    with get_openai_callback() as cb:
        output = chain.run(input=user_input, vectorstore = vectorstore, context=docs[:2], chat_history = [], question= user_input, QA_PROMPT=QA_PROMPT, CONDENSE_QUESTION_PROMPT=CONDENSE_QUESTION_PROMPT, template=_template)
        print(cb.total_tokens)
    

    st.session_state.past.append(user_input)
    # print(st.session_state.past)
    st.session_state.generated.append(output)
    
    print(st.session_state.generated)
    # PART2 ADDED
    # if st.session_state.generation includes "related topics:" remove that from st.session_state.generation and add it to a new list
    if "#" in st.session_state.generated[-1]:
        st.session_state.generated[-1], st.session_state.topics = st.session_state.generated[-1].split("#")[0], st.session_state.generated[-1].split("#")[1]
        
    print(st.session_state.generated)
    print(st.session_state.topics)
    print(type(st.session_state.topics))
# PART 2 ADDED
# write the topics to a topics.txt file each in a new line remove the brackets from the topics
    with open("topics.txt", "w") as f:
        for char in st.session_state.topics:
            if char == "[" or char == "]" or char == "'":
                continue
            else:
                f.write(char)
    # PART 2 ADDED: BUTTONS FOR WIKI ARTICLES
    # IF topics.txt exists, read it and display the topics  as buttons
#st.markdown("Click the buttons below to add text articles to be Vectorized")
# PART 2 ADDED: BUTTONS FOR WIKI ARTICLES
# buttons need to be in a separate column
col1, col2, col3 = st.columns(3)
#if "topics.txt" in os.listdir("."):
   # with open("topics.txt", "r") as f:
    #    topics = f.read().split(",")
    #    print(topics)
     #   if col1.button(topics[0]):
      #      wiki_search(topics[0])
      #  if col2.button(topics[1]):#
      #      wiki_search(topics[1])
       # if col3.button(topics[2]):
      #      wiki_search(topics[2])


# print(user_input)
#st.markdown("Click the button below to add your text articles to the vectorstore:")
if st.button("REBUILD VECTORSTORE", key="rebuild", help="This will rebuild the vectorstore with the added text articles."):
    with st.spinner('New Documents are being vectorized... This may take a while...'):
        embed_doc()
        with open("vectorstore.pkl", "rb") as f:
            vectorstore = pickle.load(f)
            print("Loading vectorstore...")
        chain = get_chain(vectorstore)


        # remove the brackets from the topics
        
# if st.checkbox("Show similar documents"):
#     st.markdown(docs)
    

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
    
    
