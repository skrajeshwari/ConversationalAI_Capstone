# Step 1: Import

#!pip install --upgrade numpy
#!pip install --upgrade pandas
!pip install --upgrade pandas==2.0.3
!pip install --upgrade numpy==1.25.2
!pip -q install openai
!pip -q install --upgrade gradio
!pip -q install langchain-openai
!pip -q install langchain-core
!pip -q install langchain-community
!pip -q install sentence-transformers
!pip -q install langchain-huggingface
!pip -q install langchain-chroma
!pip -q install langchain_core.runnables
!pip -q install chromadb
!pip -q install pypdf
!pip install faiss-cpu
!pip install -U langchain
!pip install --upgrade langchain

import os
import pickle
import openai
import numpy as np
import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import gradio as gr
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableMap, RunnableLambda

Step 2: Global Variables declaration and Agent Name Initialization

counter = True
count = 1

# Initialize agent name & execution mode selector(agent vs expert)
agent_name = ""
exec_mode = ""

# Step 3: Load Model
from google.colab import userdata
from langchain_openai import ChatOpenAI

api_key = userdata.get('OA_API')    # <-- change this as per your secret's name
os.environ['OPENAI_API_KEY'] = api_key
openai.api_key = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(
	temperature=0.5,
	openai_api_key=os.environ["OPENAI_API_KEY"],
	model_name="gpt-4o-mini"
)

# Try a quick general query to ensure the llm is working properly.
response = llm.invoke("How can a student start learning programming? Give us 5 pointers")
print(response.content)

# STEP 04: Format chat history

def format_chat_history(chat_history):
    formatted_history = ""
    for message in chat_history:
        try:
            for role, content in message.items():
                formatted_history += f"{role}: {content}\n"
        except AttributeError:
            formatted_history += f"{message}\n"
    return formatted_history

# STEP 05: Generate Human Agent Name

import random

def getHumanAgentName(exec_mode):

  input_variables=["Mani","Kamala", "Katrina", "Parul","Rahul", "Sita", "Rati"]

  counter = False

  # For debug purpose only
  #print ("inside gethuman")
  #print(counter)

  if exec_mode == "expert":
      return "Shiva"
  else:
      return random.choice(input_variables)

  #STEP 06: Write chat history to file

import datetime

def write_chat_history_to_file(chat_history, filename="chat_history.txt"):
    """Writes the chat history to a file with timestamps and session markers."""

    try:

        with open(filename, "a") as f:  # Open in append mode
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M")
            f.write(f"\n--- New Session :   {agent_name}  , {timestamp} ---\n")
            for message in chat_history:
                try:
                    # For debug purpose only
                    # Append timestamp to each message
                    #timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    for role, content in message.items():
                        f.write(f"{role}: {content}\n")
                except (AttributeError, TypeError): # Handle cases where message might not be a dictionary
                    f.write(f"{message}\n")
    except Exception as e:
        print(f"Error writing to file: {e}")

#  STEP 07: LOAD RAG DATA FILES
#  Extract audio from video files.

!pip install moviepy

from moviepy.editor import VideoFileClip

# Load all videos from video_files directory one by one, extract audio and write
# them as audio files in audio_files directory.

# Define the video and audio directories
video_dir = "/content/video_files"
audio_dir = "/content/audio_files"

# Create the audio directory if it doesn't exist
if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)

# Iterate over all .mp4 files in the video directory
for video_file in os.listdir(video_dir):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(video_dir, video_file)

        # Create the corresponding audio file path
        audio_file_name = video_file.replace(".mp4", ".wav")
        audio_path = os.path.join(audio_dir, audio_file_name)

        # Extract audio from video
        try:
            with VideoFileClip(video_path) as video:
                video.audio.write_audiofile(audio_path)
            print(f"Audio extracted successfully from {video_file} and saved as {audio_file_name}.")
        except Exception as e:
            print(f"An error occurred while processing {video_file}: {e}")

# Audio extraction completed
print("All videos processed & audios extracted.")

# STEP 08: Install requisite dependecies for audio to text conversion

!pip install moviepy SpeechRecognition openai-whisper
!pip install torch torchvision torchaudio
!pip install pydub
!pip install chromadb
!pip install faiss-cpu
!pip -q install openai
!pip -q install gradio
!pip -q install langchain-openai
!pip -q install langchain-core
!pip -q install langchain-community
!pip -q install sentence-transformers
!pip -q install langchain-huggingface
!pip install transformers
!pip install faiss-cpu
!pip -q install langchain-chroma
!pip install fpdf

# STEP 09: Import necessary modules for audio to text conversion

import torch
import torchaudio
import torchaudio.transforms as T
import chromadb
from chromadb.config import Settings
import librosa
from langchain.schema import Document
import os
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import faiss
import numpy as np
from langchain.docstore.document import Document

# STEP 10: CONVERT AUDIO INTO TEXT TRANSCRIPTION

import os
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
#from sentence_transformers import SentenceTransformer
from fpdf import FPDF

# Load the Whisper ASR model and processor
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
processor = WhisperProcessor.from_pretrained("openai/whisper-large")

# Directory containing audio files
audio_dir = "/content/audio_files/"

# Directory to save transcriptions
transcription_dir = "/content/pdf_files"
os.makedirs(transcription_dir, exist_ok=True)

# Duration of each chunk in seconds
chunk_duration = 30

# Function to split audio into smaller chunks
def split_audio(audio, sr, chunk_duration):
    num_samples_per_chunk = chunk_duration * sr
    chunks = [audio[i:i + num_samples_per_chunk] for i in range(0, len(audio), num_samples_per_chunk)]
    return chunks
  
# Process each audio file in the directory
for audio_file in os.listdir(audio_dir):
    if audio_file.endswith(".wav"):  # Filter for .wav files
        file_path = os.path.join(audio_dir, audio_file)
        print(f"Processing file: {file_path}")

        # Initialize PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)

        # Load the audio file
        audio, sr = librosa.load(file_path, sr=16000)

        # Split audio into chunks
        audio_chunks = split_audio(audio, sr, chunk_duration)
        # Initialize an empty string for the full transcription
        transcription = ""
        # Store chunk transcriptions for embedding generation
        chunk_transcriptions = []

        for i, chunk in enumerate(audio_chunks):
            # Prepare input features for Whisper
            input_features = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features

            # Perform inference
            with torch.no_grad():
                predicted_ids = model.generate(input_features)

            # Decode transcription
            chunk_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcription += f"Chunk {i + 1}:\n{chunk_transcription}\n\n"
            chunk_transcriptions.append(chunk_transcription)

        # Add the transcription to the PDF
        pdf.add_page()
        pdf.multi_cell(0, 10, f"Transcription for file: {audio_file}\n\n{transcription}")

        # Print to console
        print(f"Full Transcription for {audio_file}:\n{transcription}")

        # Create the corresponding transcription file path
        transcription_file_name = audio_file.replace(".wav", ".pdf")
        transcription_file_path = os.path.join(transcription_dir, transcription_file_name)
        # Save the PDF
        pdf.output(transcription_file_path)
        print(f"Transcription saved to {transcription_file_path}")

# All transcriptions completed
print(f"All transcriptions saved to PDF at {transcription_dir}")

# STEP 11: LOAD THE DOCUMENT FILES

# UPLOAD the Docs first to this notebook, then run this cell

# Folder where the PDF files are located
pdf_dir = "/content/pdf_files/"

# List to store the content of all loaded PDF files
docs = []

# Iterate through all files in the pdf_folder
for filename in os.listdir(pdf_dir):
    # Check if the file ends with .pdf (to only process PDF files)
    if filename.endswith(".pdf"):
        # Construct the full file path
        file_path = os.path.join(pdf_dir, filename)
        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(file_path)
        # Append the loaded content to the docs list
        docs.extend(loader.load())

# Now 'docs' contains the content of all PDF files in the /content/pdf_files
# folder. Check the number of pages in the combined docs
print(f"Number of pages in the combined docs: {len(docs)}")

# Print first & last pages as samples
print("\nPage 0: \n", docs[0].page_content)
print("\nPage -1: \n", docs[-1].page_content)

# SPLIT THE LOADED DOCUMENTS INTO CHUNKS


# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)
splits = text_splitter.split_documents(docs)

print("Total No of chunks", len(splits))
print("Length of the first chunk", len(splits[0].page_content) )
print("Contents of the first chunk=>\n", splits[0].page_content)

# STEP 12: OPENAI EMBEDDINGS

from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model='text-embedding-3-small')

# Check embedding
print("Created embedding =>\n", embedding)

# STEP 13: VECTORIZATION

# Load the chunks & embedding
persist_directory = 'docs/faiss/'
!rm -rf ./docs/faiss  # remove old database files if any
vectordb = FAISS.from_documents(documents=splits, embedding=embedding)
vectordb.save_local(persist_directory)

# STEP 14: RETRIEVAL

# Create a retriever for retrieving chunks
#retriever = vectordb.as_retriever()
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 7, "fetch_k":15})

# Check retriever
print("Created retriever =>\n", retriever)

# Try a sample question
test_question = "What are the vision models available in Llama3.2 release?"

# Print retrieved docs
ret_docs = retriever.invoke(test_question)
ret_docs

# STEP 15: Try another search technique with similarity search threshold.
# Since the threshold value is not very high and keeps on varying, we will stick to mmr. 


similar_retriever = vectordb.as_retriever(search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': 0.5, "k": 7})
# Check retriever
print("Created retriever =>\n", similar_retriever)

# Try a sample question
test_question = "What are the vision models available in Llama3.2 release?"

# Print retrieved docs
ret_docs = similar_retriever.invoke(test_question)
ret_docs

# STEP 16: PERFORM PLATFORM ANALYSIS

def PlatformAnalysis(UserQuestion):
  #print("here1")
  #prompt = PromptTemplate.from_template(template)
  global promptAnalysis
  prompt_template = PromptTemplate(
    input_variables=["UserInput","Context"],

    template = """Answer the question based only on the following context, keep it crisp and brief:
    {context}
    Question: {Question}
    Provide output in the format below throughout the conversation.
    Sentiment=>: (Accuratly detect customer sentiment viz. positive, negative or neutral).
    Category=> (Accurately identify granular sentiment categories viz. frustration, satisfaction,
    inquiry, etc.)
    Intent=> (Accurately recognize the intent behind customers' queries.).
    Topic=> (Accurately identify the key topic of discussion).
    PreTrained=>(
                   False - If the topic is related to UttaraKhand or Llama3.2 or llama3.2 or Llama 3.2 or llama 3.2
                   True - If the topic is covered as part of your training dataset
                )
    """)
  #System=>  Dont show response on  sentiment, category and Intent analysis. Show only LLM response.
  UserInput = UserQuestion

  # Create a prompt using the prompt template
  prompt = prompt_template.format(Question = UserInput, context = "Sentiment Analysis / Intent Recognition") # Placeholder for promptAnalysis

  # Generate results using the LLM application
  result = llm.invoke(prompt)
  promptAnalysis = result.content

  # For debug purpose only
  # Replace the placeholder in the template with the actual LLM output
  #final_output = prompt_template.format(Question = UserInput, context = "Sentiment Analysis / Intent Recognition", promptAnalysis = promptAnalysis) #, LLM_RESPONSE = result.content)

  return promptAnalysis

# STEP 17: Identify whether LLM is pretrained on identified topic or not

# Extract the value of Pretrained=> in the platform response string

def extract_pretrained_value(platform_response_string):

    try:
        start_index = platform_response_string.index("PreTrained=>") + len("PreTrained=>")
        end_index = platform_response_string.find("\n", start_index)
        if end_index == -1:  # If no newline is found, take the rest of the string
          end_index = len(platform_response_string)

        pretrained_value = platform_response_string[start_index:end_index].strip()
        return pretrained_value
    except ValueError:
        return None

  # STEP 18: Get PROMPT Template - RAG Vs LLM

def GetTemplate(whattocall,greet_message):
    template = ""
    if whattocall == "LLM":
        template = """The following is a friendly conversation between a human and an AI.
        This AI is intuitive and can answer questions from history.
        The AI response is brief and specific. You have a name.
        First greet the customer with " """ + greet_message + """
        Secondly, if platform_analysis_result is positive or neutral, ask questions.
        Thirdly, Say Apologies, if platform_analysis_result is negative.
        Use only training set data to generate the response. Do not use web search and your RAG implementation
        while generating the response.If the AI does not know the answer to a question,
        it truthfully says "I dont know".Based on positive or negative or neutral response present
        in platform_analysis_result, respond accordingly to the customer.
        Current conversation:
        {history}
        Friend: {input}
        AI Assistant:"""
    else:
        template = """Use the following pieces of context to answer the question at the end.
        Go through the document and answer the questions asked.
        {context}
        Question: {question}
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Helpful Answer:"""
    return template

# STEP 18: Function to process both LLM and RAG prompt using single conversational memory

# Function to process prompts using the single memory chain

def process_prompt(userQuestion, inputType, greetMessage,chatHistory):

    # Initialize conversation memory
    conversation_memory = ConversationBufferWindowMemory(k=5)

    PROMPT = PromptTemplate(input_variables=["history", "input"],template=GetTemplate("LLM",greetMessage))
    if 'conversation_with_summary' not in globals():
        global conversation_with_summary
        conversation_with_summary = ConversationChain(
        llm=llm,
        prompt=PROMPT,
        memory=conversation_memory,
        verbose=True
    )

    # Update the RAG chain to include memory
    if inputType == "RAG":
        prompt = PromptTemplate.from_template(GetTemplate("RAG",greetMessage))

        # Create a chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        response = rag_chain.invoke(userQuestion)
    else:
        response = conversation_with_summary.predict(input=userQuestion,history=chatHistory)

    return response

# STEP 19: Generate Response - RAG Vs LLM

def generate_response(submit_button, platform_analysis_result, user_question, human_agent_response, chat_history=None):
    # Initialize chat_history if it's None (first call)
    if chat_history is None:
        chat_history = []

    global counter
    global agent_name
    global count

    customer_data = []
    responses = {}
    openai_response = ""
    llm_prompt_template = ""
    rag_prompt_template = ""
    llm_input_values = ""
    rag_input_values = ""
    strVal = ""
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M")

    if counter==True:
        agent_name = getHumanAgentName(exec_mode)
        counter = False
        greet_message = "Namaste I'm " + agent_name + ""
    else:
        greet_message = ""
    try:
      # Define a single conversation memory chain
      formatted_history = format_chat_history(chat_history)
      conversation_memory = ConversationBufferWindowMemory(k=5)
      strVal = extract_pretrained_value(platform_analysis_result)

      # For debug purpose only
      #print("strval")
      #print(strVal)

      if "false" in strVal.lower(): #Retrive from RAG
          tag = "System=>"
          messageTag = "Retreiving from RAG...."
          # Example usage for the two prompts (RAG and LLM)

          # 1. RAG Prompt
          openai_response = process_prompt(user_question,"RAG",greet_message,formatted_history)
          print(f"RAG Response:\n{openai_response}")
      else : #Retrive from LLM
          # 2. LLM Prompt
          tag = "RAG=>"
          messageTag = "Not Invoked...."
          openai_response = process_prompt(user_question,"LLM",greet_message,formatted_history)
          print(f"LLM Response:\n{openai_response}")


      # Format the chat history correctly for the prompt
      formatted_history = format_chat_history(chat_history)
      print("1")
      if submit_button == "Submit Human Agent Response":
            print("2")
            chat_history.append({"Human=>": human_agent_response})
            chat_historyforLog.append({timestamp})
            chat_historyforLog.append({"Human=>": human_agent_response})
            print("3")
            return "", chat_history, chat_historyforLog
      elif submit_button == "Submit Customer Query":
            chat_history.append({"Lap No ": count})
            chat_history.append({"Customer=>": user_question})
            #chat_history.append({"Customer": user_question})
            chat_history.append(promptAnalysis)
            chat_history.append({tag: messageTag})
            #chat_historyforLog.append({tag: messageTag})
            chat_historyforLog.append({"Lap No ": count})
            chat_historyforLog.append({timestamp})
            chat_historyforLog.append({"Customer=>": user_question})
            chat_historyforLog.append({timestamp})
            chat_historyforLog.append(promptAnalysis)
            chat_historyforLog.append({tag: messageTag})
            if tag == "RAG=>":
               tag = "System=>"
            else:
              tag = "RAG=>"
            chat_history.append({tag: openai_response})
            chat_historyforLog.append({tag: openai_response})

            #chat_historyforLog.append({"Customer": user_question})
            count += 1

    except Exception as e:
        error_message = f"An error occurred: {e}"
        print(error_message)
        return error_message, chat_history, chat_historyforLog  # Return chat_historyforLog as well
    return openai_response, chat_history, chat_historyforLog
chat_history = []
chat_historyforLog = []

# STEP 20: Close Session

def stop_app():
   with gr.Blocks() as demo:
      #  Perform any necessary cleanup or saving before exiting.
        print("Stopping the Gradio application.")
        #  Here you could potentially save the chat history or other data.
        demo.close()  # Close the Gradio app.
        write_chat_history_to_file(chat_historyforLog)  # Save chat history to a file.
        print("Chat history saved.")
        return "Application stopped."  # Return a confirmation message.

# STEP 21: Select agent vs expert execution mode.
# The execution flow remains same. But in agent mode, the performance of the human agent is evaluated. 
# On the other hand, in expert mode, the LLM's performance is evaluated. 
# The getHumanAgentName() function uses this setting.

exec_mode = ""

user_ip = input("Please select execution mode (agent or expert):")
if user_ip.lower() == "agent":
    exec_mode = "agent"
elif user_ip.lower() == "expert":
    exec_mode = "expert"
else:
    print("Invalid input. Please enter 'user' or 'expert'by rerunning the cell.")

print("Selected execution mode:", exec_mode)

# PRINT 22: CREATE UI

def ResponseFromUI(UserQuestion,HumanAgentResponse,submit_button):
  global chat_history  # Access the global chat histor
  global chat_historyforLog

  platformAnalysisResponse = PlatformAnalysis(UserQuestion)

  human_response = gr.Textbox(lines=4, label="Human Agent Response")
  LLMResponse, chat_history, chat_historyforLog = generate_response(submit_button,platformAnalysisResponse,UserQuestion,HumanAgentResponse, chat_history=chat_history)
  if submit_button == "Submit Customer Query":
     human_response = LLMResponse
  formattedChatHistory = format_chat_history(chat_history)

  # For debug purpose only
  #write_chat_history_to_file(chat_historyforLog)

  if UserQuestion == "":
      return "Please enter prompt to proceed", "", "",""
  else:
        return platformAnalysisResponse, LLMResponse,human_response,formattedChatHistory

#"\n".join([f"User: {msg.get('Customer', '')}\nAI: {msg.get('AI', '')} \n User Agent: {msg.get('Human Agent', '')}" for msg in updated_chat_history])
with gr.Blocks() as demo:
    #chat_history = gr.State([])  # Initialize chat history state
    with gr.Row():
      with gr.Column(scale=1, min_width=500):
        user_query = gr.Textbox(lines=4, label="User Query")
        submit_btn = gr.Button("Submit Customer Query")
        clear_btn = gr.Button("Clear Customer Response")
      with gr.Column(scale=1, min_width=500):
        human_response = gr.Textbox(lines=4, label="Human Agent Response")
        submit_btn2 = gr.Button("Submit Human Agent Response")
        clear_btn2 = gr.Button("Clear Human Agent Response")

    with gr.Row():
      with gr.Column(scale=1, min_width=300):

        platformresponse_textbox = gr.Textbox(lines=4, label="Customer Sentiment & Intent Analysis", interactive=False)

      with gr.Column(scale=1, min_width=300):
        llm_response = gr.Textbox(lines=3, label="LLM Generated Response")
    with gr.Row():
      with gr.Column(scale=1, min_width=300):

        closeSession_btn = gr.Button("Close Session")
        closeSession_btn.click(stop_app, outputs=gr.Textbox(label="Status"))
        #copyLLMResponse_btn = gr.Button("Copy AI Response")

    with gr.Row():
      chat_history_textbox = gr.Textbox(lines=10, label="Chat History", interactive=False)

    submit_btn.click(ResponseFromUI, inputs=[user_query, human_response, gr.Textbox(value="Submit Customer Query", visible=False)], outputs=[platformresponse_textbox, llm_response,human_response, chat_history_textbox], queue=False)
    submit_btn2.click(ResponseFromUI, inputs=[user_query, human_response, gr.Textbox(value="Submit Human Agent Response", visible=False)], outputs=[platformresponse_textbox, llm_response, human_response,chat_history_textbox], queue=False)

    # For debug purpose only
    #copyLLMResponse_btn.click(CopyLLMResponse, outputs=gr.Textbox(label="Human Agent Response"))
    #submit_btn.click(ResponseFromUI, inputs=[user_query,human_response], outputs=[platformresponse_textbox, llm_response, chat_history_textbox], queue=False)

    clear_btn.click(lambda: ["", "", []], outputs=[platformresponse_textbox, llm_response, chat_history_textbox,human_response], queue=False)

    # For debug purpose only
    #submit_btn.click(ResponseFromUI, inputs=[human_response], outputs=[human_response], queue=False)


demo.launch(share=True, debug=True)

# EVALUAATE RAG PERFORMANCE

# STEP 23: Import necessary RAGA modules 
!pip install ragas

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

# STEP 24: Create necessary questions, ground truths & rag chain for rag performance analysis.

# Questions to be asked for assessing RAG performance and their best answers.

rag_perf_questions = ["What are the vision models available in Llama3.2 release?",
             "What is the context length of text-only models in Llama3.2 release?",
             "What do the presenters think they would need if the Llama model family keeps growing?",
             "How can one use Llama3.2 vision 11B model for free?",
             "What are the different stages involved in the training pipeline used for Llama3.2 vision models? Focus on the pre-training part. ",
            ]

rag_perf_ground_truths = ["The 11B and 90B vision models areavailable in Llama3.2 release.",
                 "The context length of text-only models in Llama3.2 release is 128K tokens.",
                 "According to the video, the presenters suggest that they would need a bigger desk.",
                 "One can use Llama3.2 vision 11B model for free using Hugging Face spaces.",
                 "The pre-training pipeline for LLama3.2 models consists of multiple stages. It starts with pretrained Llama 3.1 text models, then addition of image adapters and encoders, followed by pre-training on large-scale noisy (image, text) and medium-scale high quality in-domain and knowledge-enhanced (image, text) pair data.",
              ]

# Answers given by the LLM & contexts retrieved by the RAG in response to the
# RAG performance questions
rag_perf_answers = []
rag_perf_contexts = []

# For the RAG implementation used in this code, the prompt template is
# generated by the GetTemplate() function. Please modify this to adapt to
# the specific RAG implementation.
rag_perf_greet_message = "Namaste I'm " + agent_name + ""
rag_perf_prompt = PromptTemplate.from_template(GetTemplate("RAG", rag_perf_greet_message))

# Create a chain
rag_perf_chain = (
          {"context": retriever, "question": RunnablePassthrough()}
          | rag_perf_prompt
          | llm
          | StrOutputParser()
      )

print("Created necessary questions, ground truths & rag chain for rag performance analysis\n")

# STEP 25: Send questions to RAG, collect answers & contexts.

# Ensure answers given by the LLM & contexts retrieved by the RAG in response
# to the RAG performance questions are cleared to avoid any repeatations.
rag_perf_answers = []
rag_perf_contexts = []

# Shoot the questions and collect answers & contexts
for query in rag_perf_questions:
    print(query)
    rag_perf_answers.append(rag_perf_chain.invoke(query))
    rag_perf_contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

print("RAG Answers: \n", rag_perf_answers)
print("RAG Contexts: \n", rag_perf_contexts)

# Create a dictionary; add a "reference" field initialized to ground truths.
rag_perf_data = {
    "question": rag_perf_questions,
    "answer": rag_perf_answers,
    "contexts": rag_perf_contexts,
    "ground_truths": rag_perf_ground_truths,
    "reference": rag_perf_ground_truths,
}

# Convert the dictionary to a dataset
rag_perf_dataset = Dataset.from_dict(rag_perf_data)

# STEP 26: Evaluate RAG performance using RAGAS library

# Evaluate rag performance.
rag_perf_result = evaluate(
    dataset = rag_perf_dataset,
    metrics=[context_precision, context_recall, faithfulness, answer_relevancy,]
)

rag_perf_df = rag_perf_result.to_pandas()

print(rag_perf_df)

