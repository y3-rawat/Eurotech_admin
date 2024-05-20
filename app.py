
from flask import Flask, request, jsonify, render_template, session
import pandas as pd
import json
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

app = Flask(__name__)

key = "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"
DB_FAISS_PATH = 'dealer/store/db_faiss'

# Load the model
def load_llm():
    llm = ChatGroq(model="llama3-8b-8192", api_key=key)
    return llm

loader = CSVLoader(file_path="data_car.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                    model_kwargs={'device': 'cpu'})

db = FAISS.from_documents(data, embeddings)
db.save_local(DB_FAISS_PATH)
llm = load_llm()
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

def prompt_template(query):
        
    df = pd.read_json("Admin_database.json")
    
    template = f"""You are Eurotech Xchange Bot, a specialized assistant with detailed information about every thing related to dealer collector etc. You have access to the following data about the dealer:
    As the Eurotech Xchange Bot, you can answer any questions related to this like, bid amounts, or any other related information.
    , information about the data
    ----You are in a Admin Page ------
    Show humbalness while chatting with sir
    You have access to the following data about a specific collector:
    Total Cars Present In inventory : {df["total_cars_in_inventory"]}
    Total User Visited: {df["total_users_visited"]}
    Id's of people who are repeatative  :{df["Repetative_Users_id"]}
    
    User Who has viewed the most cars: {df["Most_car_viewed_user_id"]}
    
    Total Dealers on website :{df["Dealers_on_webiste"]}

    History_of_Transections_of_cars :{df["Transections_history"]}

    Total Car Sold : {df["Car_Sold_total"]}
    
    Car Sold Last month :{df["Car_sold_last_month"]}

    Top 5 cars viewd :{df["top_5_cars_viewd"]}

    common issues on website {df["common_issues_on_webiste"]}

    most_bids_on_cars :{df["most_bids_of_cars"]}


    As the Eurotech Xchange Bot, you can in the User_chat there will be a your boss who has this company
    BE PROFFESSIONAL WHILE TALKING TO HIM
    if your Boss is just doing casual chat then just do casual chatting about the cars and relevent things
    if user is asking in other language than answer him in there language not in english 
    Your words should be lessthen 50 words if applicable
    ---------------
        
    User_Chat: {query}
    """
    return template

def conversational_chat(query):
    prompt = prompt_template(query)
    result = chain({"question": prompt, "chat_history": session.get('history', [])})
    session['history'].append((query, result["answer"]))
    return result["answer"]

@app.route('/')
def index():
    if 'history' not in session:
        session['history'] = []
    if 'generated' not in session:
        session['generated'] = ["Hello! Ask me anything about 🤗"]
    if 'past' not in session:
        session['past'] = ["Hey! 👋"]
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['query']
    
    if user_input:
        output = conversational_chat(user_input)
        session['past'].append(user_input)
        session['generated'].append(output)
    return jsonify({'past': session['past'], 'generated': session['generated']})

@app.route("/get", methods=['GET'])
def get_bot_response():
    user_text = request.args.get('msg')
    
    if user_text:
        output = conversational_chat(user_text)
        return output
    return "Please provide a query parameter ?msg=Your_MSG and unique_id parameter ?unique_id=Your_ID"

if __name__ == "__main__":
    app.run(debug=True)
