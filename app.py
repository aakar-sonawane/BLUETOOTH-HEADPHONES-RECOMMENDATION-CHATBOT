from flask import Flask, render_template, request
from MODULES.retrieval_generation import generation
from MODULES.data_ingestion import data_ingestion

vstore = data_ingestion("done")
chain = generation(vstore)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST", "GET"])
def chat():
    if request.method == "POST":
        message = request.form["msg"]
        input = message

        result = chain.invoke(
            {"input": input},
            config={"configurable": {"session_id": "test"}}
        )["answer"]

        return str(result)
    return ""
    
if __name__ == "__main__" :
    app.run(host="0.0.0.0", port=5000, debug=True)