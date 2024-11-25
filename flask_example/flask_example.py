from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    user_query = request.args.get('user_query', '')
    output = ""
    if user_query:
        # Here is where you could call a function to process the user query
        # For demonstration, we'll just reverse the user query string
        output = process_query(user_query)
    
    return render_template('index.html', output=output)

def process_query(query):
    # Example function that processes the user query
    return query[::-1]

if __name__ == '__main__':
    app.run(debug=True)
