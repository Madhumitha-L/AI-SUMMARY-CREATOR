from flask import Flask,render_template,request
from duplicate import *

app = Flask(__name__)

@app.route("/index")
@app.route("/")
def index():
    return render_template("index.html")
    
@app.route('/input/image',methods = ['POST'])
def handleImageInput():
    if request.method == 'POST':
        f = request.files['file']  
        f.save(f.filename)
        result=run(f.filename)
        return render_template("index.html",result=result)

if __name__ == "__main__":
    app.run()




