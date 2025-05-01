import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from flask import Flask, request, render_template, redirect, url_for, jsonify
from cloudant.client import Cloudant
from werkzeug.utils import secure_filename

# Load the model
from keras.models import load_model

MODEL_PATH = r"major project\flask\Updated-Xception-diabetic-retinopathy (1).h5"

model = load_model(MODEL_PATH)


try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)

# Initialize Flask app
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Authenticate with Cloudant using an IAM API key
client = Cloudant.iam('9bdebd12-5220-4625-868a-61425039d894-bluemix', 'KJ7y4CpV4hIsvXdH0uUNWYzQkTvkQO-78R0x_1HQE0Tx', connect=True)

# Create a Cloudant database
my_database = client.create_database('diabetic-retinopathy')

# Default route (Home page)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index.html', methods=["GET","POST"])
def home():
    return render_template('index.html')



# Registration page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        x = [x for x in request.form.values()]
        print(x)
        data = {
            '_id': x[1],
            'name': x[0],
            'psw': x[2]    
        }
        print(data)

        query = {'_id': {'seq': data['_id']}}
        docs = my_database.get_query_result(query)
        print(docs)
        print(len(docs.all()))

        if (len(docs.all()) == 0):
            url = my_database.create_document(data)
            return render_template('register.html', pred="Registration Successful, please login using your details")
        else:
            return render_template('register.html', pred="You are already a member, please login using your details")
    return render_template('register.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form.get('id')
        passw = request.form.get('psw')
        print(user, passw)

        query = {'_id': {'$eq': user}}
        docs = my_database.get_query_result(query)
        print(docs)
        print(len(docs.all()))

        if len(docs.all()) == 0:
            return render_template('login.html', pred="The username is not found")
        else:
            if user == docs[0][0]['_id'] and passw == docs[0][0]['psw']:
                return render_template('prediction.html',pred="Logged in as")
            else:
                return render_template('login.html', pred="Invalid credentials")
    return render_template('login.html')

# Logout page
@app.route('/logout')
def logout():
    return render_template('logout.html')


@app.route('/predict', methods=["POST"])
def predict():
    return render_template('input_file.html')

# Prediction page (for diabetic retinopathy detection)
@app.route("/output", methods=["POST"])
def output():
    if request.method == "POST":
        f = request.files['file']

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(file_path)
        print(request.files)



        # basepath = os.path.dirname(f)
        # filepath = os.path.join(basepath, 'static', 'User_Images', str(f.filename))
        # f.save(filepath)
        # filename = secure_filename(f.filename)
        
        

        # Preprocess the image
        img = image.load_img(file_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)

        # Make prediction
        prediction = np.argmax(model.predict(img_data), axis=1)
        index = ['No Diabetic Retinopathy', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR']
        result = index[prediction[0]]
        print(result)
        return render_template('output.html', prediction=result)
    

# Run the app
if __name__ == "__main__":
    try:
        app.run(debug=True, port=5001, use_reloader=False)  # Changed port to avoid conflicts and disable reloader
    except Exception as e:
        print("Error while starting the Flask app:", e)
