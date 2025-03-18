
from flask import Flask, request, render_template, jsonify,session,send_file
import pandas as pd
import numpy as np
import logging,joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from threading import Lock
import os,time,json,secrets
from config import Config
from model import db,Client,Update
import ApiKeyClass
from DataValidator import DataValidator
from flask_migrate import Migrate


app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)
migrate = Migrate(app, db)
app.secret_key = "encryption_key"
global_model_lock = Lock()

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# Logging setup
logging.basicConfig(filename="static/app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load dataset
data = pd.read_csv("./new.csv")
data = data.assign(Sentence=data['Sentence'].fillna(""))
data = data.dropna(subset=['Label'])

X = data['Sentence']
y = data['Label']

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Initialize SVM classifier
svm_classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, max_iter=1000, tol=1e-3, random_state=42)
svm_classifier.fit(X_vectorized, y)

# Store connected users
connected_users = {}

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/client_register',methods=['POST','GET'])
def register_client():
    if request.method=='GET':
        return render_template('client_registration.html')
    data = request.json
    client_id = secrets.token_hex(8)

    # Generate and hash the API key
    api_key = ApiKeyClass.generate_api_key() if data.get("apiKeyRequest") == "yes" else None
    api_key_hash = ApiKeyClass.hash_api_key(api_key) 


    new_client = Client(
        client_id=client_id,
        client_name=data["clientName"],
        device_type=data["deviceType"],
        status=data["status"],
        ip_address=data["ipAddress"],
        api_key_hash=api_key_hash  # Store hashed API key
    )

    db.session.add(new_client)
    db.session.commit()

    return jsonify({
        "message": "Client registered successfully",
        "client_id": client_id,
        "api_key": api_key  # Show API key only once
    }), 201

@app.route('/client_login',methods=['GET','POST'])
def client_login():
    if request.method=='GET':
        return render_template('client_login.html')
    data=request.json
    api_key=data['apiKey']
    client_id=data['clientId']
    client=Client.query.filter_by(client_id=client_id).first()
    if not client:
        return jsonify({"message": "Invalid Client ID"}), 401
    if not ApiKeyClass.validate_api(api_key,client.api_key_hash):
        return jsonify({"message": "Invalid API Key"}), 401
    session['client_id']=client_id
    return jsonify({"message":"Login Successful"}),200

@app.route('/admin_login',methods=['GET','POST'])
def admin_login():
    if request.method=='GET':
        return render_template('admin_login.html')
    data=request.json
    password=data['password']
    adminId=data['adminId']
    if adminId!=ADMIN_USERNAME or password!=ADMIN_PASSWORD:
        return jsonify({"message": "Invalid Login"}), 401
    return jsonify({"message":"Login Successful"}),200


@app.route('/client_home',methods=['GET'])
def client_home():
    client_id=session.get('client_id')
    updates=Update.query.filter_by(client_id=client_id).all()
    return render_template('client_home.html',requests=updates)

@app.route('/admin_home',methods=['GET'])
def admin_home():
    pending_requests=Update.query.filter_by(status='pending').all()
    completed_requests=Update.query.filter_by(status='completed').all()
    return render_template('admin_home.html',pending_requests=pending_requests,completed_requests=completed_requests)

@app.route('/train_model_on_client_data', methods=['POST'])
def train_model_on_client_data():
    if 'csvFile' not in request.files:
        return "No file part", 400
    file = request.files['csvFile']
    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        client_id=session.get('client_id','')
        new_update = Update(
        client_id=client_id,
        file_path=file_path,
        type="train",
        status="pending",
        )
        db.session.add(new_update)
        db.session.commit()
        return jsonify({
        "message": "Request Created Successfully",
        }), 201 
    return jsonify({"message": "Request Failed"}), 401


@app.route('/validate-data/<int:update_id>',methods=['POST'])
def validate_data(update_id):
    try:
        update=Update.query.filter_by(update_id=update_id).first()
        validator = DataValidator(update.file_path)
        is_valid, validation_message = validator.run_validations()
        response={'success':is_valid,'message':validation_message}
        return jsonify(response)
    except Exception as e:
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

def train_model(file):
        start_time = time.time()
        df = pd.read_csv(file)        
        new_X_text = df['query'].fillna("")
        new_y = df['label']
        new_X_vectorized = vectorizer.transform(new_X_text)
        
        temp_classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, max_iter=1000, tol=1e-3, random_state=42)
        temp_classifier.fit(new_X_vectorized, new_y)
        update_duration = time.time() - start_time
        client_coef = temp_classifier.coef_
        client_intercept = temp_classifier.intercept_
        response=update_model(client_coef,client_intercept)
        if response['status']=='update successful':
            with open("model_metrics.json", "r") as file:
                data = json.load(file)
            
            data["avg_time"]=update_duration
            
            with open("model_metrics.json", "w") as file:
                json.dump(data, file, indent=4)
            return True
        return False

@app.route('/request_load_model',methods=['POST'])
def request_load_model():
    try:
        client_id=session.get('client_id','')
        new_update=Update(
            client_id=client_id,
            type='load',
            status='pending'
        )
        db.session.add(new_update)
        db.session.commit()
        return jsonify({
        "message": "Request Created Successfully",
        }), 201 
    except Exception as e:
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500


@app.route('/load_model',methods=['GET'])
def load_model():
    try:
        joblib.dump((vectorizer, svm_classifier), "model.pkl")
        file_path = os.path.abspath("model.pkl")  # Get absolute path
        return send_file(file_path, as_attachment=True, download_name="model.pkl")
    except Exception as e:
        return jsonify({"success": False, "message": f"Error loading model: {str(e)}"}), 500

@app.route('/request_update_model', methods=['POST'])
def request_update_model():
    if 'jsonFile' not in request.files:
        return "No file part", 400
    file = request.files['jsonFile']
    if file and file.filename.endswith('.json'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        client_id=session.get('client_id','')
        new_update = Update(
        client_id=client_id,
        file_path=file_path,
        type="update",
        status="pending",
        )
        db.session.add(new_update)
        db.session.commit()
        return jsonify({
        "message": "Request Created Successfully",
        }), 201 
    return jsonify({"message": "Request Failed"}), 401

@app.route('/validate-update/<int:update_id>', methods=['POST'])
def validate_update(update_id):
    try:
        update = Update.query.filter_by(update_id=update_id).first()
        file_path = update.file_path

        with open(file_path, "r") as file:
            update_data = json.load(file)

        if "coefficients" not in update_data or "intercept" not in update_data:
            return jsonify({"success": False, "message": "Missing 'coefficients' or 'intercept' keys"}), 400

        client_coef = np.array(update_data["coefficients"])
        client_intercept = np.array(update_data["intercept"])

        # Validate numerical lists
        if not isinstance(client_coef, np.ndarray) or not isinstance(client_intercept, np.ndarray):
            return jsonify({"success": False, "message": "Coefficients and intercept must be numerical lists"}), 400

        # Validate shape compatibility
        server_coef_shape = svm_classifier.coef_.shape
        if client_coef.shape != server_coef_shape:
            return jsonify({"success": False, "message": f"Incorrect coefficient shape. Expected {server_coef_shape}, got {client_coef.shape}"}), 400

        return jsonify({"success": True, "message": "Validation successful", "update_id": update_id})

    except Exception as e:
        return jsonify({"success": False, "message": f"Validation error: {str(e)}"}), 500

def update_model(client_coef,client_intercept):
    global svm_classifier
    
    with global_model_lock:
        # Federated averaging
        server_coef = svm_classifier.coef_
        server_intercept = svm_classifier.intercept_
        
        new_coef = (server_coef + client_coef) / 2
        new_intercept = (server_intercept + client_intercept) / 2
        
        svm_classifier.coef_ = new_coef
        svm_classifier.intercept_ = new_intercept
    (accuracy,precision)=get_model_parameters()
    with open("model_metrics.json", "r") as file:
        data = json.load(file)
    data["accuracy"].append(round(accuracy,5)*100)
    data["precision"].append(round(precision,5)*100)
    with open("model_metrics.json", "w") as file:
        json.dump(data, file, indent=4)

    response={'status':'update successful'}
    return response

@app.route('/accept-request/<int:update_id>',methods=['POST'])
def accept_request(update_id):
    try:
        update=Update.query.filter_by(update_id=update_id).first()
        response=False
        if update.type=='train':
            response=train_model(update.file_path)

        elif update.type=='update':
            update=Update.query.filter_by(update_id=update_id).first()
            file_path=update.file_path
            with open(file_path,'r') as file:
                update_data=json.load(file)
            client_coef=np.array(update_data["coefficients"])
            client_intercept=np.array(update_data["intercept"])
            temp_response=update_model(client_coef,client_intercept)
            if temp_response['status']=='update successful':
                response=True
        else:
            response=True
        if response:
            update.status='completed'
            db.session.commit()
        return jsonify({'success':response})
    except Exception as e:
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500


@app.route('/analyze_query',methods=['GET'])
def analyze_query():
    return render_template('analyze_query.html')

def detect_sql_injection(query, classifier):
    query_vectorized = vectorizer.transform([query])
    prediction = classifier.predict(query_vectorized)
    return "Malicious SQL Injection Attempt" if prediction == '1' else "Benign SQL Query"

@app.route('/newpage', methods=['GET', 'POST'])
def newpage():
    with open('model_metrics.json','r') as file:
        data=json.load(file)
    data['no_of_queries']=data['no_of_queries']+1
    with open('model_metrics.json','w') as file:
        json.dump(data,file,indent=4)
    user_input = request.form['user_input']
    svm_result = detect_sql_injection(user_input, svm_classifier)
    return render_template('result.html', result=svm_result)

def requests_share(type):
    no_of_requests=Update.query.filter_by(type=type).count()
    no_of_updates=Update.query.count()
    share_percentage=(no_of_requests/no_of_updates)*100
    return share_percentage

@app.route('/model_metrics',methods=['GET'])
def model_metrics():
    no_of_clients=Client.query.count()
    no_of_updates=Update.query.count()
    (train,load,update)=(requests_share('train'),requests_share('load'),requests_share('update'))
    y_pred = svm_classifier.predict(X_vectorized)
    recall = recall_score(y, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=1)
    with open('model_metrics.json','r') as file:
        data=json.load(file)
    data['accuracy']=data['accuracy'][-4:]
    data['precision']=data['precision'][-4:]
    data['avg_time']=round(data['avg_time'],5)
    updated_data = {
        **data,  # Preserve existing data
        "no_of_clients": no_of_clients,
        "no_of_updates": no_of_updates,
        "train": train,
        "load": load,
        "update1": update,
        "recall": round(recall,5),
        "f1Score": round(f1,5)
    }
    print(updated_data)
    return render_template("model_metrics.html",data=updated_data)


def get_model_parameters():
    """Return current model parameters"""
    with global_model_lock:
        y_pred = svm_classifier.predict(X_vectorized)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=1)
    return (accuracy,precision)


# @app.route('/metrics', methods=['GET'])
def get_metrics():
    y_pred = svm_classifier.predict(X_vectorized)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=1)   
    # Calculate additional metrics
    data_coverage = len(X.dropna()) / len(X) * 100  # Coverage as percentage of non-null training data
    malicious_query_count = sum(1 for label in y if label == '0')  # Assuming '0' indicates malicious queries
    total_queries = len(y)
    malicious_query_percent = (malicious_query_count / total_queries) * 100 if total_queries > 0 else 0

    # Example of calculating query size and duration
    query_sample = "SELECT * FROM users WHERE username = 'test'"  # Example query, update this based on actual use
    query_size = len(query_sample.encode('utf-8')) / 1024  # Query size in KB
    
    # Simulate update duration (example for illustration)

    # Your update model code here (simulate some operation)
    time.sleep(1)  # Simulating model update time


    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "dataCoverage": data_coverage,
        "maliciousQueryPercent": malicious_query_percent,
        "querySize": query_size    }
    print(metrics)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)



