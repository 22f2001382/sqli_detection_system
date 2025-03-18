from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Client(db.Model):
    client_id = db.Column(db.String(16), primary_key=True)  
    client_name = db.Column(db.String(255), nullable=False)  
    device_type = db.Column(db.Enum("edge", "cloud", "local"), nullable=False) 
    status = db.Column(db.Enum("active", "inactive"), nullable=False)  
    ip_address = db.Column(db.String(45), nullable=False)  
    api_key_hash = db.Column(db.String(64), nullable=False)  

class Update(db.Model):
    update_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    client_id = db.Column(db.String(16), db.ForeignKey("client.client_id"), nullable=False)  
    file_path = db.Column(db.String(512), nullable=True)  
    type = db.Column(db.Enum("train", "load", "update"), nullable=False) 
    status = db.Column(db.Enum("pending", "completed"), nullable=False) 
    updated_data = db.Column(db.JSON, nullable=True) 
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())  