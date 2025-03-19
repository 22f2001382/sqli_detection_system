DataShield is a robust, privacy-preserving ML model designed to detect SQL injection attacks. It leverages federated learning with a Support Vector Machine (SVM) as the core algorithm. Federated learning addresses challenges associated with centralized data collection in traditional methods, enhancing security and data privacy.

**Tech Stack**
**Backend**: Flask (web application)
**Frontend**: HTML, CSS, JavaScript
**ML & Data Processing**: Scikit-learn, Pandas, NumPy

**Running the Project Locally**
After cloning the repository, execute the following commands:
  pip install -r requirements.txt
  flask run --app app
  
**Deployment**
The project has also been hosted on an AWS EC2 instance.
http://ec2-13-48-29-63.eu-north-1.compute.amazonaws.com:5000
