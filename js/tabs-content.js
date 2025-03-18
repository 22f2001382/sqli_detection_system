document.addEventListener("DOMContentLoaded", function () {
    var firstTab = new bootstrap.Tab(document.querySelector(".nav-link.active"));
    firstTab.show();
});
document.getElementById("train-model-content").innerHTML = `
    <p>Our system enables clients to enhance the accuracy of our SQL injection detection model by training it on their own data. The central server hosts an SVM-based model, and clients can upload their datasets to refine its performance for their specific needs. Before training, all submitted data undergoes a validation check to prevent malicious inputs. Once verified, the data is used to further generalize and improve the model. This approach ensures a more adaptive and robust detection system while maintaining security and efficiency.</p>
    <p>Please upload your data below for training</p>
    <form action="{{url_for('train_model_on_client_data')}}" method="POST">
    <label for="csvFile">Upload CSV File:</label>
    <input type="file" id="csvFile" name="csvFile" accept=".csv" required>
    <button type="submit">Train Model</button>
    </form>
`;

document.getElementById("load-model-content").innerHTML = `
    <p>Profile Information</p>
    <input type="text" class="form-control" placeholder="Enter your name">
    <button class="btn btn-success mt-2">Save</button>
`;

document.getElementById("update-model-content").innerHTML = `
    <p>Contact us at:</p>
    <ul>
        <li>Email: example@example.com</li>
        <li>Phone: 123-456-7890</li>
    </ul>
`;
document.getElementById("requests-list-content").innerHTML = `
    <p>Contact us at:</p>
    <ul>
        <li>Email: example@example.com</li>
        <li>Phone: 123-456-7890</li>
    </ul>
`;

document.getElementById("update-model-content").innerHTML = `
    <p>Contact us at:</p>
    <ul>
        <li>Email: example@example.com</li>
        <li>Phone: 123-456-7890</li>
    </ul>
`;
document.getElementById("model-metrics-content").innerHTML = `
    <p>Contact us at:</p>
    <ul>
        <li>Email: example@example.com</li>
        <li>Phone: 123-456-7890</li>
    </ul>
`;
