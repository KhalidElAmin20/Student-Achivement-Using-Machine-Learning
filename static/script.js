document.getElementById('prediction-form').addEventListener('submit', function(e) {
    e.preventDefault();

    var formData = new FormData(this);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    

    .then(data => {
            document.getElementById('prediction-result').innerText = 'Predicted Grade: ' + data.prediction;
            
    })
    
    .catch(error => {
        console.error('Error:', error);
    });
});
