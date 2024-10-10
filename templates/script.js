const fileInput = document.getElementById('file');
const uploadedImage = document.getElementById('uploadedImage');
const checkButton = document.getElementById('checkButton');

document.getElementById('uploadButton').addEventListener('click', function() {
    const file = fileInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            uploadedImage.src = e.target.result;
            uploadedImage.style.display = 'block';
            checkButton.style.display = 'block';
        }
        reader.readAsDataURL(file);
    }
});

checkButton.addEventListener("click", function() {
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const predictionResult = document.getElementById("predictionResult");
        predictionResult.innerHTML = `Morphology: ${data.prediction}`;
    })
    .catch(error => console.error("Error:", error));
});
