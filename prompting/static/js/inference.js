$(document).ready(function() {
    
    $("#generate-form-button").on("click", function () {

        const fileInput = $('#id_raw_image')[0].files[0];

        if (fileInput) {
            const formData = new FormData();
            formData.append('file', fileInput);

            $.ajax({
                url: 'http://127.0.0.1:5000/inference-page',
                type: 'POST',
                data: formData,
                processData: false,
                contentType: false,
                success: function(response) {
                    console.log(response);
                    // var html = "<h5>Input</h5>";
                    // html += "<img src='" + response.input + "'>";
                    
                    // $('#shot-image').prepend(html);
                },
                error: function(jqXHR, textStatus, errorMessage) {
                    alert('Error uploading file: ' + errorMessage);
                }
            });

        } else {
            alert('No file selected');
        }
    });


});