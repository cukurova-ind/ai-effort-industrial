$(document).ready(function() {

    const userName = JSON.parse($("#userName").text());
    const userEmail = JSON.parse($("#userEmail").text());
    const profileName = JSON.parse($("#profileName").text());
    const modelName = JSON.parse($("#modelName").text());
    const modelType = JSON.parse($("#modelType").text());
    
    function connect() {
        inferenceSocket = new WebSocket("ws://" + window.location.host + "/ws/inference/" + userName + "/");
    
        inferenceSocket.onopen = function(e) {
            console.log("Successfully connected to the inference.");
            inferenceSocket.send(JSON.stringify({
                "message": "loadModel",
                "username": userName,
                "email": userEmail,
                "profilename": profileName,
                "modelname": modelName,
                "modeltype": modelType
            }));
        }
    
        inferenceSocket.onclose = function(e) {
            console.log("WebSocket connection closed unexpectedly. Trying to reconnect in 2s...");
            setTimeout(function() {
                console.log("Reconnecting...");
                connect();
            }, 2000);
        };
    
        inferenceSocket.onmessage = function(e) {
            
            const data = JSON.parse(e.data);
            if (data.type=="operation_message") {
                $("#flow").append("<p class='small text-muted'>" + data.message + "</p>")
            }
            if (data.type=="inference_message") {
                if (data.event==="simple_gan") {
                    $("#shot_image").html("<img src='/" + data.dir + "/prediction_" + data.prediction + ".png'/>")
                } else {
                    $('#shot_pred').html('<p>' + data.prediction + '</p>');
                }
                
            }

            $("#loading img").hide();
            $("#overlay").hide();
    
        };
    
        inferenceSocket.onerror = function(err) {
            console.log("WebSocket encountered an error: " + err.message);
            console.log("Closing the socket.");
            inferenceSocket.close();
        }
    }
    connect();

    $("#inference-form-button").click(function() {

        const fileInput = document.getElementById("id_raw_image");
        if (fileInput) {
            const formData = new FormData();
            formData.append("raw_image", fileInput.files[0]);

            fetch("upload-image/", {
                method: "POST",
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                console.log("Buffered file path:", data.file_path);

                var form_data = new FormData($('#feature_form')[0]);
                const formObject = {};
                form_data.forEach((value, key) => {
                    formObject[key] = value;
                });
                formObject["input_image"] = data.file_path;
                $("#overlay").show();
                inferenceSocket.send(JSON.stringify({
                    "message": "data",
                    "data": formObject
                }));

            })
            .catch(error => {
                console.error("Upload failed:", error);
            });

        } else {

            var form_data = new FormData($('#feature_form')[0]);
            const formObject = {};
            form_data.forEach((value, key) => {
                formObject[key] = value;
            });

            $("#overlay").show();
            inferenceSocket.send(JSON.stringify({
                "message": "data",
                "data": formObject
            }));

        }

    });


});
