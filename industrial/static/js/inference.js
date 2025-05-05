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
                $('#shot_pred').html('<p>' + data.prediction + '</p>');
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
    });


});
