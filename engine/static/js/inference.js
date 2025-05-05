$(document).ready(function() {
    alert("ok");
    // const userName = JSON.parse($("#userName").text());
    // const userEmail = JSON.parse($("#userEmail").text());
    // //const profileName = JSON.parse($("#profileName").text());
    
    // function connect() {
    //     inferenceSocket = new WebSocket("ws://" + window.location.host + "/ws/inference/" + userName + "/");
    
    //     inferenceSocket.onopen = function(e) {
    //         console.log("Successfully connected to the inference.");
    //     }
    
    //     inferenceSocket.onclose = function(e) {
    //         console.log("WebSocket connection closed unexpectedly. Trying to reconnect in 2s...");
    //         setTimeout(function() {
    //             console.log("Reconnecting...");
    //             connect();
    //         }, 2000);
    //     };

    //     // inferenceSocket.send(JSON.stringify({
    //     //     "message": "loadModel-" + userName + "-" + userEmail + "-" + "unknown"
    //     // }));
    
    //     inferenceSocket.onmessage = function(e) {
    //         const data = JSON.parse(e.data);
    //         if (data.type=="operation_message") {
    //             $("#flow").append("<p class='small text-muted'>" + data.message + "</p>")
    //         }
    //         if (data.type=="train_message") {
                
    //             if (data.event=="step_ten"){
    //                 $('#step').html('<p class="small text-muted">Step: ' + data.step + '/' + data.steps + '</p>');
    //                 $('#step-prog').css("width", data.percentage + '%').attr("aria-valuenow", data.percentage + '%').text(data.percentage + '% completed');
    //                 $('#score').html('<p class="small text-muted">Loss: ' + data.loss + '</p>');
    //             }
                
    //         }
    
    //     };
    
    //     inferenceSocket.onerror = function(err) {
    //         console.log("WebSocket encountered an error: " + err.message);
    //         console.log("Closing the socket.");
    //         inferenceSocket.close();
    //     }
    // }
    // connect();


});
