$(document).ready(function() {

    const userName = JSON.parse($("#userName").text());
    const userEmail = JSON.parse($("#userEmail").text());
    const profileName = JSON.parse($("#profileName").text());
    
    function connect() {
        engineSocket = new WebSocket("ws://" + window.location.host + "/ws/engine/" + userName + "/");
    
        engineSocket.onopen = function(e) {
            console.log("Successfully connected to the WebSocket.");
        }
    
        engineSocket.onclose = function(e) {
            console.log("WebSocket connection closed unexpectedly. Trying to reconnect in 2s...");
            setTimeout(function() {
                console.log("Reconnecting...");
                connect();
            }, 2000);
        };
    
        engineSocket.onmessage = function(e) {
            const data = JSON.parse(e.data);
            if (data.type=="operation_message") {
                $("#flow").append("<p class='small text-muted'>" + data.message + "</p>")
            }
            if (data.type=="train_message") {
                
                if (data.event=="step_ten"){
                    $('#step').html('<p class="small text-muted">Step: ' + data.step + '/' + data.steps + '</p>');
                    $('#step-prog').css("width", data.percentage + '%').attr("aria-valuenow", data.percentage + '%').text(data.percentage + '% completed');
                    $('#score').html('<p class="small text-muted">Loss: ' + data.loss + '</p>');
                }

                if (data.event=="epoch_end") {
                    $('#epoch').html('<p class="small text-muted">Epoch: ' + data.epoch + '/' + data.epochs + '</p>');
                    $('#epoch-prog').css("width", data.percentage + '%')
                            .attr("aria-valuenow", data.percentage)
                            .text(data.percentage + '% completed');
                    $('#score').html('<p class="small text-muted">Epoch Loss: ' + data.epoch_loss + '</p>');
                }

                if (data.event=="validation"){
                    $('#val-message').html('<p class="small text-muted">Performance over validation set:</p><hr>');
                    var loss_html = '<p class="small text-muted">MAPE (%): ' + data.val_mape + '</p>';
                    loss_html += '<p class="small text-muted">MAE: ' + data.val_mae + '</p>';
                    loss_html += '<p class="small text-muted">MSE: ' + data.val_mse + '</p>';
                    $('#loss').html(loss_html);
                }
                
            }
    
        };
    
        engineSocket.onerror = function(err) {
            console.log("WebSocket encountered an error: " + err.message);
            console.log("Closing the socket.");
            engineSocket.close();
        }
    }
    connect();

    $("#start").on("click", function(){
        engineSocket.send(JSON.stringify({
            "message": "start-train-" + userName + "-" + userEmail + "-" + profileName
        }));
    });


});
