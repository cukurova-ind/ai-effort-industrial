$(document).ready(function() {

    const userName = JSON.parse($("#userName").text());
    const userEmail = JSON.parse($("#userEmail").text());
    const profileName = JSON.parse($("#profileName").text());
    const retrainingModelName = JSON.parse($("#retrainingModelName").text());
        

    function connect() {
        engineSocket = new WebSocket("ws://" + window.location.host + "/ws/engine/" + userName + "/");
    
        engineSocket.onopen = function(e) {
            console.log("Successfully connected to the WebSocket.");
            if (retrainingModelName) {
                console.log(retrainingModelName);
                engineSocket.send(JSON.stringify({
                    "message": "reload",
                    "username": userName,
                    "email": userEmail,
                    "profilename": profileName,
                    "retrainingmodelname": retrainingModelName,
                }));
            }
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
                $("#flow").append("<p class='small text-muted'>" + data.message + "</p>");
                if (data.event=="imagesave"){
                    $(".sample-images").html("<img src='" + data.savedir + "prediction_" + data.timestamp + ".png'/>")
                }
            }

            if (data.type=="train_message") {
                
                if (data.event=="step_ten"){
                    $('#step').html('<p class="small text-muted">Step: ' + data.step + '/' + data.steps + '</p>');
                    $('#step-prog').css("width", data.percentage + '%').attr("aria-valuenow", data.percentage + '%').text(data.percentage + '% completed');
                    if (data.loss.d_loss) {
                        $('#score').html('<p class="small text-muted">Loss: D: ' + data.loss.d_loss +  " G: " + data.loss.g_loss + '</p>');
                    } else {
                        $('#score').html('<p class="small text-muted">Loss: ' + data.loss + '</p>');
                    }
                    
                }

                if (data.event=="epoch_end") {
                    $('#epoch').html('<p class="small text-muted">Epoch: ' + data.epoch + '/' + data.epochs + '</p>');
                    $('#epoch-prog').css("width", data.percentage + '%')
                            .attr("aria-valuenow", data.percentage)
                            .text(data.percentage + '% completed');
                    if (data.epoch_loss.d_loss) {
                        $('#score').html('<p class="small text-muted">Epoch Loss: D: ' + data.epoch_loss.d_loss +  " G: " + data.epoch_loss.g_loss + '</p>');
                    } else {
                        $('#score').html('<p class="small text-muted">Epoch Loss: ' + data.epoch_loss + '</p>');
                    }
                    
                }

                if (data.event=="validation"){
                    $('#val-message').html('<p class="small text-muted">Performance over validation set:</p><hr>');
                    if (data.clips) {
                        var loss_html = '<p class="small text-muted">CLIP: ' + data.clips + '</p>';
                        loss_html += '<p class="small text-muted">RGB Distance: ' + data.rgbd + '</p>';
                    } else {
                        var loss_html = '<p class="small text-muted">MAPE (%): ' + data.val_mape + '</p>';
                        loss_html += '<p class="small text-muted">MAE: ' + data.val_mae + '</p>';
                        loss_html += '<p class="small text-muted">MSE: ' + data.val_mse + '</p>';
                    }
                    $('#loss').html(loss_html);
                }

                if (data.event=="test"){
                    $('#test-message').html('<p class="small text-muted">Performance over test set:</p><hr>');
                    if (data.test_clip) {
                        var loss_html = '<p class="small text-muted">CLIP: ' + data.test_clip + '</p>';
                        loss_html += '<p class="small text-muted">RGB Distance: ' + data.test_rgbd + '</p>';
                    } else {
                        var loss_html = '<p class="small text-muted">MAPE (%): ' + data.test_mape + '</p>';
                        loss_html += '<p class="small text-muted">MAE: ' + data.test_mae + '</p>';
                        loss_html += '<p class="small text-muted">MSE: ' + data.test_mse + '</p>';
                    }
                    $('#loss_test').html(loss_html);
                }
                
            }

            if (data.event === "stopped") {
                //$("#start").prop("disabled", false);
                $("#stop").prop("disabled", true);
                $("#naming-bar").removeClass("is-hidden"); // Show naming bar
                engineSocket.send(JSON.stringify({
                    "message": "test",
                }));
                if (retrainingModelName) {
                    $("#model_name").val(retrainingModelName);
                    $("#model_name").prop("readonly", true);
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

    $("#stop").prop("disabled", true);
    $("#start").on("click", function(){
        $(this).prop("disabled", true);
        $("#stop").prop("disabled", false);
        engineSocket.send(JSON.stringify({
            "message": "startTrain",
            "username": userName,
            "email": userEmail,
            "profilename": profileName,
        }));
    });

    window.addEventListener("beforeunload", function (e) {
        // Show confirmation dialog
        const confirmationMessage = "Training is still running. Are you sure you want to leave?";
        
        e.preventDefault(); // Standard in some browsers
    
        return confirmationMessage; // For older browsers
    });

    window.addEventListener("unload", function () {
        if (engineSocket.readyState === WebSocket.OPEN) {
            engineSocket.send(JSON.stringify({
                "message": "stop"
            }));
        }
    });

    $("#stop").click(function() {
        engineSocket.send(JSON.stringify({
            "message": "stop"
        }));
    });


});
