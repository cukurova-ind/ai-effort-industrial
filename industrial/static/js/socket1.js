const socket = new WebSocket('ws://' + window.location.host + '/ws/engine/');

$(document).ready(function() {

    socket.onmessage = (e) => {
        result = JSON.parse(e.data).result;
        var res = $("#results").val()
        res += "Server: " + result + "\n";
        $("#results").val(res);
    }

    socket.onclose = (e) => {
        console.log("Socket closed!");
    }

    $('#exp').on("keyup", function(e){
        if (e.keyCode === 13) {
            $('#submit ').click();
        }
    });

    $('#submit').on("click", function(){
        var exp = $('#exp').val();
        socket.send(JSON.stringify(
                    {
                        expression: exp
                    }
                ))

        var res = $("#results").val(); 
        res += "You: " + exp + "\n";
        $("#results").val(res);
        $('#exp').val("");
    });


});