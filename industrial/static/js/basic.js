// function handleFileSelect(e) {
//     var files = e.target.files;
//     if (files.length < 1) {
//         alert('select a file...');
//         return;
//     }
//     var file = files[0];
//     var reader = new FileReader();
//     reader.onload = onFileLoaded;
//     reader.readAsDataURL(file);
// }

// function onFileLoaded(e) {
//     var match = /^data:(.*);base64,(.*)$/.exec(e.target.result);
//     if (match == null) {
//         throw 'Could not parse result'; // should not happen
//     }
//     var mimeType = match[1];
//     var content = match[2];
//     alert(mimeType);
//     alert(content);
// }

// $(function () {
//     $('#import-pfx-button').click(function (e) {
//         $('#file-input').click();
//     });
//     $('#file-input').change(handleFileSelect);
// });

// document.getElementById("filepicker").addEventListener(
//     "change",
//     (event) => {
//       let output = document.getElementById("listing");
//       for (const file of event.target.files) {
//         let item = document.createElement("li");
//         item.textContent = file.webkitRelativePath;
//         output.appendChild(item);
//       }
//     },
//     false,
//   );

$(document).ready(function() {
    $("#max_steps").on("change", function(){
        if ($("#max_steps").attr("checked")) {
            $("#step").removeAttr("disabled");
            $("#max_steps").removeAttr("checked");
        } else {
            $("#step").attr("disabled", "true");
            $("#max_steps").attr("checked", "true");
        }
    });

    $("#prompt_type").on("change", function(){

        var form_data = new FormData($("#model_selection")[0]);
        $.ajax({
            url: "/prompting/selection-change/",
            type: 'POST',
            data: form_data,
            contentType: false,
            processData: false,
            cache: false,
            success: function(res) {

                if (res.type_options.length>0){
                    to = res.type_options;
                    $("#prompt_model_type").empty();
                    $("#prompt_model_type").append("<option value=''>Seçiniz</option>");
                    for (var i in to) {
                        $("#prompt_model_type").append("<option value='"+ to[i].value +"'>" + to[i].label + "</option>");
                    }
                } else {
                    $("#prompt_model_type").empty();
                    $("#prompt_model_type").append("<option value=''>Seçiniz</option>");
                }

                if (res.versions.length>0){
                    mv = res.versions;
                    $("#prompt_model_version").empty();
                    $("#prompt_model_version").append("<option value=''>Seçiniz</option>");
                    for (var i in mv) {
                        $("#prompt_model_version").append("<option value='"+ mv[i] +"'>" + mv[i] + "</option>");
                    }
                } else {
                    $("#prompt_model_version").empty();
                    $("#prompt_model_version").append("<option value=''>Seçiniz</option>");
                }

                if (res.status){
                    $("#model_loading").removeClass("is-hidden");
                    $("#model_loading").addClass("is-active");
                } else {
                    $("#model_loading").addClass("is-hidden");
                    $("#model_loading").removeClass("is-active");
                }

            },
            error: function(jqXHR, textStatus, errorMessage) {
                alert(errorMessage);
            }
        });
    });

    $("#prompt_model_type").on("change", function(){

        var form_data = new FormData($("#model_selection")[0]);
        $.ajax({
            url: "/prompting/selection-change/",
            type: 'POST',
            data: form_data,
            contentType: false,
            processData: false,
            cache: false,
            success: function(res) {

                if (res.versions.length>0){
                    mv = res.versions;
                    $("#prompt_model_version").empty();
                    $("#prompt_model_version").append("<option value=''>Seçiniz</option>");
                    for (var i in mv) {
                        $("#prompt_model_version").append("<option value='"+ mv[i] +"'>" + mv[i] + "</option>");
                    }
                } else {
                    $("#prompt_model_version").empty();
                    $("#prompt_model_version").append("<option value=''>Seçiniz</option>");
                }

                if (res.status=="complete"){
                    $("#model_loading").removeClass("is-hidden");
                    $("#model_loading").addClass("is-active");
                } else {
                    $("#model_loading").addClass("is-hidden");
                    $("#model_loading").removeClass("is-active");
                }

            },
            error: function(jqXHR, textStatus, errorMessage) {
                alert(errorMessage);
            }
        });
    });

    $("#prompt_model_version").on("change", function(){

        var form_data = new FormData($("#model_selection")[0]);
        $.ajax({
            url: "/prompting/selection-change/",
            type: 'POST',
            data: form_data,
            contentType: false,
            processData: false,
            cache: false,
            success: function(res) {

                if (res.status=="complete"){
                    var q = "model=" + form_data.get("prompt_model_type") + "&version=" + form_data.get("prompt_model_version")
                    var f = $("#for-data").data("for");
                    q += "&for=" + f;
                    var link = "http://127.0.0.1:5000/inference-page?" + q;
                    $("#model_loading").removeClass("is-hidden");
                    $("#model_loading").addClass("is-active");
                    $("#model_loading").attr("href", link)
                } else {
                    $("#model_loading").addClass("is-hidden");
                    $("#model_loading").removeClass("is-active");
                }

            },
            error: function(jqXHR, textStatus, errorMessage) {
                alert(errorMessage);
            }
        });
    });

    var retrain = function() {
        $("#retrain").on("change", function(){
            if ($("#retrain").attr("checked")) {
                $("#saved_model").empty();
                $("#saved_model").attr("disabled", "true");
                $("#retrain").removeAttr("checked");
            } else {
                $("#saved_model").attr("disabled", "true");
                $("#saved_model").removeAttr("disabled");
                $("#retrain").attr("checked", "true");
                
                var form_data = new FormData($("#config-form")[0]);
                $.ajax({
                    url: "/prompting/selection-change/",
                    type: 'POST',
                    data: form_data,
                    contentType: false,
                    processData: false,
                    cache: false,
                    success: function(res) {
        
                        console.log(res);
                        if (res.versions.length>0){
                            mv = res.versions;
                            $("#saved_model").empty();
                            for (var i in mv) {
                                $("#saved_model").append("<option value='"+ mv[i] +"'>" + mv[i] + "</option>");
                            }
                        } else {
                            $("#saved_model").empty();
                        }
                        
                    },
                    error: function(jqXHR, textStatus, errorMessage) {
                        alert(errorMessage);
                    }
                });
    
            }
        });
    };

    retrain();

    $("#model_type").on("change", function(){
        $("#re-train").html("<input class='form-check-input' type='checkbox' role='switch' id='retrain' name='retrain'>");
        $("#saved_model").empty();
        $("#saved_model").attr("disabled", "true");
        retrain();
    });

    // let raw_files = [];
    // let hypo_files = [];
    // let exp_files = [];
    // document.getElementById("raw_folder").addEventListener(
    //     "change",
    //     (event) => {

    //         raw_files = Array.from(event.target.files);
    //         if (raw_files.length === 0) {
    //             alert("No files selected.");
    //             return;
    //         }
    //     },
    //     false,
    // );

    // document.getElementById("hypo_folder").addEventListener(
    //     "change",
    //     (event) => {

    //         hypo_files = Array.from(event.target.files);
    //         if (hypo_files.length === 0) {
    //             alert("No files selected.");
    //             return;
    //         }
    //     },
    //     false,
    // );

    // document.getElementById("exp_folder").addEventListener(
    //     "change",
    //     (event) => {

    //         exp_files = Array.from(event.target.files);
    //         if (exp_files.length === 0) {
    //             alert("No files selected.");
    //             return;
    //         }
    //     },
    //     false,
    // );

    // $("#image_upload_button").on("click", function(){
    //     const form_data = new FormData($("#image_upload")[0]);
    //     // raw_files.forEach(file => {
    //     //     form_data.append("raw_files[]", file, file.webkitRelativePath);
    //     // });
    //     // hypo_files.forEach(file => {
    //     //     form_data.append("hypo_files[]", file, file.webkitRelativePath);
    //     // });
    //     // exp_files.forEach(file => {
    //     //     form_data.append("exp_files[]", file, file.webkitRelativePath);
    //     // });


    //     $.ajax({
    //         url: "/dataops/import/image",
    //         type: 'POST',
    //         data: form_data,
    //         contentType: false,
    //         processData: false,
    //         cache: false,
    //         success: function(res) {
    //             console.log(res);

    //         },
    //         error: function(jqXHR, textStatus, errorMessage) {
    //             alert(errorMessage);
    //         }
    //     });
    // });



});
