
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

    const $checkboxes = $('.input_columns');
    const $outputField = $('#id_selection');
    
    function updateSelectedColumns() {
        
        const selected = $checkboxes
            .filter(':checked')
            .map(function () {
                return $(this).data("label");
            })
            .get();
        $outputField.val(selected.join(', '));
    }

    // Init on page load
    updateSelectedColumns();

    // Update on change
    $checkboxes.on('change', updateSelectedColumns);


    $('#checkAllBtn').click(function() {
        $('.input_columns').prop('checked', true);
        updateSelectedColumns()
    });

    $('#uncheckAllBtn').click(function() {
        $('.input_columns').prop('checked', false);
        updateSelectedColumns();
    });

    const $outboxes = $('.output_columns');
    const $selectedField = $('#id_selection_output');
    
    function updateSelected() {
        
        const selected = $outboxes
            .filter(':checked')
            .map(function () {
                return $(this).data("label");
            })
            .get();
        $selectedField.val(selected.join(', '));
    }

    // Init on page load
    updateSelected();

    // Update on change
    $outboxes.on('change', updateSelected);

    $('#saveUpdate').click(function() {
        const profile = $('#currentProfile').val();

        if (profile === 'unknownprofile') {
            $('#namePrompt').slideDown();
            return; // Stop here until user provides a name
        }

        var form_data = new FormData($("#settingsForm")[0]);
        form_data.append("saveupdate", 1)
        $.ajax({
            url: "/modeling/dataset/settings/",
            type: 'POST',
            data: form_data,
            contentType: false,
            processData: false,
            cache: false,
            success: function(res) {
                console.log(res);
                window.location.href = "/engine/" + res.profile + "/";
            },
            error: function(jqXHR, textStatus, errorMessage) {
                alert(errorMessage);
            }
        });
    });

    $('#newProfileName').on('change', function () {
        const newName = $(this).val().trim();
        if (newName !== '') {
            $('#currentProfile').val(newName);
            $('#namePrompt').slideUp();
            $('#saveUpdate').click(); // Retry save
        }
    });

    $('#noSave').click(function() {
        var form_data = new FormData($("#settingsForm")[0]);

        $.ajax({
            url: "/modeling/dataset/settings/",
            type: 'POST',
            data: form_data,
            contentType: false,
            processData: false,
            cache: false,
            success: function(res) {
                console.log(res);
                window.location.href = "/engine/";
            },
            error: function(jqXHR, textStatus, errorMessage) {
                alert(errorMessage);
            }
        });
    });


});
