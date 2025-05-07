
$(document).ready(function() {

    // const inputTable = $('#inputTable').DataTable({
    //     dom: 'Bfrtip',
    //     buttons: ['colvis'],
    //     // initComplete: function () {
    //     //     this.api().columns().every(function () {
    //     //         let column = this;

    //     //         let inputMin = $('<input type="number" placeholder="min">')
    //     //             .appendTo($(column.header()).find(".minFilter"))
    //     //             .on('keyup change', function () {
    //     //                 const min = parseFloat($(this).val());
    //     //                 filtered = column.data().toArray().map(parseFloat).filter(val => val >= min);
    //     //                 column.data() = filtered;
    //     //                 inputTable.draw();
    //     //                 //column.search($(this).val(), false, false).draw();
    //     //             });
    //     //         let inputMax = $('<input type="number" placeholder="max">')
    //     //             .appendTo($(column.header()).find(".maxFilter"))
    //     //             .on('keyup change', function () {
    //     //                 column.search($(this).val(), false, false).draw();
    //     //             });
    //     //     });
    //     // }
    // });

    // const targetTable = $('#targetTable').DataTable({
    //     dom: 'Bfrtip',
    //     buttons: ['colvis'],
    //     // initComplete: function () {
    //     //     this.api().columns().every(function () {
    //     //         let column = this;
    //     //         let inputMin = $('<input type="number" placeholder="Filter">')
    //     //             .appendTo($(column.header()))
    //     //             .on('keyup change', function () {
    //     //                 column.search($(this).val(), false, false).draw();
    //     //             });
                
                    
    //     //     });
    //     // }
    // });


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

                    $("#model_deleting").removeClass("is-hidden");
                    $("#model_deleting").addClass("is-active");
                } else {
                    $("#model_loading").addClass("is-hidden");
                    $("#model_loading").removeClass("is-active");

                    $("#model_deleting").addClass("is-hidden");
                    $("#model_deleting").removeClass("is-active");
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
                    const modelType =  form_data.get("prompt_model_type");
                    const modelName = form_data.get("prompt_model_version");
                    const profileName = res.profile;
                    var href = 'model_type=' + modelType + '&model_name=' + modelName + '&profile_name=' + profileName;
                    var load = '/engine/inference/?' + href;
                    const path = window.location.pathname.split("/")[3];
                    var del = '/prompting/model-delete/?' + href + '&path=' + path;
                    
                    $("#model_loading").removeClass("is-hidden");
                    $("#model_loading").addClass("is-active");

                    $("#model_deleting").removeClass("is-hidden");
                    $("#model_deleting").addClass("is-active");

                    $("#model_loading").attr("href", load)
                    $("#model_deleting").attr("href", del)
                } else {
                    $("#model_loading").addClass("is-hidden");
                    $("#model_loading").removeClass("is-active");

                    $("#model_deleting").addClass("is-hidden");
                    $("#model_deleting").removeClass("is-active");
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
                
                var form_data = new FormData($("#settingsForm")[0]);
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

    const initialValues = {};

    $('#settingsForm').find('input, select, textarea').each(function () {
        const name = $(this).attr('name');
        if (name) {
            if ($(this).attr('type') === 'checkbox') {
                initialValues[name] = $(this).prop('checked');
            } else {
                initialValues[name] = $(this).val();
            }
        }
    });
    
    $('#settingsForm').on('change input', 'input, select, textarea', function () {
        let changed = false;

        $('#settingsForm').find('input, select, textarea').each(function () {
            const name = $(this).attr('name');
            if (!name) return;

            let current;
            if ($(this).attr('type') === 'checkbox') {
                current = $(this).prop('checked');
            } else {
                current = $(this).val();
            }

            if (current !== initialValues[name]) {
                changed = true;
                $('#skip').prop('disabled', !changed);
            }
        });
        $('#saveUpdate').prop('disabled', !changed);
        $('#noSave').prop('disabled', !changed);
        $('#saveAs').prop('disabled', !changed);
        $('#skip').prop('disabled', changed);
    });

    const mlpParameters = [$("#batch"), $("#epoch"), $("#loss_function"), $("#learning_rate")];
    function updateIputs() {
        if ($("#model_type").val() === "mlp") {
            mlpParameters.forEach(param => param.prop("disabled", false));
        } else {
            mlpParameters.forEach(param => param.prop("disabled", true));
        }
        
    }

    updateIputs();
    $("#model_type").on("change", updateIputs);


    $('#saveUpdate, #noSave, #skip, #saveAs').click(function() {
        const profile = $('#currentProfile').val();

        if (profile === 'unknownprofile' || $(this).attr('id') === 'saveAs') {
            $('#namePrompt').slideDown();
            return; // Stop here until user provides a name
        }

        var form_data = new FormData($("#settingsForm")[0]);
        if ($(this).is('#saveUpdate, #saveAs')) {
            form_data.append("save", 1);
        } else {
            form_data.append("save", 0);
        }
        
        $.ajax({
            url: "/modeling/dataset/settings/",
            type: 'POST',
            data: form_data,
            contentType: false,
            processData: false,
            cache: false,
            success: function(res) {
                console.log(res);
                if (res.err) {
                    $('#err').text(res.err);
                }
                if (res.status === "skip") {
                    window.location.href = "/engine/?profile=" + res.profile;
                } else {
                    window.location.href = "/modeling/dataset/settings/" + res.profile;
                }
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

    $(document).bind("ajaxSend", function () {
        $("#overlay").show();
    }).bind("ajaxComplete", function () {
        $("#loading img").hide();
        $("#overlay").hide();
    });

    $('#model_save').click(function() {
        
        var form_data = new FormData($('#save_form')[0]);
        const modelName = $('#model_name').val();
        const profileName = $('#profile_name').val();
        const modelType = $('#model_type').val();
        
            $.ajax({
                url: '/engine/model-save/',
                type: 'POST',
                data: form_data,
                contentType: false,
                processData: false,
                cache: false,
                async: true,
                success: function(res) {
                    console.log(res);
                    if (res.status=="error") {
                        $('#err').text(res.alert);
                    } else {
                        var hRef = '/engine/inference/?model_type=' + modelType + '&model_name=' + res.alert + '&profile_name=' + profileName; 
                        $('#naming-bar').addClass("is-hidden");
                        $('#naming-bar').removeClass("is-active");
                        $("#flow").append("<p class='small text-muted'>model saved.<p>");
                        $("#flow").append("<a href='/engine/test' target='_blank' class='small text-muted'>Test your model<a>");
                        $("#flow").append(
                            "<a href=" + hRef + " target='_blank' class='small text-muted'>Make an inference<a>");
                    }
                },
                error: function(jqXHR, textStatus, errorMessage) {
                    alert('Error saving weights: ' + errorMessage);
                }
            });

    });



});
