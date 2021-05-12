function performAPI() {
        var files = document.getElementById("audio_file").files
        var formData = new FormData();
        var endpoint = 'gender/route_gender';

        formData.append('audio', files[0])

        $.ajax({
            type: 'POST',
            url: endpoint,
            data: formData,
            contentType: false,
            cache: false,
            processData: false,
            success : function(data){
                console.log("Upload successful",data);
            },
            error: function(e){
                console.log("Upload failed", e);
            }
        });

//        var xhr = new XMLHttpRequest();
//        xhr.open("POST", endpoint, true);
//        xhr.onreadystatechange = function() {
//            if(xhr.readyState == 4 && xhr.status == 200) {
//                var response = null;
//                try {
//                    response = JSON.parse(xhr.responseText);
//                } catch (e) {
//                    response = xhr.responseText;
//                }
//                uploadFormCallback(response);
//            }
//        }
//        xhr.send(formData);
}

