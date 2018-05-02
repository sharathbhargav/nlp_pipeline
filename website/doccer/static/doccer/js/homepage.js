function pleasewait(){
    var modal = document.getElementById('wait_modal');
    modal.style.display = "block";
};

function pleasewaitoffline(){
    var modal = document.getElementById('wait_offline_modal');
    modal.style.display = "block";
};

var filedir = '';

function filelist(event) {
    let files = event.target.files;
    var file = files[0].webkitRelativePath;
    filedir = file.toString().split('/')[0];
};

function submitoffline(){
    pleasewaitoffline();
    var url = "http://127.0.0.1:8000/doccer/offline/" + filedir;
    location.href = url;
};