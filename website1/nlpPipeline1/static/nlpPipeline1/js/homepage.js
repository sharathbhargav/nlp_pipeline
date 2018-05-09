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
    var url = "offline/" + filedir;
    //console.log("url",url)
    location.href = url;

};