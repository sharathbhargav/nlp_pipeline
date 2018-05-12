function load(data) {
    var fnames = data['fnames'];
    var table = document.getElementById('similarities_table_body');
    for (var i=0 ; i<fnames.length ; i++){
        var row_data = "<tr><th scope='row'>" + fnames[i] + "</th>";
        row_data = row_data + "<td>" + data[fnames[i]][0] + "</td>";
        row_data = row_data + "<td>" + data[fnames[i]][1] + "</td>";
        row_data = row_data + "<td>" + data[fnames[i]][2] + "</td>";
        row_data = row_data + "<td>" + data[fnames[i]][3] + "</td>";
        var newrow = table.insertRow(table.rows.length);
        newrow.innerHTML = row_data;
    };

    var dis_table = document.getElementById('dissimilarities_table_body');
    for (var i=0 ; i<fnames.length ; i++){
        var indx = 'dis_' + i.toString();
        var row_data = "<tr><th scope='row'>" + data[indx][0] + "</th><th>" + data[indx][1] + "</th>";
        row_data = row_data + "<td>" + data[indx][2] + "</td>";
        row_data = row_data + "<td>" + data[indx][3] + "</td>";
        row_data = row_data + "<td>" + data[indx][4] + "</td>";
        row_data = row_data + "<td>" + data[indx][5] + "</td>";
        var newrow = dis_table.insertRow(dis_table.rows.length);
        newrow.innerHTML = row_data;
    };
}