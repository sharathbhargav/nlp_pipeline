function load(data) {
    var filenames = data['fnames'];
    var X = [], Y = [], colors = [];
    var org = [], person = [], place = [], loc = [], noun = [], summary = [];
    var i = 0;
    for (i=0 ; i<filenames.length ; i++){
        X[i] = data[filenames[i]]['xy'][0];
        Y[i] = data[filenames[i]]['xy'][1];
        colors[i] = data[filenames[i]]['color'];
        org[i] = data[filenames[i]]['org_entities'];
        person[i] = data[filenames[i]]['person_entities'];
        place[i] = data[filenames[i]]['place_entities'];
        loc[i] = data[filenames[i]]['loc_entities'];
        noun[i] = data[filenames[i]]['noun_entities'];
        summary[i] = data[filenames[i]]['summary'];
    };

    var file_entities = [org, person, place, loc, noun];
    var file_entities_names = ['Organizations', 'Persons', 'Places', 'Locations', 'Nouns'];
    var plot_data = [{
        x : X,
        y : Y,
        type : 'scatter',
        mode : 'markers',
        marker : {size : 16, color : colors}
    }];

    var layout = {
        hovermode : 'closest',
        title : 'A Cluster of Documents'
    };

    var plot = document.getElementById('plot');
    var entities = document.getElementById('named_entities_text');
    var entities_fname = document.getElementById('named_entities_filename');

    Plotly.newPlot(plot, plot_data, layout);

    plot.on('plotly_hover', function(data){
        var pn = '';
        for(var i=0; i < data.points.length; i++){
            pn = data.points[i].pointNumber;
        };
        var file = filenames[pn];

        var infoText = '';
        for (var i=0 ; i<file_entities.length ; i++){
            infoText = infoText + '<b>' +file_entities_names[i] + '</b> : ';
            for(var j=0 ; j<file_entities[i][pn].length ; j++){
                infoText = infoText + file_entities[i][pn][j] + ', ';
            };
            infoText += '<hr />';
        };
        infoText += '<b>Summary</b> : ';
        for (var i=0 ; i<summary[pn].length ; i++){
            infoText += summary[pn][i] + ' ';
        };
        entities_fname.innerHTML = file;
        entities.innerHTML = infoText;
    });

    plot.on('plotly_unhover', function(data){
        //entities.innerHTML = '<filename>';
        //entities_fname.innerHTML = '<entities>';
    });

    plot.on('plotly_click', function(data){
        var pn = '';
        for(var i=0; i < data.points.length; i++){
            pn = data.points[i].pointNumber;
        };
        var file = filenames[pn];
        var url = 'http://127.0.0.1:8000/doccer/displaydoc/' + file.replace(new RegExp('/', 'g'), '+');
        var win = window.open(url, '_blank');
        win.focus();
    });
}