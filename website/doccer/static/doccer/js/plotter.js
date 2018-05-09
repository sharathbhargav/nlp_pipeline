function load(data) {
    var filenames = data['fnames'];
    var X = [], Y = [], colors = [];
    var org = [], person = [], place = [], loc = [], noun = [], summary = [];
    for (var i=0 ; i<filenames.length ; i++){
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
    var entities_names = ['Organizations', 'Persons', 'Places', 'Locations', 'Nouns'];

    var n_clusters = data['n_clusters'];
    var cX = [], cY = [], cColors = [];
    var c_org = [], c_person = [], c_place = [], c_loc = [], c_noun = [], c_summary = [];
    for (var i=0 ; i<n_clusters ; i++){
        cX[i] = data[i.toString()]['xy'][0];
        cY[i] = data[i.toString()]['xy'][1];
        cColors[i] = data[i.toString()]['color'];
        c_org[i] = data[i.toString()]['org_entities'];
        c_person[i] = data[i.toString()]['person_entities'];
        c_place[i] = data[i.toString()]['place_entities'];
        c_loc[i] = data[i.toString()]['loc_entities'];
        c_noun[i] = data[i.toString()]['noun_entities'];
        c_summary[i] = data[i.toString()]['summary'];
    };
    var cluster_entities = [c_org, c_person, c_place, c_loc, c_noun];

    var plot_data = [{
        x : X,
        y : Y,
        type : 'scatter',
        mode : 'markers',
        marker : {size : 14, color : colors},
        name : 'Files'
    },
    {
        x : cX,
        y : cY,
        type : 'scatter',
        mode : 'markers',
        marker : {size : 22, color : cColors},
        name : 'Clusters'
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
        var cn = '';
        for(var i=0; i < data.points.length; i++){
            pn = data.points[i].pointNumber;
            cn = data.points[i].curveNumber;
        };
        var file = filenames[pn];

        var infoText = '';
        var entities_array;
        if (cn == 0){
            entities_array = file_entities;
            entities_fname.innerHTML = file;
        }
        else if (cn == 1){
            entities_array = cluster_entities;
            entities_fname.innerHTML = 'Cluster ' + pn;
        }
        for (var i=0 ; i<entities_array.length ; i++){
            infoText = infoText + '<b>' + entities_names[i] + '</b> : ';
            for(var j=0 ; j<entities_array[i][pn].length ; j++){
                infoText = infoText + entities_array[i][pn][j] + ', ';
            };
            infoText += '<hr />';
        };
        infoText += '<b>Summary</b> : ';
        for (var i=0 ; i<summary[pn].length ; i++){
            infoText += summary[pn][i] + ' ';
        };
        entities.innerHTML = infoText;
    });

    plot.on('plotly_unhover', function(data){
        //entities.innerHTML = '<filename>';
        //entities_fname.innerHTML = '<entities>';
    });

    plot.on('plotly_click', function(data){
        var pn = '';
        var cn = ''
        for(var i=0; i < data.points.length; i++){
            pn = data.points[i].pointNumber;
            cn = data.points[i].curveNumber;
        };
        if (cn == 0){
            var file = filenames[pn];
            var url = 'http://127.0.0.1:8000/doccer/displaydoc/' + file.replace(new RegExp('/', 'g'), '+');
            var win = window.open(url, '_blank');
            win.focus();
        }
    });
}