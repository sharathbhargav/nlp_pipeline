function load(data) {
    var filenames = data['fnames'];
    var X = [], Y = [], colors = [];
    var i = 0;
    for (i=0 ; i<filenames.length ; i++){
        //document.write("x = " + data[filenames[i]]['xy'][0] + " ; y = " + data[filenames[i]]['xy'][1] + '<br>');
        X[i] = data[filenames[i]]['xy'][0];
        Y[i] = data[filenames[i]]['xy'][1];
        colors[i] = data[filenames[i]]['color']
    }

    var plot_data = [{
        x : X,
        y : Y,
        type : 'scatter',
        mode : 'markers',
        marker : {size : 16, color : colors}
    }];

    var layout = {
        hovermode : 'closest',
        title : 'Trial Clustering'
    };

    var plot = document.getElementById('plot');
    var entities = document.getElementById('named_entities');

    //Dummy Stuff
    var dummy_named_entities = ['Apple', 'Google', 'Modi', 'Eggs', 'Sopranos'];

    Plotly.newPlot(plot, plot_data, layout);

    plot.on('plotly_hover', function(data){
        var pn = '';
        for(var i=0; i < data.points.length; i++){
            pn = data.points[i].pointNumber;
        };
        var file = filenames[pn];

        //Creating a dummy info out of dummy entities
        var infoText = '';
        infoText = infoText + file + '<br>';
        for (var i=0 ; i<dummy_named_entities.length ; i++){
            infoText = infoText + dummy_named_entities[i] + ', ';
        };
        entities.innerHTML = infoText;
    });

    plot.on('plotly_unhover', function(data){
        var infoText = '';
        entities.innerHTML = infoText;
    });
}