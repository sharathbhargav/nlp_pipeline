function load(data) {
    var fname_1 = data['fname_1'];
    var fname_2 = data['fname_2'];
    var similarity = data['similarity'];
    var f1_points = data['f1_points'];
    var f2_points = data['f2_points'];
    var common_points = data['common_points'];

    // co-ordinates
    var f1_X = [], f1_Y = [];
    var f2_X = [], f2_Y = [];
    var c_X = [], c_Y = [];
    var f1_words = [], f2_words = [], common_words = [];

    for (var i=0 ; i<f1_points.length ; i++){
        f1_words[i] = f1_points[i][0];
        f1_X[i] = f1_points[i][1][0];
        f1_Y[i] = f1_points[i][1][1];
    };

    for (var i=0 ; i<f2_points.length ; i++){
        f2_words[i] = f2_points[i][0];
        f2_X[i] = f2_points[i][1][0];
        f2_Y[i] = f2_points[i][1][1];
    };

    for (var i=0 ; i<common_points.length ; i++){
        common_words[i] = common_points[i][0];
        c_X[i] = common_points[i][1][0];
        c_Y[i] = common_points[i][1][1];
    };

    var plot_data = [{
        x : f1_X,
        y : f1_Y,
        type : 'scatter',
        mode : 'markers',
        marker : {size : 8, color : 'rgb(255, 97, 25)'},
        text : f1_words,
        name : fname_1.split('/').slice(-1)[0]
    },{
        x : f2_X,
        y : f2_Y,
        type : 'scatter',
        mode : 'markers',
        marker : {size : 8, color : 'rgb(0, 178, 158)'},
        text : f2_words,
        name : fname_2.split('/').slice(-1)[0]
    },{
        x : c_X,
        y : c_Y,
        mode : 'markers',
        marker : {size : 8, color : 'rgb(0, 0, 0)'},
        text : common_words,
        name : 'commons'
    }];

    var layout = {
        hovermode : 'closest',
        title : 'Words in the Documents'
    };

    var plot = document.getElementById('plot');
    window.alert(similarity);
    Plotly.newPlot(plot, plot_data, layout);
}