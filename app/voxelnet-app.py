import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np

app = dash.Dash(__name__)

# Initial camera settings
camera_state = {
    'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}
}

app.layout = html.Div([
    html.H1("MUIA UAX 2023, 2024"),
    html.P("VOXELNet demonstration app."),
    
    dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': 'Non-inverted', 'value': 'non-inverted'},
            {'label': 'Inverted', 'value': 'inverted'}
        ],
        value='option1'
    ),
    
    html.Div(id='dropdown-output'),

    html.Div([
        html.Div([
            html.H2("Left Webcam"),
            html.Img(id='webcam-preview-left', src='http://127.0.0.1:5001/video_feed_0', style={'width': '100%', 'height': 'auto'})
        ], style={'width': '16%', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        html.Div([
            html.H2("Right Webcam"),
            html.Img(id='webcam-preview-right', src='http://127.0.0.1:5001/video_feed_1', style={'width': '100%', 'height': 'auto'})
        ], style={'width': '16%', 'display': 'inline-block', 'vertical-align': 'top'}),
    ]),

    dcc.Graph(id='3d-plot', config={'scrollZoom': True}),
    
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0),  # Update every 50 ms

    html.Div(id='camera-state', style={'display': 'none'}),
    
    dcc.Store(id='store-data'),
])

@app.callback(
    dash.dependencies.Output('store-data', 'data'),
    [dash.dependencies.Input('3d-plot', 'relayoutData')],
    [dash.dependencies.State('store-data', 'data')]
)
def capture_camera_state(relayout_data, current_camera_state):
    if relayout_data and 'scene.camera' in relayout_data:
        camera = relayout_data['scene.camera']
        camera_state.update(camera)  # Update global camera_state
        print(f"camera_state updated to {camera_state}")
    return camera_state

@app.callback(
    dash.dependencies.Output('3d-plot', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals')],
    [dash.dependencies.State('store-data', 'data')]
)
def update_graph(n, camera_state):
    print("UPDATE GRAPH")  # This will be printed every time the callback is triggered
    x = np.random.randn(100)
    y = np.random.randn(100)
    z = np.random.randn(100)
    
    trace = go.Scatter3d(x=x, y=y, z=z, mode='markers')
    
    layout = go.Layout(
        scene=dict(
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1]),
            camera=camera_state
        )
    )
    
    return {'data': [trace], 'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True, port=8555)
