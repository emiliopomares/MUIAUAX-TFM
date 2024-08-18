import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np
import threading
from model import get_model, get_test_batch
from inference import run_inference_thread

TEST_DATASET_PATH = "/media/emilio/2TBDrive/robovision_test" # Point to your test data

inference_model = get_model()
test_data = get_test_batch()

app = dash.Dash(__name__)

x_data = np.zeros(1)
y_data = np.zeros(1)
z_data = np.zeros(1)

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
    
    # html.Div(id='dropdown-output'),

    # html.Div([
    #     html.Div([
    #         html.H2("Left Webcam"),
    #         html.Img(id='webcam-preview-left', src='http://127.0.0.1:5001/video_feed_0', style={'width': '100%', 'height': 'auto'})
    #     ], style={'width': '16%', 'display': 'inline-block', 'vertical-align': 'top'}),
        
    #     html.Div([
    #         html.H2("Right Webcam"),
    #         html.Img(id='webcam-preview-right', src='http://127.0.0.1:5001/video_feed_1', style={'width': '100%', 'height': 'auto'})
    #     ], style={'width': '16%', 'display': 'inline-block', 'vertical-align': 'top'}),
    # ]),
    html.Div(id='dropdown-output'),

    html.Div([
        html.Div([
            html.H2("Left Webcam"),
            html.Img(id='webcam-preview-left', src='', style={'width': '100%', 'height': 'auto', 'aspect-ratio': '1.7777'})
        ], style={'width': '24%', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        html.Div([
            html.H2("Right Webcam"),
            html.Img(id='webcam-preview-right', src='', style={'width': '100%', 'height': 'auto', 'aspect-ratio': '1.7777'})
        ], style={'width': '24%', 'display': 'inline-block', 'vertical-align': 'top'}),
    ]),

    dcc.Graph(id='3d-plot', config={'scrollZoom': True}),
    
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0),  # Update every 50 ms

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

# Callback to update the webcam feed URLs based on the dropdown selection
@app.callback(
    [Output('webcam-preview-left', 'src'),
     Output('webcam-preview-right', 'src')],
    [Input('dropdown', 'value')]
)
def update_webcam_feed(selected_option):
    base_url_left = "http://127.0.0.1:5001/video_feed_0"
    base_url_right = "http://127.0.0.1:5001/video_feed_1"
    
    # Add the selected option as a query parameter
    left_url = f"{base_url_left}?mode={selected_option}"
    right_url = f"{base_url_right}?mode={selected_option}"
    print(left_url)
    
    return left_url, right_url

@app.callback(
    dash.dependencies.Output('3d-plot', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals')],
    [dash.dependencies.State('store-data', 'data')]
)
def update_graph(n, camera_state):
    global x_data
    global y_data
    global z_data
    print("UPDATE GRAPH")  # This will be printed every time the callback is triggered
    x = y_data # Yet another tweezing of dimensions!
    y = x_data
    z = z_data
    
    trace = go.Scatter3d(
        x=x, 
        y=y, 
        z=z, 
        mode='markers',
        marker=dict(
            size=5,  # Adjust the size of the markers
            color=y,  # Use the z values for color mapping
            colorscale='Viridis',  # Choose a colorscale (e.g., 'Viridis', 'Cividis', 'Plasma')
            colorbar=dict(
                title="Depth (z)",  # Title for the color bar
                thickness=15,
                len=0.5,  # Adjusts the length of the color bar
            ),
            opacity=0.8  # Adjust transparency for better visibility
        )
    )
    
    layout = go.Layout(
        scene=dict(
            xaxis=dict(range=[0, 25]),
            yaxis=dict(range=[0, 35]),
            zaxis=dict(range=[0, 17]),
            camera=camera_state
        )
    )
    
    return {'data': [trace], 'layout': layout}

def gather_results(x_arr, y_arr, z_arr):
    global x_data
    global y_data
    global z_data
    print(f"Done gathering results len: {x_arr.shape}")
    x_data = x_arr
    y_data = y_arr
    z_data = z_arr

if __name__ == '__main__':
    run_inference_thread(inference_model, test_data[0], gather_results)
    app.run_server(debug=True, port=5000)
