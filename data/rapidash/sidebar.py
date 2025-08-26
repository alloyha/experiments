import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output

# Create the Dash application
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    # Side menu
    html.Div([
        html.H2('Menu'),
        dcc.Link('Page 1', href='/page-1'),
        html.Br(),
        dcc.Link('Page 2', href='/page-2'),
    ], style={'width': '20%', 'float': 'left'}),

    # Main content
    html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content')
    ], style={'width': '80%', 'float': 'right'})
])

# Define callback to update page content based on URL
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/page-1':
        return html.H1('Page 1 content')
    elif pathname == '/page-2':
        return html.H1('Page 2 content')
    else:
        return html.H1('404 - Page not found')

# Run the Dash application
if __name__ == '__main__':
    app.run_server(debug=True)
