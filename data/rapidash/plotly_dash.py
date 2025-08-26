import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    "x": [1, 2, 3, 4, 5],
    "y": [10, 20, 30, 40, 50]
})

# Create a Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Graph(figure=px.line(df, x="x", y="y", title="Line Chart"))
])

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)