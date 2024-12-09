import dash_html_components as html

from app import snap


def layout():
    return html.Div([snap.ArchiveTable()])
