# ========== (c) JP Hwang 1/2/21  ==========

import traceback
import dash_design_kit as ddk
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from app import app, snap
from ids import IDS
import pages


server = app.server
celery_instance = snap.celery_instance

theme = {"breakpoint_stack_blocks": "1000px"}

learn_more_menu = ddk.CollapsibleMenu(
    title="Learn More",
    default_open=False,
    children=[
        html.A(
            "Low-code Design",
            href="https://plotly.com/dash/design-kit/",
            target="_blank",
        ),
        html.A(
            "Snapshot Engine",
            href="https://plotly.com/dash/snapshot-engine/",
            target="_blank",
        ),
        html.A("Enterprise Demo", href="https://plotly.com/get-demo/", target="_blank"),
        html.A("Request Code", href="https://plotly.com/contact-us/", target="_blank"),
    ],
)

header = ddk.Header(
    [
        ddk.Logo(src=app.get_relative_path("/assets/logo.png")),
        ddk.Title([ddk.Icon(icon_name="basketball-ball"), " Shot Chart Explorer"]),
        ddk.Menu(
            [
                html.Div(
                    id="button-container",
                    children=html.Span(
                        [
                            html.Span(
                                id="take-snapshot-status", style={"paddingRight": 10}
                            ),
                            html.Button("Take Snapshot", id="take-snapshot"),
                        ],
                        style={"paddingRight": 10},
                    ),
                ),
                dcc.Link(href=app.get_relative_path("/"), children="Home"),
                dcc.Link(href=app.get_relative_path("/archive"), children="Archive"),
                learn_more_menu,
            ]
        ),
    ]
)

app.layout = ddk.App(
    theme=theme,
    show_editor=True,
    children=[
        header,
        dcc.Location(id=IDS["LOCATION"]),
        html.Div(id=IDS["SNAPSHOT_ID"]),
    ],
)


@app.callback(
    [Output(IDS["SNAPSHOT_ID"], "children"), Output("button-container", "style")],
    [Input(IDS["LOCATION"], "pathname")],
)
def display_content(pathname):
    button_style = {"hidden": {"width": 0, "visibility": "hidden"}, "displayed": None}
    page_name = app.strip_relative_path(pathname)
    if not page_name:  # None or ''
        return [pages.home.layout(), button_style["displayed"]]
    elif page_name == "archive":
        return [pages.archive.layout(), button_style["hidden"]]
    elif page_name.startswith("snapshot-"):
        return [pages.snapshot.layout(page_name), button_style["hidden"]]
    else:
        return ["404", button_style["hidden"]]


@app.callback(
    Output("take-snapshot-status", "children"),
    [Input("take-snapshot", "n_clicks")],
    [
        State(IDS["HEADER-AT-A-GLANCE"], "title"),
        State(IDS["GRAPH-TEAM-SIM"], "figure"),
        State(IDS["GRAPH-SHOT-SUM"], "figure"),
        State(IDS["GRAPH-SHOTCHART-1"], "figure"),
        State(IDS["GRAPH-SHOTCHART-2"], "figure"),
        State(IDS["GRAPH-SHOTCHART-FILT"], "figure"),
        State(IDS["GRAPH-VAR-IMPACT"], "figure"),
        State(IDS["GRAPH-VAR-DETAIL"], "figure"),
        State(IDS["FILT-PL"], "value"),
        State(IDS["FILT-DEF"], "value"),
        State(IDS["FILT-CLOCK"], "value"),
        State(IDS["FILT-DRIBBLE"], "value"),
    ],
    prevent_initial_call=True,
)
def save_snapshot(
    n_clicks,
    header_at_glance,
    fig_team_sim,
    fig_shot_sum,
    fig_shotchart_1,
    fig_shotchart_2,
    fig_shotchart_filt,
    fig_var_impact,
    fig_var_detail,
    filt_pl,
    filt_def,
    filt_clock,
    filt_dribble,
):
    try:
        # snap.snapshot_save(children)
        snap.snapshot_save_async(
            save_snapshot_in_background,
            header_at_glance,
            fig_team_sim,
            fig_shot_sum,
            fig_shotchart_1,
            fig_shotchart_2,
            fig_shotchart_filt,
            fig_var_impact,
            fig_var_detail,
            filt_pl,
            filt_def,
            filt_clock,
            filt_dribble,
        )
        return "Saved!"
    except Exception as e:
        traceback.print_exc()
        return "An error occurred saving this snapshot"


@snap.celery_instance.task
@snap.snapshot_async_wrapper(
    save_pdf=True, pdf_page_size="A5", pdf_orientation="landscape"
)
def save_snapshot_in_background(
    header_at_glance,
    fig_team_sim,
    fig_shot_sum,
    fig_shotchart_1,
    fig_shotchart_2,
    fig_shotchart_filt,
    fig_var_impact,
    fig_var_detail,
    filt_pl,
    filt_def,
    filt_clock,
    filt_dribble,
):
    # This function is called in a separate task queue managed by celery
    # This function's parameters are
    # provided by the callback above with `snap.snapshot_save_async`

    # Whatever is returned by this function will be saved to the database
    # with the `snapshot_id`. It needs to be JSON-serializable

    # This dataframe is loaded by `snapshot.layout` and transformed
    # into a set of `ddk.Report` & `ddk.Page` components.
    # This allows you to change your `ddk.Report` & `ddk.Page` reports
    # for older datasets.

    # You could also return a `ddk.Report` etc here if you want previously
    # saved reports to not change when you deploy new changes to your
    # `ddk.Report` layout code
    return {
        "header-at-glance": header_at_glance,
        "fig-team-sim": fig_team_sim,
        "fig-shot-sum": fig_shot_sum,
        "fig-shotchart-1": fig_shotchart_1,
        "fig-shotchart-2": fig_shotchart_2,
        "fig-shotchart-filt": fig_shotchart_filt,
        "fig-var-impact": fig_var_impact,
        "fig-var-detail": fig_var_detail,
        "filt-pl": filt_pl,
        "filt-def": filt_def,
        "filt-clock": filt_clock,
        "filt-dribble": filt_dribble,
    }


if __name__ == "__main__":

    app.run_server(debug=False)
