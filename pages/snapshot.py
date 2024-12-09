import dash_design_kit as ddk
import dash_html_components as html

from app import snap


def layout(snapshot_id):
    # This function is called when the snapshot URL is loaded by an end
    # user (displaying the web report) or by the Snapshot Engine's
    # PDF rendering service (when taking a PDF snapshot)
    #
    # The data that was saved by the asynchronous task is loaded and
    # then transformed into a set of `ddk.Report` calls.
    # We're using mock data here just for illustration purposes.
    #
    # You can also save the `ddk.Report` in the task queue instead
    # of just the dataset. Then, you would simply `return snapshot`
    # here. If you saved report, you wouldn't be able to change
    # the layout of the report after it was saved. In this case model,
    # you can update the look and feel of your report in this function
    # _on-the-fly_ when the snapshot is loaded. Note that any changes
    # that you make here won't be reflected in the previously saved PDF
    # version

    snapshot = snap.snapshot_get(snapshot_id)
    header_at_glance = snapshot["header-at-glance"]
    fig_team_sim = snapshot["fig-team-sim"]
    fig_shot_sum = snapshot["fig-shot-sum"]
    fig_shotchart_1 = snapshot["fig-shotchart-1"]
    fig_shotchart_2 = snapshot["fig-shotchart-2"]
    fig_shotchart_filt = snapshot["fig-shotchart-filt"]
    fig_var_impact = snapshot["fig-var-impact"]
    fig_var_detail = snapshot["fig-var-detail"]
    filt_pl = snapshot["filt-pl"]
    filt_def = snapshot["filt-def"]
    filt_clock = snapshot["filt-clock"]
    filt_dribble = snapshot["filt-dribble"]
    return report(
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


def report(
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
    # Generate the report a separate function from the snapshot layout
    # so that you can debug the report under a separate URL

    # Check if we're in dev mode, where the report is generated under the
    # /dev URL just to tweak the report layout
    return ddk.Report(
        display_page_numbers=True,
        children=[
            ddk.Page(
                children=[
                    html.Div(
                        "NBA Shot Location Data Analysis",
                        style={"marginTop": "2in", "fontSize": "28px"},
                    ),
                    html.Div(
                        header_at_glance, style={"marginTop": "2in", "fontSize": "20px"}
                    ),
                    ddk.PageFooter("SHOT CHART APP - REPORT"),
                ],
                style={
                    "backgroundColor": "var(--accent)",
                    "color": "white",
                    "font-family": "Arial, Open Sans",
                },
            ),
            ddk.Page(
                [
                    html.H1("In League Context"),
                    html.P(
                        "These charts place the team in the context of the league as a whole in the year.",
                        style={"fontSize": "15px"},
                    ),
                    ddk.Block(
                        width=100, margin=5, children=[ddk.Graph(figure=fig_team_sim)]
                    ),
                    html.P(
                        "How does the team fare when compared by court zones?",
                        style={"fontSize": "15px"},
                    ),
                    ddk.Block(
                        width=100, margin=5, children=[ddk.Graph(figure=fig_shot_sum)]
                    ),
                ],
                style={"font-family": "Arial, Open Sans"},
            ),
            ddk.Page(
                [
                    ddk.Block(
                        width=25,
                        children=[
                            html.H1("Shot Charts"),
                            html.P(
                                "Compare shot charts for the team against another",
                                style={"fontSize": "15px"},
                            ),
                        ],
                    ),
                    ddk.Block(
                        width=75,
                        margin=5,
                        children=[
                            ddk.Graph(figure=fig_shotchart_1),
                            ddk.Graph(figure=fig_shotchart_2),
                        ],
                    ),
                ],
                style={"font-family": "Arial, Open Sans"},
            ),
            ddk.Page(
                [
                    html.H1("Variable sensitivity analysis"),
                    html.P(
                        "How does each variable, or its subset, impact the shot efficiency of the team?",
                        style={"fontSize": "15px"},
                    ),
                    ddk.Block(
                        width=100, margin=5, children=[ddk.Graph(figure=fig_var_impact)]
                    ),
                    html.P("And again, in further detail", style={"fontSize": "15px"}),
                    ddk.Block(
                        width=100, margin=5, children=[ddk.Graph(figure=fig_var_detail)]
                    ),
                ],
                style={"font-family": "Arial, Open Sans"},
            ),
            ddk.Page(
                [
                    html.H1("Filtered shot chart"),
                    html.P(
                        "Detailed shot chart based on below filters:",
                        style={"fontSize": "15px"},
                    ),
                    html.P(f"Players: {filt_pl}", style={"fontSize": "15px"}),
                    html.P(
                        f"Closest defenders: {filt_def}", style={"fontSize": "15px"}
                    ),
                    html.P(f"Shot Clock: {filt_clock}", style={"fontSize": "15px"}),
                    html.P(f"Dribble: {filt_dribble}", style={"fontSize": "15px"}),
                    ddk.Block(
                        width=50,
                        margin=5,
                        children=[ddk.Graph(figure=fig_shotchart_filt)],
                    ),
                ],
                style={"font-family": "Arial, Open Sans"},
            ),
        ],
    )
