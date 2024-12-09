# ========== (c) JP Hwang 1/2/21  ==========


import os
import pandas as pd
import numpy as np
import random
import json
from scipy.spatial.distance import cosine
import dash_design_kit as ddk
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from flask_caching import Cache
import plotly.express as px
import viz
from app import app
from ids import IDS

scatter_stats = ["acc", "freq"]
col_list = ["shot_clock", "closest_def_dist", "dribble_range"]
var_labels = {
    "shot_clock": "Shot Clock",
    "closest_def_dist": "Closest Defender",
    "dribble_range": "Dribbles",
}
seasons_list = [17, 18, 19, 20, 21]


# ====================================================================
# ========== SET UP DASH APP LAYOUT ==========
# ====================================================================
CACHE_CONFIG = {
    # try 'filesystem' if you don't want to setup redis
    "CACHE_TYPE": "redis",
    "CACHE_REDIS_URL": os.environ.get("REDIS_URL", "redis://localhost:6379"),
}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)


@cache.memoize()
def load_df(yr):
    df_out = pd.read_csv(f"data/proc_stats_20{yr - 1}_20{yr}.csv", index_col=0)
    return df_out


with open("data/team_list.json", "r") as f:
    team_list = json.load(f)


fig1_w = 510
fig2_w = 510
fig3_w = 510


@cache.memoize()
def filter_df(tm, yr, def_dist_vals=None, shot_clock_vals=None, dribble_vals=None):

    df = load_df(yr)

    if tm == "NBA":
        filt_df = df
    else:
        filt_df = df[df.team == tm]

    def filter_by_col(df_in, col_name_in, col_vals_in):
        if col_vals_in is None or len(col_vals_in) == 0:
            df_out = df_in
        else:
            if type(col_vals_in) != list:
                col_vals_in = [col_vals_in]
            df_out = df_in[df_in[col_name_in].isin(col_vals_in)]
        return df_out

    for (col_name, col_vals) in [
        ("closest_def_dist", def_dist_vals),
        ("shot_clock", shot_clock_vals),
        ("dribble_range", dribble_vals),
    ]:
        filt_df = filter_by_col(filt_df, col_name, col_vals)

    return filt_df


@app.callback(
    [
        Output(IDS["FILT-PL"], "options"),
        Output(IDS["FILT-PL"], "value"),
        Output(IDS["FILT-DEF"], "options"),
        Output(IDS["FILT-DEF"], "value"),
        Output(IDS["FILT-CLOCK"], "options"),
        Output(IDS["FILT-CLOCK"], "value"),
        Output(IDS["FILT-DRIBBLE"], "options"),
        Output(IDS["FILT-DRIBBLE"], "value"),
    ],
    [Input("team-select-1", "value"), Input("season-select-1", "value")],
)
def fill_filter_options(tm, yr):
    tm_df = filter_df(tm, yr)
    pl_list = list(
        tm_df.groupby("player").count()["x"].sort_values(ascending=False).index
    )
    return (
        [{"label": p, "value": p} for p in pl_list],
        pl_list[:3],
        [{"label": t, "value": t} for t in np.sort(tm_df.closest_def_dist.unique())],
        [i for i in np.sort(tm_df.closest_def_dist.unique())],
        [{"label": t, "value": t} for t in np.sort(tm_df.shot_clock.unique())],
        [i for i in np.sort(tm_df.shot_clock.unique())],
        [{"label": t, "value": t} for t in np.sort(tm_df.dribble_range.unique())],
        [i for i in np.sort(tm_df.dribble_range.unique())],
    )


def add_logo(fig_in, tm_name, logo_size, xloc, yloc, opacity=0.5):
    fig_in.add_layout_image(
        dict(
            source=f"assets/logos/{tm_name}-2021.png",
            xref="x",
            yref="y",
            xanchor="center",
            yanchor="middle",
            x=xloc,
            y=yloc,
            sizex=logo_size,
            sizey=logo_size,
            sizing="contain",
            layer="above",
            opacity=opacity,
        )
    )
    return fig_in


def get_efg_pct(df_in):
    return np.multiply(df_in.value, df_in.made).sum() / len(df_in) * (0.5 * 100)


def get_freq_df(df_in, col="simple_zone"):
    tmp_ser = df_in.groupby(col).count()["made"] / len(df_in) * 100
    tmp_grp_df = pd.DataFrame(tmp_ser.rename("freq"))
    return tmp_grp_df


def get_acc_df(df_in, col="simple_zone"):
    tmp_ser = (
        df_in.groupby(col).sum()["made"] / df_in.groupby(col).count()["made"] * 100
    )
    tmp_grp_df = pd.DataFrame(tmp_ser.rename("acc"))
    return tmp_grp_df


def get_suffix(n):
    if n == 1:
        return "st"
    elif n == 2:
        return "nd"
    else:
        return "th"


def draw_team_comp_chart(tm, yr):
    df = load_df(yr)

    col = "shot_zone"
    stats = ["acc", "freq"]

    k = ["team", col]
    acc_ser = (df.groupby(k).sum()["made"] / df.groupby(k).count()["made"]).rename(
        "acc"
    )
    freq_ser = (df.groupby(k).count()["made"] / len(df)).rename("freq")
    grp_df = pd.concat([acc_ser, freq_ser], axis=1).reset_index()

    # Calculate similarities
    ref_teams = [t for t in team_list if t != tm]
    dist_dict = dict()
    for stat in stats:
        dist_list = list()
        tm_vec = grp_df.loc[grp_df.team == tm, stat].values
        for ref_tm in ref_teams:
            ref_vec = grp_df.loc[grp_df.team == ref_tm, stat].values
            temp_sim = cosine(tm_vec, ref_vec)
            dist_list.append(temp_sim)
        dist_dict[stat] = dist_list

    dist_df = pd.DataFrame(dist_dict)
    dist_df = dist_df.assign(team=ref_teams)
    dist_df = dist_df.assign(dist=(dist_df.acc ** 2 + dist_df.freq ** 2) ** 0.5)

    fig = px.scatter(
        dist_df,
        x="acc",
        y="freq",
        # text="team", color="team",
        hover_data={"team": True, "acc": ":.3f", "freq": ":.3f"},
        labels={
            "freq": f"Location similarity (up: less similar)",
            "acc": f"Accuracy similarity (right: less similar)",
        },
        height=300,
    )

    fig.update_traces(
        marker=dict(size=1, opacity=0.5, line=dict(width=0, color="navy"))
    )
    fig.update_layout(
        showlegend=False,
        yaxis=dict(ticks="", showticklabels=False, zeroline=True),
        xaxis=dict(ticks="", showticklabels=False, zeroline=True),
        hovermode="closest",
    )

    for tmp_tm in [t for t in team_list if t != tm]:
        fig = add_logo(
            fig,
            tmp_tm,
            0.15 * max(dist_df["acc"].max(), dist_df["freq"].max()),
            xloc=dist_df[dist_df.team == tmp_tm]["acc"].values[0],
            yloc=dist_df[dist_df.team == tmp_tm]["freq"].values[0],
        )
    return fig


@app.callback(
    Output("team-sim-chart-footer", "children"),
    [Input("team-select-1", "value"), Input("team-sim-chart-select", "value")],
    [State("season-select-1", "value")],
)
def update_card_headers(tm, chart_type, yr):

    if chart_type == "eff":
        footer_txt = "Actual scoring efficiency vs theoretical scoring efficiency (based on shot locations)"
    else:
        footer_txt = (
            f"Teams sorted by shot location & accuracy profile similarity to {tm}"
        )

    return footer_txt


@app.callback(
    [
        Output("tm_rating_points", "value"),
        Output("tm_rating_points", "sub"),
        Output("tm_rating_points", "color"),
        Output("tm_rating_location", "value"),
        Output("tm_rating_location", "sub"),
        Output("tm_rating_location", "color"),
        Output("tm_rating_shotmaking", "value"),
        Output("tm_rating_shotmaking", "sub"),
        Output("tm_rating_shotmaking", "color"),
        Output(IDS["HEADER-AT-A-GLANCE"], "title"),
        Output("team-sim-chart-header", "title"),
        Output(IDS["GRAPH-TEAM-SIM"], "figure"),
    ],
    [
        Input("team-select-1", "value"),
        Input("season-select-1", "value"),
        Input("team-sim-chart-select", "value"),
    ],
)
def get_team_ratings(tm, yr, chart_select):

    df = load_df(yr)

    tmp_colscale = px.colors.diverging.Portland_r
    col_divisor = np.ceil(len(team_list) / len(tmp_colscale))

    def get_pps(df_in):
        tmp_pps = (df_in.value * df_in.made).sum() / len(df_in)
        return tmp_pps

    # Get points per shot data
    tmp_df = filter_df(tm, yr)
    tm_pps = get_pps(tmp_df)
    tm_pps_list = [get_pps(filter_df(t, yr)) for t in team_list]
    tm_pps_rank = len([t for t in tm_pps_list if t > tm_pps]) + 1
    tm_pps_txt = f"{tm_pps:.3f}"
    tm_pps_sub = f"({tm_pps_rank}{get_suffix(tm_pps_rank)})"
    tm_pps_col = tmp_colscale[np.int(((tm_pps_rank - 1) // col_divisor))]
    # Get shot location data - Measure how good / bad shot locations are
    league_ev = df.groupby("simple_zone").mean()["res_pts"].rename("league_ev")

    def get_loc_ev(df_in):
        tmp_counts = df_in.groupby("simple_zone").count()["x"].rename("count")
        tmp_loc_df = pd.DataFrame(league_ev).join(tmp_counts)
        tmp_loc_df = tmp_loc_df.assign(
            loc_ev=tmp_loc_df["league_ev"]
            * tmp_loc_df["count"]
            / tmp_loc_df["count"].sum()
        )
        tmp_loc_ev = tmp_loc_df.loc_ev.sum()
        return tmp_loc_ev

    tm_loc_ev = get_loc_ev(tmp_df)
    tm_loc_ev_list = [get_loc_ev(filter_df(t, yr)) for t in team_list]
    tm_loc_ev_rank = len([t for t in tm_loc_ev_list if t > tm_loc_ev]) + 1
    tm_loc_ev_txt = f"{tm_loc_ev:.3f}"
    tm_loc_ev_sub = f"({tm_loc_ev_rank}{get_suffix(tm_loc_ev_rank)})"
    tm_loc_ev_col = tmp_colscale[np.int(((tm_loc_ev_rank - 1) // col_divisor))]

    # Compile data & get relative data
    tm_rank_df = pd.DataFrame(
        {"pps": tm_pps_list, "loc_ev": tm_loc_ev_list, "team": team_list}
    )
    tm_rank_df = tm_rank_df.assign(rel_ev=tm_rank_df.pps - tm_rank_df.loc_ev)

    tm_rel_ev = tm_rank_df[tm_rank_df.team == tm].rel_ev.values[0]
    tm_rel_ev_rank = 1 + len(tm_rank_df[tm_rank_df.rel_ev > tm_rel_ev])
    plus_prefix = "+" if tm_rel_ev > 0 else ""
    tm_rel_ev_txt = f"{plus_prefix}{tm_rel_ev:.3f}"
    tm_rel_ev_sub = f"({tm_rel_ev_rank}{get_suffix(tm_rel_ev_rank)})"
    tm_rel_ev_col = tmp_colscale[np.int(((tm_rel_ev_rank - 1) // col_divisor))]

    if chart_select == "eff":

        min_val = min(tm_rank_df["pps"].min(), tm_rank_df["loc_ev"].min())
        max_val = max(tm_rank_df["pps"].max(), tm_rank_df["loc_ev"].max())
        range_y = tm_rank_df["pps"].max() - tm_rank_df["pps"].min()
        max_y = tm_rank_df["pps"].max()

        fig = px.scatter(
            tm_rank_df,
            x="loc_ev",
            y="pps",
            # color="rel_ev", text="team",
            hover_data={"team": True, "loc_ev": ":.3f", "pps": ":.3f"},
            labels={
                "pps": "Points per shot (PPS)",
                "loc_ev": "Location-based PPS",
                "rel_ev": "Shotmaking<BR>advantage",
            },
            range_x=[
                tm_rank_df["loc_ev"].min() * 0.99,
                tm_rank_df["loc_ev"].max() * 1.01,
            ],
            range_y=[tm_rank_df["pps"].min() * 0.97, tm_rank_df["pps"].max() * 1.03],
            height=300,
        )
        fig.update_traces(
            marker=dict(size=1, opacity=0.5, line=dict(width=0, color="navy"))
        )

        fig.update_layout(
            margin=dict(t=0),
            # yaxis=dict(scaleanchor="x", scaleratio=1),
            hovermode="closest",
            shapes=[
                dict(
                    type="line",
                    x0=0.999 * min_val,
                    y0=0.999 * min_val,
                    x1=1.001 * max_val,
                    y1=1.001 * max_val,
                    line=dict(color="#bbbbbb", width=1, dash="dash"),
                    layer="below",
                )
            ],
        )

        for tmp_tm in [t for t in team_list if t != tm]:
            fig = add_logo(
                fig,
                tmp_tm,
                0.15 * range_y * max_y,
                xloc=tm_rank_df[tm_rank_df.team == tmp_tm]["loc_ev"].values[0],
                yloc=tm_rank_df[tm_rank_df.team == tmp_tm]["pps"].values[0],
                opacity=0.4,
            )
        fig = add_logo(
            fig,
            tm,
            0.3 * range_y * max_y,
            xloc=tm_rank_df[tm_rank_df.team == tm]["loc_ev"].values[0],
            yloc=tm_rank_df[tm_rank_df.team == tm]["pps"].values[0],
            opacity=0.85,
        )
    else:
        fig = draw_team_comp_chart(tm, yr)
        fig.update_layout(margin=dict(t=0))

    glance_header = f"{tm} ('{yr-1}-'{yr}) - At a glance"
    context_header = f"{tm} ('{yr-1}-'{yr}) - In league context"

    return (
        tm_pps_txt,
        tm_pps_sub,
        tm_pps_col,
        tm_loc_ev_txt,
        tm_loc_ev_sub,
        tm_loc_ev_col,
        tm_rel_ev_txt,
        tm_rel_ev_sub,
        tm_rel_ev_col,
        glance_header,
        context_header,
        fig,
    )


@app.callback(
    Output("GRAPH-SHOT-SUM", "figure"),
    [
        Input("team-select-1", "value"),
        Input("season-select-1", "value"),
        Input("team-select-ovr", "value"),
    ],
)
def draw_simple_summary(tm, yr, ref_tms):

    if ref_tms is None:
        ref_tms = ["NBA"]
    elif type(ref_tms) != list:
        ref_tms = list(ref_tms)

    tm_df = filter_df(tm, yr)
    grp_freq_df = get_freq_df(tm_df)
    grp_acc_df = get_acc_df(tm_df)
    grp_df = grp_freq_df.join(grp_acc_df, how="left")
    grp_df = grp_df.assign(group=tm)

    ref_grp_dfs = list()
    for ref_tm in ref_tms:
        ref_df = filter_df(ref_tm, yr)
        ref_freq_df = get_freq_df(ref_df)
        ref_acc_df = get_acc_df(ref_df)
        ref_grp_df = ref_freq_df.join(ref_acc_df, how="left")
        ref_grp_df = ref_grp_df.assign(group=ref_tm)
        ref_grp_dfs.append(ref_grp_df)

    tot_df = pd.concat([grp_df] + ref_grp_dfs).reset_index()

    fig = px.scatter(
        tot_df,
        y="simple_zone",
        x="acc",
        size="freq",
        color="group",
        hover_data={"group": True, "acc": ":.3f", "freq": ":.3f"},
        labels={"group": "Team", "simple_zone": "Area", "acc": "Accuracy (%)"},
        height=320,
    )
    fig.update_traces(
        hovertemplate="%{y}<br>Frequency: %{marker.size:.1f}%<br>Accuracy: %{x:.1f}%"
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="navy")))
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
    )
    return fig


@app.callback(
    Output("GRAPH-SHOTCHART-1", "figure"),
    [Input("team-select-1", "value"), Input("season-select-1", "value")],
)
def get_shotchart_1(tm, yr):
    tmp_df = filter_df(tm, yr)

    title = f"<B>{tm}</B> - {yr - 1}'/{yr}' season<BR><BR>Size: Frequency<BR>Color: Points / 100 shots"

    hexbin_dict = viz.get_hexbin_data(tmp_df, use_zones=True, show_runtime=True)
    fig = viz.draw_shotchart(
        hexbin_dict,
        stat_type="shot_ev",
        mode="white",
        min_freq=0.0005,
        min_hex_samples=2,
        max_freq=0.005,
        title_txt=title,
        legend_title="PTS/<BR>100",
        width=fig1_w - 20,
    )
    fig = add_logo(fig, tm, 50, 160, 375, opacity=0.75)
    fig.update_layout(hovermode="closest")
    return fig


@app.callback(
    Output("GRAPH-SHOTCHART-2", "figure"),
    [Input("team-select-2", "value"), Input("season-select-2", "value")],
)
def get_shotchart_2(tm, yr):
    tmp_df = filter_df(tm, yr)

    title = f"<B>{tm}</B> - {yr - 1}'/{yr}' season<BR><BR>Size: Frequency<BR>Color: Points / 100 shots"

    hexbin_dict = viz.get_hexbin_data(tmp_df, use_zones=True, show_runtime=True)
    fig = viz.draw_shotchart(
        hexbin_dict,
        stat_type="shot_ev",
        mode="white",
        min_freq=0.0005,
        min_hex_samples=2,
        max_freq=0.005,
        title_txt=title,
        legend_title="PTS/<BR>100",
        width=fig2_w - 20,
    )
    fig = add_logo(fig, tm, 50, 160, 375, opacity=0.75)
    fig.update_layout(hovermode="closest")
    return fig


@app.callback(
    Output("GRAPH-SHOTCHART-FILT", "figure"),
    [
        Input("team-select-1", "value"),
        Input("season-select-1", "value"),
        Input(IDS["FILT-PL"], "value"),
        Input(IDS["FILT-DEF"], "value"),
        Input(IDS["FILT-CLOCK"], "value"),
        Input(IDS["FILT-DRIBBLE"], "value"),
    ],
)
def get_filt_shotchart(tm, yr, pl_list, def_dist_vals, shot_clock_vals, dribble_vals):

    tmp_df = filter_df(
        tm,
        yr,
        def_dist_vals=def_dist_vals,
        shot_clock_vals=shot_clock_vals,
        dribble_vals=dribble_vals,
    )

    if pl_list != None and len(pl_list) > 0:
        tmp_df = tmp_df[tmp_df.player.isin(pl_list)]

    title = f"<B>{tm}</B> - {yr - 1}'/{yr}' season<BR>Filtered shot chart<BR><BR>Size: Frequency<BR>Color: Points / 100 shots"

    hexbin_dict = viz.get_hexbin_data(tmp_df, use_zones=True, show_runtime=True)
    fig = viz.draw_shotchart(
        hexbin_dict,
        stat_type="shot_ev",
        mode="white",
        min_freq=0.0005,
        min_hex_samples=2,
        max_freq=0.005,
        title_txt=title,
        legend_title="PTS/<BR>100",
        width=fig1_w - 20,
    )
    fig = add_logo(fig, tm, 50, 160, 375, opacity=0.75)
    fig.update_layout(hovermode="closest")
    return fig


def get_var_impact(df_in, col_name):
    # Determine how much impact a variable has
    freq_df = get_freq_df(df_in).rename({"freq": "avg"}, axis=1)
    acc_df = get_acc_df(df_in).rename({"acc": "avg"}, axis=1)

    for val in df_in[col_name].unique():
        tmp_df = df_in[df_in[col_name] == val]
        freq_ser = get_freq_df(tmp_df).rename({"freq": val}, axis=1)
        freq_df = freq_df.join(freq_ser, how="left")
        freq_df = freq_df.fillna(0)
        acc_ser = get_acc_df(tmp_df).rename({"acc": val}, axis=1)
        acc_df = acc_df.join(acc_ser, how="left")
        acc_df = acc_df.fillna(0)

    return freq_df, acc_df


def compare_var_impact(tm, yr, col_name):

    df = load_df(yr)
    tm_df = filter_df(tm, yr)

    freq_df, acc_df = get_var_impact(tm_df, col_name)
    ref_freq_df, ref_acc_df = get_var_impact(df, col_name)

    freq_df = freq_df.assign(NBA=ref_freq_df["avg"])
    acc_df = acc_df.assign(NBA=ref_acc_df["avg"])

    flat_freq_df = (
        freq_df.reset_index()
        .melt(id_vars="simple_zone")
        .rename({"value": "freq"}, axis=1)
    )
    flat_acc_df = acc_df.reset_index().melt(id_vars="simple_zone")

    flat_df = flat_freq_df.assign(acc=flat_acc_df["value"])

    return flat_df


@app.callback(
    [
        Output("GRAPH-VAR-IMPACT", "figure"),
        Output("variable-impact-chart-header", "title"),
    ],
    [Input("team-select-1", "value"), Input("season-select-1", "value")],
)
def draw_var_impact(tm, yr):

    df = load_df(yr)
    tm_df = filter_df(tm, yr)
    # Get baseline data

    def get_var_imp_df(df_in):
        ref_freq_df = get_freq_df(df_in)
        ref_shot_dist = df_in.shot_dist_calc.mean()
        ref_efg = get_efg_pct(df_in)

        results_list = list()
        for tmp_col in col_list:
            for col_val in df_in[tmp_col].unique():
                if col_val != "No Data":  # Skip uncategorised data rows
                    temp_df = df_in[df_in[tmp_col] == col_val]
                    temp_freq_df = get_freq_df(temp_df)
                    temp_efg = get_efg_pct(temp_df)
                    temp_shot_dist = temp_df.shot_dist_calc.mean()
                    rel_freq_df = ref_freq_df.join(temp_freq_df, lsuffix="_ref").fillna(
                        0
                    )

                    rel_efg = temp_efg - ref_efg
                    prof_change = cosine(rel_freq_df["freq_ref"], rel_freq_df["freq"])
                    rel_shot_dist = temp_shot_dist - ref_shot_dist

                    results_list.append(
                        dict(
                            variable=var_labels[tmp_col],
                            value=col_val,
                            rel_efg=rel_efg,
                            prof_change=prof_change,
                            rel_shot_dist=rel_shot_dist,
                            sample=len(temp_df) / len(df_in),
                        )
                    )

        df_out = pd.DataFrame(results_list)
        df_out = df_out.assign(txt="")
        df_out.loc[(df_out.value == "0 Dribbles"), "txt"] = "Catch &<BR>Shoot"
        df_out.loc[(df_out.value == "6+ Feet - Wide Open"), "txt"] = "Wide open"
        df_out.loc[(df_out.value == "4-6 Feet - Open"), "txt"] = "Open"
        df_out.loc[(df_out.value == "4-0 Very Late"), "txt"] = "Last 4s"
        df_out.loc[(df_out.value == "22-18 Very Early"), "txt"] = "Early<BR>clock"
        df_out.loc[(df_out.value == "2-4 Feet - Tight"), "txt"] = "Tight"
        df_out.loc[(df_out.value == "0-2 Feet - Very Tight"), "txt"] = "Very tight"
        return df_out

    tm_var_imp_df = get_var_imp_df(tm_df)
    tm_var_imp_df = tm_var_imp_df.assign(group=tm)

    nba_var_imp_df = get_var_imp_df(df)
    nba_var_imp_df = nba_var_imp_df.assign(group="NBA")

    var_imp_df = pd.concat([tm_var_imp_df, nba_var_imp_df])

    fig = px.scatter(
        var_imp_df,
        x="variable",
        y="rel_shot_dist",
        size="sample",
        color="rel_efg",
        text="txt",
        facet_col="group",
        facet_col_spacing=0.1,
        labels={
            "rel_shot_dist": "Shot distance vs avg",
            "variable": "Variable",
            "rel_efg": "eFG%<BR>Added",
        },
        color_continuous_midpoint=0,
        color_continuous_scale=px.colors.diverging.RdYlBu_r,
        hover_data={
            "value": True,
            "rel_shot_dist": ":.1f",
            "sample": ":.3f",
            "rel_efg": ":.1f",
        },
    )
    fig.update_traces(marker=dict(opacity=0.75, line=dict(width=1, color="navy")))
    fig.update_layout(hovermode="closest")

    return fig, f"{tm} ('{yr-1}-'{yr}) - Sensitivity analysis - overview"


@app.callback(
    [Output("GRAPH-VAR-DETAIL", "figure"), Output("var-impact-graph-title", "title")],
    [
        Input("team-select-1", "value"),
        Input("season-select-1", "value"),
        Input("stat-select-sensitivity-1", "value"),
    ],
)
def draw_var_impact_chart(tm, yr, col_name):
    flat_df = compare_var_impact(tm, yr, col_name)
    flat_df = flat_df[flat_df.simple_zone != "7 - 30+ ft"]

    order_lists = {
        "closest_def_dist": [
            "0-2 Feet - Very Tight",
            "2-4 Feet - Tight",
            "4-6 Feet - Open",
            "6+ Feet - Wide Open",
        ],
        "shot_clock": [
            "24-22",
            "22-18 Very Early",
            "18-15 Early",
            "15-7 Average",
            "7-4 Late",
            "4-0 Very Late",
        ],
        "dribble_range": [
            "0 Dribbles",
            "1 Dribble",
            "2 Dribbles",
            "3-6 Dribbles",
            "7+ Dribbles",
        ],
    }

    fig_freq = px.scatter(
        flat_df,
        x="acc",
        y="simple_zone",
        color="variable",
        size="freq",
        color_discrete_sequence=["Teal", "MediumBlue"]
        + px.colors.sequential.Oranges[3:9][: len(order_lists[col_name])]
        + ["gray"],
        labels={
            "acc": "Accuracy (%)",
            "simple_zone": "Court area",
            "variable": "Variable",
        },
        category_orders={
            "variable": ["NBA", "avg"] + order_lists[col_name] + ["No Data"]
        },
        hover_data={"variable": True, "acc": ":.1f", "freq": ":.1f"},
        height=300,
    )
    fig_freq.update_traces(marker=dict(line=dict(width=1, color="navy")))
    fig_freq.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1),
        hovermode="closest",
    )

    return fig_freq, f"{tm} ('{yr-1}-'{yr}) - Sensitivity analysis - detailed"


def layout():
    body = ddk.Block(
        [
            ddk.Block(
                [
                    ddk.ControlCard(
                        [
                            ddk.ControlItem(
                                [
                                    dcc.Dropdown(
                                        id="team-select-1",
                                        options=[
                                            {"label": t, "value": t} for t in team_list
                                        ],
                                        value=team_list[
                                            random.randrange(len(team_list))
                                        ],
                                        clearable=False,
                                    ),
                                ],
                                label="Select a team to review",
                                width=20,
                            ),
                            ddk.ControlItem(
                                [
                                    dcc.Slider(
                                        id="season-select-1",
                                        min=seasons_list[0],
                                        max=seasons_list[-1],
                                        value=seasons_list[-1],
                                        marks={
                                            s: {"label": f"{s - 1}-{s}"}
                                            for s in seasons_list
                                        },
                                        included=False,
                                    )
                                ],
                                label="Which season?",
                                width=30,
                            ),
                        ],
                        orientation="horizontal",
                    ),
                ],
                width=100,
            ),
            ddk.SectionTitle("Overview"),
            ddk.Row(
                [
                    ddk.Card(
                        [
                            ddk.CardHeader(
                                title="At a glance", id=IDS["HEADER-AT-A-GLANCE"]
                            ),
                            ddk.DataCard(
                                width=100,
                                value=f"N/A",
                                label=f"Points per shot",
                                id="tm_rating_points",
                                icon="tachometer-alt",  # map, map-marked-alt?
                                color="#bbbbbb",
                            ),
                            ddk.DataCard(
                                width=100,
                                value=f"N/A",
                                label=f"Shot location",
                                id="tm_rating_location",
                                icon="map-marked-alt",  # compass, map, map-marked-alt?
                                color="#bbbbbb",
                            ),
                            ddk.DataCard(
                                width=100,
                                value=f"N/A",
                                label=f"Shotmaking",
                                id="tm_rating_shotmaking",
                                icon="crosshairs",  # percentage, percent ?
                                color="#bbbbbb",
                            ),
                        ],
                        width=28,
                    ),
                    ddk.Card(
                        [
                            ddk.CardHeader(
                                title="In league context", id="team-sim-chart-header"
                            ),
                            dcc.RadioItems(
                                options=[
                                    {"label": "Shot efficiencies ", "value": "eff"},
                                    {"label": "Profile similarities ", "value": "sim"},
                                ],
                                value="eff",
                                labelStyle={
                                    "display": "inline-block",
                                    "padding": "10px",
                                },
                                id="team-sim-chart-select",
                            ),
                            dcc.Loading(children=[ddk.Graph(id=IDS["GRAPH-TEAM-SIM"])]),
                            ddk.CardFooter([""], id="team-sim-chart-footer"),
                        ],
                        width=40,
                    ),
                    ddk.Card(
                        [
                            ddk.CardHeader(title="Comparisons - by court areas"),
                            dcc.Dropdown(
                                id="team-select-ovr",
                                options=[{"label": "NBA", "value": "NBA"}]
                                + [{"label": t, "value": t} for t in team_list],
                                multi=True,
                                clearable=False,
                                value=["NBA"],
                            ),
                            dcc.Loading(children=[ddk.Graph(id="GRAPH-SHOT-SUM")]),
                            ddk.CardFooter(
                                [
                                    "Comparisons of shot frequency (size) and accuracy (color) by area of court"
                                ]
                            ),
                        ],
                        width=32,
                    ),
                ]
            ),
            ddk.Card(
                [
                    ddk.CardHeader(title="Compare shot charts"),
                    ddk.ControlCard(
                        [
                            ddk.ControlItem(
                                [
                                    dcc.Dropdown(
                                        id="team-select-2",
                                        options=[{"label": "NBA", "value": "NBA"}]
                                        + [{"label": t, "value": t} for t in team_list],
                                        value=team_list[
                                            random.randrange(len(team_list))
                                        ],
                                        clearable=False,
                                    ),
                                ],
                                label="Compare with",
                            ),
                            ddk.ControlItem(
                                [
                                    dcc.Dropdown(
                                        id="season-select-2",
                                        options=[
                                            {"label": f"{s - 1}-{s}", "value": s}
                                            for s in seasons_list
                                        ],
                                        value=seasons_list[-1],
                                        clearable=False,
                                    ),
                                ],
                                label="Season",
                            ),
                        ],
                        width=100,
                        orientation="horizontal",
                    ),
                    ddk.Block(
                        [
                            dcc.Loading(
                                children=[
                                    dcc.Graph(
                                        id="GRAPH-SHOTCHART-1",
                                        style={
                                            "width": fig1_w,
                                            "display": "block",
                                            "margin-left": "auto",
                                            "margin-right": "auto",
                                            "margin-top": "auto",
                                            "margin-bottom": "auto",
                                        },
                                        config={"displayModeBar": False},
                                    )
                                ]
                            )
                        ],
                        width=50,
                    ),
                    ddk.Block(
                        [
                            dcc.Loading(
                                children=[
                                    dcc.Graph(
                                        id="GRAPH-SHOTCHART-2",
                                        style={
                                            "width": fig2_w,
                                            "display": "block",
                                            "margin-left": "auto",
                                            "margin-right": "auto",
                                            "margin-top": "auto",
                                            "margin-bottom": "auto",
                                        },
                                        config={"displayModeBar": False},
                                    )
                                ]
                            )
                        ],
                        width=50,
                    ),
                    ddk.CardFooter(
                        ["Compare with another team or the league in general"]
                    ),
                ]
            ),
            ddk.SectionTitle("Data deep dive"),
            ddk.Row(
                [
                    ddk.Card(
                        [
                            ddk.CardHeader(
                                title="Sensitivity analysis - overview",
                                id="variable-impact-chart-header",
                            ),
                            dcc.Loading(children=[ddk.Graph(id="GRAPH-VAR-IMPACT")]),
                            ddk.CardFooter(
                                [
                                    "See the impact of each filter on shot location and accuracy"
                                ]
                            ),
                        ],
                        width=50,
                    ),
                    ddk.Block(
                        [
                            ddk.ControlCard(
                                [
                                    ddk.ControlItem(
                                        [
                                            dcc.Dropdown(
                                                id="stat-select-sensitivity-1",
                                                options=[
                                                    {"label": var_labels[t], "value": t}
                                                    for t in col_list
                                                ],
                                                value="shot_clock",
                                                clearable=False,
                                            ),
                                        ],
                                        label="Select a variable",
                                    )
                                ]
                            ),
                            ddk.Card(
                                [
                                    ddk.CardHeader(
                                        title="Sensitivity analysis - detailed",
                                        id="var-impact-graph-title",
                                    ),
                                    dcc.Loading(
                                        children=[ddk.Graph(id="GRAPH-VAR-DETAIL")]
                                    ),
                                    ddk.CardFooter(
                                        [
                                            "Data breakdown by zone for each filtered value"
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=50,
                    ),
                ]
            ),
            ddk.SectionTitle("Explore further details with these filters"),
            ddk.Row(
                [
                    ddk.ControlCard(
                        [
                            # ddk.SectionTitle("Filters"),
                            ddk.ControlItem(
                                [
                                    dcc.Dropdown(
                                        id=IDS["FILT-PL"],
                                        options=[],
                                        multi=True,
                                        clearable=True,
                                        value=None,
                                    ),
                                ],
                                label="Players (clear for all)",
                            ),
                            ddk.ControlItem(
                                [
                                    dcc.Dropdown(
                                        id=IDS["FILT-DEF"],
                                        options=[],
                                        multi=True,
                                        clearable=False,
                                        value=None,
                                    ),
                                ],
                                label="Closest Defender",
                            ),
                            ddk.ControlItem(
                                [
                                    dcc.Dropdown(
                                        id=IDS["FILT-CLOCK"],
                                        options=[],
                                        multi=True,
                                        clearable=False,
                                        value=None,
                                    ),
                                ],
                                label="Shot Clock",
                            ),
                            ddk.ControlItem(
                                [
                                    dcc.Dropdown(
                                        id=IDS["FILT-DRIBBLE"],
                                        options=[],
                                        multi=True,
                                        clearable=False,
                                        value=None,
                                    ),
                                ],
                                label="Dribbles",
                            ),
                        ],
                        width=40,
                    ),
                    ddk.Card(
                        [
                            dcc.Loading(
                                children=[
                                    dcc.Graph(
                                        id="GRAPH-SHOTCHART-FILT",
                                        style={
                                            "width": fig3_w,
                                            "display": "block",
                                            "margin-left": "auto",
                                            "margin-right": "auto",
                                            "margin-top": "auto",
                                            "margin-bottom": "auto",
                                        },
                                        config={"displayModeBar": False},
                                    )
                                ]
                            )
                        ],
                        width=60,
                    ),
                ]
            ),
        ]
    )

    return body
