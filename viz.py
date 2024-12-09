# ========== (c) JP Hwang 1/2/21  ==========

import logging
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

desired_width = 320
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", desired_width)


def reshape_shotchart(fig, fig_width, margins=5):

    fig_height = fig_width * (470 + 2 * margins) / (500 + 2 * margins)
    fig.update_layout(width=fig_width, height=fig_height)
    # Set axes ranges
    fig.update_xaxes(range=[-250 - margins, 250 + margins])
    fig.update_yaxes(range=[-52.5 - margins, 417.5 + margins])

    return fig


# From: https://community.plot.ly/t/arc-shape-with-path/7205/5
def ellipse_arc(
    x_center=0.0,
    y_center=0.0,
    a=10.5,
    b=10.5,
    start_angle=0.0,
    end_angle=2 * np.pi,
    N=200,
    closed=False,
):
    """
    Build ellipses/arcs on plotly
    :param x_center:
    :param y_center:
    :param a:
    :param b:
    :param start_angle:
    :param end_angle:
    :param N:
    :param closed:
    :return:
    """
    t = np.linspace(start_angle, end_angle, N)
    x = x_center + a * np.cos(t)
    y = y_center + b * np.sin(t)
    path = f"M {x[0]}, {y[0]}"
    for k in range(1, len(t)):
        path += f"L{x[k]}, {y[k]}"
    if closed:
        path += " Z"
    return path


def draw_plotly_court(fig, fig_width=600, mode="white"):
    """
    :param fig:
    :param fig_width:
    :param mode: Colour mode - should be "dark" or "light"
    :return:
    """

    fig = reshape_shotchart(fig, fig_width)

    threept_break_y = (
        89.47765084  # Y coord of break point between the 3pt arc and the side 3pt line
    )

    if mode == "white":
        three_line_col = "dimgray"
        main_line_col = "gray"
        paper_bgcolor = "#F9F9F9"
        plot_bgcolor = "#F9F9F9"
        paint_col = "BlanchedAlmond"
        paint_col2 = "NavajoWhite"
        ft_circle_col = "BlanchedAlmond"
        twopt_color = "FloralWhite"
        hoop_col = "Firebrick"
    elif mode == "light":
        three_line_col = "orange"
        main_line_col = "#333333"
        paper_bgcolor = "wheat"
        plot_bgcolor = "Cornsilk"
        paint_col2 = "#F0F0F0"
        paint_col = "#F8F8F8"
        ft_circle_col = None
        twopt_color = "#F8F8F8"
        hoop_col = "orange"
    else:
        if mode != "dark":
            logger.info(
                f"Mode input of ({mode}) unrecognised - defaulting to dark colours."
            )
        three_line_col = "#ffffff"
        main_line_col = "#dddddd"
        paper_bgcolor = "DimGray"
        plot_bgcolor = "black"
        paint_col2 = None
        paint_col = None
        ft_circle_col = None
        twopt_color = "#DarkGray"
        hoop_col = "gray"

    fig.update_layout(
        # Line Horizontal
        margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks="",
            showticklabels=False,
            fixedrange=True,
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks="",
            showticklabels=False,
            fixedrange=True,
        ),
        shapes=[
            # Colour areas of the court
            dict(
                type="circle",
                x0=-237.5,
                y0=-237.5,
                x1=237.5,
                y1=237.5,
                xref="x",
                yref="y",
                line=dict(color=three_line_col, width=1),
                layer="below",
                fillcolor=twopt_color,
            ),
            dict(
                type="rect",
                x0=-250,
                y0=-200,
                x1=-220,
                y1=417.5,
                line=dict(color=None, width=0),
                fillcolor=plot_bgcolor,
                layer="below",
            ),
            dict(
                type="rect",
                x0=-1000,
                y0=-200,
                x1=1000,
                y1=-52.5,
                line=dict(color=None, width=0),
                fillcolor=plot_bgcolor,
                layer="below",
            ),
            dict(
                type="rect",
                x0=250,
                y0=-200,
                x1=220,
                y1=417.5,
                line=dict(color=None, width=0),
                fillcolor=plot_bgcolor,
                layer="below",
            ),
            dict(
                type="path",
                path=ellipse_arc(
                    a=237.5,
                    b=237.5,
                    start_angle=0.386283101,
                    end_angle=np.pi - 0.386283101,
                ),
                line=dict(color=three_line_col, width=1),
                layer="below",
            ),
            dict(
                type="line",
                x0=-220,
                y0=-52.5,
                x1=-220,
                y1=threept_break_y,
                line=dict(color=three_line_col, width=1),
                layer="below",
            ),
            dict(
                type="line",
                x0=220,
                y0=-52.5,
                x1=220,
                y1=threept_break_y,
                line=dict(color=three_line_col, width=1),
                layer="below",
            ),
            # Painted area
            dict(
                type="rect",
                x0=-250,
                y0=-52.5,
                x1=250,
                y1=417.5,
                line=dict(color=main_line_col, width=1),
                layer="below",
            ),
            dict(
                type="rect",
                x0=-80,
                y0=-52.5,
                x1=80,
                y1=137.5,
                line=dict(color=main_line_col, width=1),
                fillcolor=paint_col2,
                layer="below",
            ),
            dict(
                type="rect",
                x0=-60,
                y0=-52.5,
                x1=60,
                y1=137.5,
                line=dict(color=main_line_col, width=1),
                fillcolor=paint_col,
                layer="below",
            ),
            dict(
                type="circle",
                x0=-60,
                y0=77.5,
                x1=60,
                y1=197.5,
                xref="x",
                yref="y",
                line=dict(color=main_line_col, width=1),
                fillcolor=ft_circle_col,
                layer="below",
            ),
            dict(
                type="line",
                x0=-60,
                y0=137.5,
                x1=60,
                y1=137.5,
                line=dict(color=main_line_col, width=1),
                layer="below",
            ),
            # Hoop
            dict(
                type="rect",
                x0=-2,
                y0=-7.25,
                x1=2,
                y1=-12.5,
                line=dict(color=hoop_col, width=1),
                fillcolor=main_line_col,
            ),
            dict(
                type="circle",
                x0=-7.5,
                y0=-7.5,
                x1=7.5,
                y1=7.5,
                xref="x",
                yref="y",
                line=dict(color=hoop_col, width=1),
            ),
            dict(
                type="line",
                x0=-30,
                y0=-12.5,
                x1=30,
                y1=-12.5,
                line=dict(color=hoop_col, width=1),
            ),
            dict(
                type="path",
                path=ellipse_arc(a=40, b=40, start_angle=0, end_angle=np.pi),
                line=dict(color=main_line_col, width=1),
                layer="below",
            ),
            dict(
                type="line",
                x0=-250,
                y0=227.5,
                x1=-220,
                y1=227.5,
                line=dict(color=main_line_col, width=1),
                layer="below",
            ),
            dict(
                type="line",
                x0=250,
                y0=227.5,
                x1=220,
                y1=227.5,
                line=dict(color=main_line_col, width=1),
                layer="below",
            ),
            dict(
                type="line",
                x0=-90,
                y0=17.5,
                x1=-80,
                y1=17.5,
                line=dict(color=main_line_col, width=1),
                layer="below",
            ),
            dict(
                type="line",
                x0=-90,
                y0=27.5,
                x1=-80,
                y1=27.5,
                line=dict(color=main_line_col, width=1),
                layer="below",
            ),
            dict(
                type="line",
                x0=-90,
                y0=57.5,
                x1=-80,
                y1=57.5,
                line=dict(color=main_line_col, width=1),
                layer="below",
            ),
            dict(
                type="line",
                x0=-90,
                y0=87.5,
                x1=-80,
                y1=87.5,
                line=dict(color=main_line_col, width=1),
                layer="below",
            ),
            dict(
                type="line",
                x0=90,
                y0=17.5,
                x1=80,
                y1=17.5,
                line=dict(color=main_line_col, width=1),
                layer="below",
            ),
            dict(
                type="line",
                x0=90,
                y0=27.5,
                x1=80,
                y1=27.5,
                line=dict(color=main_line_col, width=1),
                layer="below",
            ),
            dict(
                type="line",
                x0=90,
                y0=57.5,
                x1=80,
                y1=57.5,
                line=dict(color=main_line_col, width=1),
                layer="below",
            ),
            dict(
                type="line",
                x0=90,
                y0=87.5,
                x1=80,
                y1=87.5,
                line=dict(color=main_line_col, width=1),
                layer="below",
            ),
            dict(
                type="path",
                path=ellipse_arc(
                    y_center=417.5, a=60, b=60, start_angle=-0, end_angle=-np.pi
                ),
                line=dict(color=main_line_col, width=1),
                layer="below",
            ),
        ],
    )
    return fig


def draw_shot_hexbins(
    xlocs,
    ylocs,
    hex_sizes,
    hex_colors,
    marker_cmin=None,
    marker_cmax=None,
    colorscale="RdYlBu_r",
    fig_width=800,
    legend_title="",
    ticktexts=[],
    img_out=None,
    mode="white",
):
    """
    Plot shot chart as hexbins
    :param xlocs: list of x locations
    :param ylocs: list of x locations
    :param hex_sizes: list of shot frequencies, for each location (prev freqs_by_hex)
    :param hex_colors: list of shot accuracies, for each location (prev acc_by_hex)
    :param marker_cmin: min value for marker color range
    :param marker_cmax: max value for marker color range
    :param colorscale: plotly colorscale name
    :param fig_width:
    :param legend_title: Colorscale title
    :param ticktexts: Colorscale text
    :param hexbin_text: Hovertext on markers
    :param img_out:
    :param mode: color mode (dark or anything else)
    # :param size_factor: Controls the max size of markers  # TODO - scale based on fig size
    :return:
    """

    hex_size_factor = 18 * (fig_width / 800)
    tickfont_size = 13 * (fig_width / 800)
    colorbar_thick = 17 * (fig_width / 800)

    import plotly.graph_objects as go

    if mode == "dark":
        textcolor = "#eeeeee"
        marker_color = "#eeeeee"
    else:
        textcolor = "#222222"
        marker_color = "#404040"

    if marker_cmin is None:
        marker_cmin = min(hex_colors)
    if marker_cmax is None:
        marker_cmax = max(hex_colors)

    fig = go.Figure()
    draw_plotly_court(fig, fig_width=fig_width, mode=mode)
    fig.add_trace(
        go.Scatter(
            x=xlocs,
            y=ylocs,
            mode="markers",
            name="markers",
            marker=dict(
                # Variable size according to frequency
                size=hex_sizes,
                sizemode="area",
                sizeref=2.0 * max(hex_sizes) / (float(hex_size_factor) ** 2),
                sizemin=1.5,
                # Set up colors/colorbar
                color=hex_colors,
                colorscale=colorscale,
                colorbar=dict(
                    x=0.87,
                    y=0.97,
                    thickness=colorbar_thick,
                    yanchor="top",
                    len=0.22,
                    title=dict(
                        text=legend_title,
                        font=dict(size=tickfont_size * 1.15, color=textcolor),
                    ),
                    tickvals=[
                        marker_cmin,
                        (marker_cmin + marker_cmax) / 2,
                        marker_cmax,
                    ],
                    ticktext=ticktexts,
                    tickfont=dict(size=tickfont_size, color=textcolor),
                ),
                cmin=marker_cmin,
                cmax=marker_cmax,
                line=dict(width=1, color=marker_color),
                symbol="hexagon",
            ),
            # hoverinfo='text'
            hovertemplate="%{marker.color:.2f}pts/100<br>from this area",
        )
    )

    if img_out is not None:
        fig.write_image(img_out)

    return fig


def get_def_hexbin_params(gridsize=None, min_hex_freqs=None):
    """
    Get default parameters
    :param gridsize:
    :param min_hex_freqs:
    :return:
    """

    if gridsize is None:
        gridsize = 41
    if min_hex_freqs is None:
        min_hex_freqs = 0.0005

    return gridsize, min_hex_freqs


# TODO - check accuracy
def mark_hexbin_threes(xlocs, ylocs):

    hexbin_isthree = [False] * len(xlocs)
    for i in range(len(xlocs)):
        temp_xloc = xlocs[i]
        temp_yloc = ylocs[i]
        isthree = False
        if temp_xloc < -220 or temp_xloc > 220:
            isthree = True

        shot_dist = (temp_xloc ** 2 + temp_yloc ** 2) ** 0.5
        if shot_dist > 237.5:
            isthree = True

        if isthree:
            hexbin_isthree[i] = True

    return hexbin_isthree


def get_threes_mask(gridsize=41):

    x_coords = list()
    y_coords = list()
    for i in range(-250, 250, 5):
        for j in range(-48, 423, 5):
            x_coords.append(i)
            y_coords.append(j)

    fig, axs = plt.subplots(ncols=2)
    shots_hex = axs[0].hexbin(
        x_coords,
        y_coords,
        extent=(-250, 250, 422.5, -47.5),
        cmap=plt.cm.Reds,
        gridsize=gridsize,
    )
    plt.close()

    xlocs = [i[0] for i in shots_hex.get_offsets()]
    ylocs = [i[1] for i in shots_hex.get_offsets()]

    threes_mask = mark_hexbin_threes(xlocs, ylocs)

    return threes_mask


def get_zones(x, y, excl_angle=False):
    def append_name_by_angle(temp_angle):

        if excl_angle:
            temp_text = ""
        else:
            if temp_angle < 60 and temp_angle >= -90:
                temp_text = "_right"
            elif temp_angle < 120 and temp_angle >= 60:
                temp_text = "_middle"
            else:
                temp_text = "_left"
        return temp_text

    import math

    zones_list = list()
    for i in range(len(x)):

        temp_angle = math.atan2(y[i], x[i]) / math.pi * 180
        temp_dist = ((x[i] ** 2 + y[i] ** 2) ** 0.5) / 10

        if temp_dist > 30:
            zone = "7 - 30+ ft"
        elif (x[i] < -220 or x[i] > 220) and y[i] < 90:
            zone = "4 - Corner 3"
            zone += append_name_by_angle(temp_angle)
        elif temp_dist > 26:
            zone = "6 - Long 3"
            zone += append_name_by_angle(temp_angle)
        elif temp_dist > 23.75:
            zone = "5 - Short 3 (<27 ft)"
            zone += append_name_by_angle(temp_angle)
        elif temp_dist > 14:
            zone = "3 - Long 2 (14+ ft)"
            zone += append_name_by_angle(temp_angle)
        elif temp_dist > 4:
            zone = "2 - Short 2 (4-14 ft)"
            zone += append_name_by_angle(temp_angle)
        else:
            zone = "1 - Within 4 ft"

        zones_list.append(zone)

    return zones_list


def get_hexbin_data(
    shots_df,
    gridsize=None,
    use_zones=True,
    smoothing=1,
    filt_sm_samples=True,
    show_runtime=False,
):
    """

    :param shots_df:
    :param gridsize:
    :param smoothing:
    :param filt_sm_samples:
    :return:
    """
    if show_runtime:
        logger.info("Getting hexbin data")

    matplotlib.use("Agg")
    # court_extents = (-250, 250, 422.5, -47.5)  # Old references that I was using - seems wrong
    court_extents = (-250, 250, 417.5, -52.5)  # ie (0, 0) is the basket

    # Get parameters
    gridsize, _ = get_def_hexbin_params(gridsize)

    # Build hexbins & collect data
    fig, axs = plt.subplots(ncols=2)
    shots_hex = axs[0].hexbin(
        shots_df.x,
        shots_df.y,
        extent=court_extents,
        cmap=plt.cm.Reds,
        gridsize=gridsize,
    )

    makes_df = shots_df[shots_df.made]
    makes_hex = axs[0].hexbin(
        makes_df.x,
        makes_df.y,
        extent=court_extents,
        cmap=plt.cm.Reds,
        gridsize=gridsize,
    )

    x = shots_hex.get_offsets()[:, 0]
    y = shots_hex.get_offsets()[:, 1]

    shots_by_hex = shots_hex.get_array()
    makes_by_hex = makes_hex.get_array()

    if show_runtime:
        logger.info("Getting shot accuracies")
    # ========================================
    # ===== Calculate shot accuracies
    # ========================================
    if not use_zones:
        # ===== METHOD 1 - LOCAL AVERAGING =====
        # raw accuracies
        acc_by_hex = makes_by_hex / shots_by_hex
        acc_by_hex[np.isnan(makes_by_hex / shots_by_hex)] = 0  # Filling div/0 values

        # Smooth data with neighboring bins by local averaging
        if smoothing >= 1:
            # get closest points for averaging
            xy_df = (
                pd.DataFrame([x, y, makes_by_hex, shots_by_hex])
                .transpose()
                .rename({0: "x", 1: "y", 2: "makes", 3: "shots"}, axis=1)
            )
            x_spacing = 2 * (
                np.sort(xy_df.x.unique())[1] - np.sort(xy_df.x.unique())[0]
            )  # Due to hexagon geometry
            y_spacing = np.sort(xy_df.y.unique())[1] - np.sort(xy_df.y.unique())[0]

            len_list = list()
            for i in range(len(x)):
                tmp_x = x[i]
                tmp_y = y[i]
                if shots_by_hex[i] > 0:
                    # Get data from adjoining cells
                    adj_xy_df = xy_df[
                        ((tmp_x - xy_df.x).abs() <= (x_spacing * smoothing * 1.1))
                        & ((tmp_y - xy_df.y).abs() <= (y_spacing * smoothing * 1.1))
                    ]
                    adj_xy_df = adj_xy_df[
                        -((adj_xy_df.x == tmp_x) & (adj_xy_df.y == tmp_y))
                    ]
                    len_list.append(len(adj_xy_df))
                    adj_shots = adj_xy_df.shots.sum()
                    adj_makes = adj_xy_df.makes.sum()
                    if adj_shots > 0:
                        if (
                            adj_shots / shots_by_hex[i] > 0.5
                        ):  # If volume of surround shots > 0.5 of shots in main hex:
                            adj_makes = (adj_makes / adj_shots) * (
                                shots_by_hex[i] * 0.5
                            )  # Scale to a max of 0.5 times shots in that hex
                            adj_shots = shots_by_hex[i] * 0.5
                        corr_factor = shots_by_hex[i] / adj_shots
                        acc_by_hex[i] = (makes_by_hex[i] + adj_makes) / (
                            shots_by_hex[i] + adj_shots
                        )
                    else:
                        acc_by_hex[i] = 0

        # Get shot expected values
        shot_ev_by_hex = acc_by_hex * 2
        threes_mask = get_threes_mask(gridsize=gridsize)
        for i in range(len(threes_mask)):
            if threes_mask[i]:
                shot_ev_by_hex[i] = shot_ev_by_hex[i] * 1.5

    else:
        # ===== METHOD 2 - BY ZONES =====
        # Calc grouped value
        grp_acc = (
            shots_df.groupby("shot_zone").sum()["made"]
            / shots_df.groupby(["shot_zone"]).count()["made"]
        ).rename("acc")
        shots_df = shots_df.assign(
            pts=shots_df["value"] * shots_df["made"]
        )  # actual points gained from shot
        grp_shot_ev = (
            shots_df.groupby("shot_zone").sum()["pts"]
            / shots_df.groupby(["shot_zone"]).count()["made"]
        ).rename("shot_ev")
        grp_df = pd.concat([grp_acc, grp_shot_ev], axis=1).reset_index()

        # Need a mapping of hex gridpoint to an equivalent zone
        tmp_hex_df = pd.DataFrame([x, y], index=["x", "y"]).transpose()
        tmp_hex_df = tmp_hex_df.assign(
            shot_zone=get_zones(list(tmp_hex_df["x"]), list(tmp_hex_df["y"]))
        )
        tmp_hex_df = tmp_hex_df.merge(grp_df, how="left", on="shot_zone")

        acc_by_hex = tmp_hex_df["acc"].values
        shot_ev_by_hex = tmp_hex_df["shot_ev"].values

    # ========================================
    # ===== END - Calculate shot accuracies
    # ========================================

    freq_by_hex = shots_by_hex / sum(shots_by_hex)

    hexbin_dict = dict()

    hexbin_dict["gridsize"] = gridsize
    hexbin_dict["n_shots"] = len(shots_df)

    hexbin_dict["xlocs"] = x
    hexbin_dict["ylocs"] = y

    hexbin_dict["shots_by_hex"] = shots_by_hex
    hexbin_dict["freq_by_hex"] = freq_by_hex

    hexbin_dict["acc_by_hex"] = acc_by_hex
    hexbin_dict["shot_ev_by_hex"] = shot_ev_by_hex

    plt.close()

    if show_runtime:
        logger.info("Finished getting hexbin data")
    return hexbin_dict


def clip_hex_freqs(input_freqs, max_freq=0.002):
    freq_by_hex = np.array([min(max_freq, i) for i in input_freqs])
    return freq_by_hex


def filt_hexbin_sm_samples(hexbin_data, min_hex_freq=0.0002, min_hex_samples=5):
    """
    Filter hexbin stats to exclude hexbin values below a threshold value (of frequency)

    :param hexbin_data:
    :param min_hex_freq: Frequency (as a part of 1)
    :param min_hex_samples Frequency (as a number of shots)
    :return:
    """
    from copy import deepcopy

    # TODO - consider removing markers altogether rather than setting them to zero
    #   If doing so - neeeds to be the LAST step in preprocessing

    filt_hexbin_data = deepcopy(hexbin_data)

    temp_len = len(filt_hexbin_data["freq_by_hex"])
    filt_array = [
        (
            filt_hexbin_data["freq_by_hex"][i] > min_hex_freq
            and filt_hexbin_data["shots_by_hex"][i] > min_hex_samples
        )
        for i in range(temp_len)
    ]
    filt_array = np.array(filt_array)
    logger.info(
        f"Filtering hex markers by thresholds of {min_hex_freq} / {min_hex_samples}"
    )
    logger.info(f"{sum(filt_array)} of {len(filt_array)} hex will be filtered out.")
    for k, v in filt_hexbin_data.items():
        if k in ["freq_by_hex", "shots_by_hex"]:
            if len(v) == temp_len:
                filt_hexbin_data[k] = v * filt_array
            else:
                logger.warning(f"WEIRD! The {k} array has a wrong length!")
        else:
            pass

    return filt_hexbin_data


# TODO -
#   Workflow:
#       Filter shot data (players / teams)
#       Get hexbin data (relative or absolute)
#       Filter hexbin data for minimum samples
#       Plot shot chart
#       Write annotations
#   function to filter hexbin data (get_rel_hexbin_data)
#   function to draw hexbin plot w/ court (draw_shot_hexbins)
#   function to add annotation / text
#   wrapper function with presets (hexbin data as input, simple options)


def get_rel_hexbin_data(shots_df, ref_shots_df, show_runtime=False):
    """
    For getting relative stats - e.g. team shots per location vs league avg
    :param shots_df:
    :param ref_shots_df:
    :return:
    """
    if show_runtime:
        logger.info("Getting relative hexbin data")
    hexbin_dict = get_hexbin_data(shots_df)
    ref_hexbin_dict = get_hexbin_data(ref_shots_df)

    rel_hexbin_dict = dict()
    id_keys = ["xlocs", "ylocs", "gridsize", "n_shots"]  # Invariant stats
    for k in id_keys:
        if not np.array_equal(hexbin_dict[k], ref_hexbin_dict[k]):
            logger.warning(f"Difference between the two values in key {k}")
            logger.warning(
                f"Input value: {hexbin_dict[k]}, Ref value: {ref_hexbin_dict[k]}"
            )
            if ref_hexbin_dict["xlocs"].shape == ref_hexbin_dict["xlocs"].shape:
                logger.info(f"Well, at least the shapes are the same!")
            else:
                logger.info(f"Hmm - even the shapes are not the same! Returning None")
                return None
        rel_hexbin_dict[k] = hexbin_dict[k]

    data_keys = ["shots_by_hex", "freq_by_hex"]
    for k in data_keys:
        rel_hexbin_dict[k] = hexbin_dict[k]

    var_keys = ["acc_by_hex", "shot_ev_by_hex"]  # Relative stats keys
    for k in var_keys:
        rel_hexbin_dict[k] = ref_hexbin_dict[k] - hexbin_dict[k]

    if show_runtime:
        logger.info("Finished getting relative hexbin data")
    return rel_hexbin_dict


def add_plotly_logo(fig):
    # TODO - add plotly logo to the court
    return fig


def draw_shotchart(
    hexbin_dict,
    stat_type="shot_ev",
    # coord="cart",
    marker_cmin=None,
    marker_cmax=None,
    min_freq=0.0002,
    min_hex_samples=5,
    max_freq=0.005,
    marker_texts=(),
    title_txt="",
    legend_title="",
    mode="white",
    width=600,
    tm_logo=False,
):
    """

    :param hexbin_dict:
    :param stat_type:
    :param marker_cmin:
    :param marker_cmax:
    :param min_freq:
    :param min_hex_samples:
    :param max_freq:
    :param marker_texts:
    :param title_txt: Chart title text
    :param mode: Color mode
    :param width: Figure width
    :return:
    """
    if min_freq is not None or min_hex_samples is not None:
        hexbin_dict = filt_hexbin_sm_samples(
            hexbin_dict, min_hex_freq=min_freq, min_hex_samples=min_hex_samples
        )

    freq_by_hex = hexbin_dict["freq_by_hex"]

    stat_types = ["shot_ev", "acc"]

    if stat_type not in stat_types:
        logger.info(
            f"Stat type {stat_type} not recognised, defaulting to {stat_types[0]}"
        )
        stat_type = stat_types[0]

    if stat_type == "shot_ev":
        hex_colors = hexbin_dict["shot_ev_by_hex"] * 100
        if marker_cmin is None:
            marker_cmin = 80  # 1.048 -> avg
        if marker_cmax is None:
            marker_cmax = 130
    elif stat_type == "acc":
        hex_colors = hexbin_dict["acc_by_hex"]
        if marker_cmin is None:
            marker_cmin = 0.3
        if marker_cmax is None:
            marker_cmax = 0.6
    else:
        logger.error("Something went wrong! Setting stat type to 'shot_ev_by_hex'")
        hex_colors = hexbin_dict["shot_ev_by_hex"]

    if max_freq is not None:
        freq_by_hex = clip_hex_freqs(freq_by_hex, max_freq=max_freq)

    # Set ticktexts
    if marker_cmax < max(hex_colors):
        cmax_text = f"{marker_cmax}+"
    else:
        cmax_text = f"{marker_cmax}"
    if marker_cmin < min(hex_colors):
        cmin_text = f"{marker_cmin}-"
    else:
        cmin_text = f"{marker_cmin}"

    if type(marker_cmax) == int:
        marker_cmid = int((marker_cmin + marker_cmax) / 2)
    else:
        marker_cmid = (marker_cmin + marker_cmax) / 2
    ticktexts = [cmin_text, f"{marker_cmid}", cmax_text]

    colorscale = "RdYlBu_r"

    fig = draw_shot_hexbins(
        hexbin_dict["xlocs"],
        hexbin_dict["ylocs"],
        hex_sizes=freq_by_hex,
        hex_colors=hex_colors,
        marker_cmin=marker_cmin,
        marker_cmax=marker_cmax,
        colorscale=colorscale,
        legend_title=legend_title,
        mode=mode,
        fig_width=width,
    )

    fig.update_traces(text=marker_texts, marker=dict(colorbar=dict(ticktext=ticktexts)))

    if mode == "dark":
        textcolor = "white"
    else:
        textcolor = "#222222"

    fig = add_shotchart_title(fig, title_txt, textcolor=textcolor)

    return fig


def add_shotchart_title(
    fig, title_txt, title_xloc=0.05, title_yloc=0.93, size=12, textcolor="#eeeeee"
):

    fig.update_layout(
        title=dict(
            text=title_txt,
            y=title_yloc,
            x=title_xloc,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            font=dict(
                # family="Helvetica, Arial, Tahoma",
                size=size,
                color=textcolor,
            ),
        ),
        font=dict(family="Open Sans, Arial", size=14, color=textcolor),
    )
    return fig


def add_annotation(fig, text="Test text", xloc=0.5, yloc=0.95):
    fig.add_annotation(
        go.layout.Annotation(
            x=xloc,
            y=yloc,
            showarrow=False,
            text=text,
            xanchor="middle",
            yanchor="top",
            xref="paper",
            yref="paper",
            font=dict(family="Open Sans, Arial", size=11, color="white"),
        ),
    )
    return fig


def main():

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    sh.setFormatter(formatter)
    root_logger.addHandler(sh)
