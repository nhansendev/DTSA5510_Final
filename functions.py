import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from adjustText import adjust_text


def inspect_df(dataset):
    # Compact summary of dataFrame
    clen = max([len(c) for c in dataset.columns])
    for col in dataset.columns:
        try:
            vals, counts = np.unique(dataset[col], return_counts=True)
        except TypeError:
            # Can't sort mixed types like string and float, so make all string
            vals, counts = np.unique(dataset[col].astype(str), return_counts=True)

        if vals.dtype == np.object_:
            # Describe categorical data

            # Get missing count
            if "nan" in vals.tolist():
                missing = counts[vals.tolist().index("nan")]
            else:
                missing = 0

            if len(vals) > 5:
                # Show ends
                print(
                    f"{col.ljust(clen)}: ({len(vals)}) {vals.tolist()[:5]}, {counts.tolist()[:5]}",
                    end=" ... ",
                )
                print(
                    f"{vals.tolist()[-5:]}, {counts.tolist()[-5:]} | Missing: {missing}"
                )
            else:
                # Show all
                print(
                    f"{col.ljust(clen)}: ({len(vals)}) {vals.tolist()}, {counts.tolist()} | Missing: {missing}"
                )
        else:
            # Describe numerical data range, including missing value (nan) count
            idx = np.isnan(vals)
            notnan = vals[np.invert(idx)]
            missing = np.sum(counts[idx])
            print(
                f"{col.ljust(clen)}: {np.min(notnan):.3f} to {np.max(notnan):.3f} | Missing: {missing}"
            )


def plot_hist(
    dataframeA,
    dataframeB=None,
    bins=None,
    clone_bins=True,
    figsize=(12, 10),
    legend=["Before", "After"],
    nrows=5,
    ncols=5,
    legend_idx=4,
    density=False,
):
    # Plot histograms for each column in a dataframe
    # Optionally, include two dataframes for before and after comparison
    fig, ax = plt.subplots(nrows, ncols)
    fig.set_size_inches(figsize)
    axs = fig.axes
    bins_ = []
    for i, col in enumerate(np.unique(dataframeA.columns)):
        if bins is not None:
            h1 = axs[i].hist(dataframeA[col], bins=bins[i], rwidth=0.9, density=density)
        else:
            h1 = axs[i].hist(dataframeA[col], rwidth=0.9, density=density)
        bins_.append(h1[1])
        if dataframeB is not None:
            if clone_bins:
                # Force the bins of the second histogram to match the first
                axs[i].hist(dataframeB[col], bins=h1[1], rwidth=0.5, density=density)
            else:
                axs[i].hist(dataframeB[col], rwidth=0.5, density=density)
            if i == legend_idx:
                axs[i].legend(legend)
        axs[i].set_title(col)
    for j in range(i + 1, nrows * ncols):
        axs[j].set_axis_off()
    plt.tight_layout()
    plt.show()

    return bins_


def plot_map(
    SCRIPT_DIR,
    raw_data,
    xlims=[93.258, 171.557],
    ylims=[-45.47, -7.412],
    xadj=110,
    title="Australian Weather Stations",
):
    # Retrieve the location data for the stations
    locs = pd.read_csv(os.path.join(SCRIPT_DIR, "mapped_locs.csv"))
    locs.sort_values(by="Name", inplace=True, ignore_index=True)

    # Get the station name for each observation
    names = np.unique(raw_data["Location"])

    fig, ax = plt.subplots(1, 2, width_ratios=[0.7, 0.3])
    axs = fig.axes
    fig.set_size_inches(14, 6)

    # Plot the pre-generated map of Australia
    img = plt.imread(os.path.join(SCRIPT_DIR, "basemap.png"))
    axs[0].imshow(img, extent=xlims + ylims)
    axs[0].grid()

    # Plot a point for each station
    axs[0].plot(locs["Lon"], locs["Lat"], "k.")
    txt = []
    # Add a number tag to each point
    for i in range(len(locs)):
        txt.append(
            axs[0].text(
                locs["Lon"][i],
                locs["Lat"][i],
                str(i),
                va="bottom",
                ha="center",
                color="gray",
            )
        )

    # Fix overlaps
    adjust_text(txt, ax=axs[0], arrowprops=dict(arrowstyle="-", color="m", lw=0.5))

    axs[0].set_title(title)
    axs[0].set_xlabel("Longitude")
    axs[0].set_ylabel("Latitude")
    axs[0].set_xlim(xadj, xlims[1])
    axs[0].set_ylim(ylims)

    # Generate a table to pair number tags with station names
    N = int(np.ceil(len(names) / 2))
    table_text = [
        [
            str(i),
            names[i],
            " ",
            str(i + N),
            names[i + N] if (i + N) < len(names) else "",
        ]
        for i in range(N)
    ]
    tab = axs[1].table(table_text, loc="center")
    tab.auto_set_column_width([0, 1, 2, 3, 4])
    axs[1].axis("off")

    plt.tight_layout()

    return fig, axs, locs, [xadj, xlims[1]], ylims
