import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import streamlit as st


def main(args):
    names = ["frame", "id", "x1", "y1", "x2", "y2"]
    df = pd.read_csv(args.results_file, header=None, names=names, usecols=list(range(len(names))))
    ids = np.unique(df["id"])
    res = {}
    fig, ax = plt.subplots()
    for id in ids:
        dfid = df[df["id"] == id][["frame", "x1", "y1", "x2", "y2"]]
        dfid["xc"] = (dfid["x1"] + dfid["x2"]) * 0.5
        dfid["yc"] = (dfid["y1"] + dfid["y2"]) * 0.5
        dfid["dx"] = np.r_[0.0, np.diff(dfid["xc"], n=1)]
        dfid["dy"] = np.r_[0.0, np.diff(dfid["yc"], n=1)]
        dfid["dnorm"] = np.sqrt(dfid["dx"]**2 + dfid["dy"]**2)
        res[id] = dfid
        ax.plot(dfid["frame"], dfid["dnorm"])
    ax.set_xlabel("Frame no")
    ax.set_ylabel("Speed(pixel/frame)")
    # plt.show()
    st.pyplot(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analize results.")
    parser.add_argument("results_file", action="store", type=str, help="Input results file.")
    args = parser.parse_args()
    main(args)