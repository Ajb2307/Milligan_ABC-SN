import matplotlib.pyplot as plt
import numpy as np

def plot_cm(cm, classes, figsize=(10, 10), title=False):
    ''' Code sent to me by Fed created by Willow to plot confusion matrix
    '''
    # I removed the variable R as it was not used

    textargs = {"fontname": "Serif"}

    # Normalize confusion matrix and set image parameters
    cm = cm.astype("float") / np.nansum(cm, axis=1)[:, np.newaxis]
    off_diag = ~np.eye(cm.shape[0], dtype=bool)
    cm[off_diag] *= -1
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm, interpolation="none", cmap="RdBu", vmin=-1, vmax=1)

    cbticks = np.linspace(-1, 1, num=9)
    cbticklabels = ["100%", "75%", "50%", "25%", "0%", "25%", "50%", "75%", "100%"]
    cb = plt.colorbar(im, shrink=0.82)
    cb.set_ticks(cbticks, labels=cbticklabels, fontsize= 12, **textargs)

    if title:
        ax.set_title(title, **textargs, fontsize=15)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, classes, rotation=90, **textargs, fontsize= 14)
    ax.set_yticks(tick_marks, classes, **textargs, fontsize= 14)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = np.abs(cm[i, j])
            if val == 0:
                text = ""
            elif np.isnan(val):
                text = "" # Handle NaN values
            elif val == 1:
                text = "100"
            else:
                text = f"{val*100:.1f}"
            color = "w" if val >= 0.50 else "k"
            ax.text(
                j, i, text,
                ha="center", va="center",
                c=color, **textargs, fontsize=11,
            )
    ax.set_ylabel("True label", fontsize=20, **textargs)
    ax.set_xlabel("Predicted label", fontsize=20, **textargs)

    return fig