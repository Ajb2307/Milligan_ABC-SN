import matplotlib.pyplot as plt
import numpy as np

def plot_cm(cm, classes, figsize=(10, 10), title=False, normalize=True):
    ''' Code sent to me by Fed created by Willow to plot confusion matrix
    '''
    # I removed the variable R as it was not used

    textargs = {"fontname": "Serif"}

    # Normalize confusion matrix and set image parameters
    if normalize:
        cm = cm.astype("float") / np.nansum(cm, axis=1)[:, np.newaxis]
    else:
        cm = np.round(cm.astype("float"))
    off_diag = ~np.eye(cm.shape[0], dtype=bool)
    cm[off_diag] *= -1
    fig, ax = plt.subplots(figsize=figsize)
    
    if normalize:
        im = ax.imshow(cm, interpolation="none", cmap="RdBu", vmin=-1, vmax=1)
    else:
        im = ax.imshow(cm, interpolation="none", cmap="RdBu")

    if normalize:
        cbticks = np.linspace(-1, 1, num=9)
        cbticklabels = ["100%", "75%", "50%", "25%", "0%", "25%", "50%", "75%", "100%"]
        cb = plt.colorbar(im, shrink=0.82)
        cb.set_ticks(cbticks, labels=cbticklabels, fontsize= 12, **textargs)
    else:
        cb = plt.colorbar(im, shrink=0.82)

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
                text = "100 %"
            else:
                text = f"{val:.1%}"
            color = "w" if val >= 0.50 else "k"
            ax.text(
                j, i, text,
                ha="center", va="center",
                c=color, **textargs, fontsize=11,
            )
    ax.set_ylabel("True label", fontsize=20, **textargs)
    ax.set_xlabel("Predicted label", fontsize=20, **textargs)

    return fig

def graph_names(epochs, dataset_name, train_test_validation, weighted= False, balanced = False, notes=None):
    """
    Generates standard names for a file and a graph title.

    Parameters:
    - epochs (int): Number of epochs trained
    - weighted (bool): Whether weighted loss/weights were applied
    - balanced (bool): Whether balanced datasets were used
    - dataset_name (str): Name of the training dataset
    - training_test (str): "training", "testing", or "validation"
    - notes (str, optional): Additional notes to include in the graph title. Defaults to None.

    Returns:
    - filename (str): Suitable name for saving a file (CSV/Keras/PNG)
    - graph_title (str): Clean and informative title for plots
    """

    assert weighted == True or weighted == False, "weighted must be True or False"
    assert balanced == True or balanced == False, "balanced must be True or False"
    assert sum([weighted, balanced]) != 2, "weighted and balanced cannot both be True"

    assert train_test_validation.lower() in ["training", "testing", "validation"], "train_test must be 'training', 'testing', or 'validation'"
    assert type(dataset_name) == str, "dataset_name must be a string"
    assert type(epochs) == int, "epochs must be an integer"

    weight_str = "weighted" if weighted else ""
    balance_str = "balanced" if balanced else ""
    weight_balence_str = weight_str + balance_str

    # Generate a clean, snake_case filename
    filename = f"{dataset_name.lower().replace(' ', '_')}_{weight_balence_str}_{epochs}epochs_{train_test_validation.lower()}"

    # Generate a clean, human-readable graph title
    graph_title = f"{dataset_name.capitalize()} {train_test_validation.capitalize()} Data | Epochs: {epochs} | {weight_balence_str.capitalize()}"

    if notes is not None:
        assert type(notes) == str, "notes must be a string or None"
        graph_title += f" \n {notes}"

    return filename, graph_title


def model_name(epochs, dataset_name, weighted= False, balanced = False, notes=None):
    """
    Generates standard names for a file and a graph title.

    Parameters:
    - epochs (int): Number of epochs trained
    - weighted (bool): Whether weighted loss/weights were applied
    - balanced (bool): Whether balanced datasets were used
    - dataset_name (str): Name of the training dataset
    - notes (str, optional): Additional notes to include in the graph title. Defaults to None.

    Returns:
    - filename (str): Suitable name for saving a file (CSV/Keras/PNG)
    """

    assert weighted == True or weighted == False, "weighted must be True or False"
    assert balanced == True or balanced == False, "balanced must be True or False"
    assert sum([weighted, balanced]) != 2, "weighted and balanced cannot both be True"
    assert type(dataset_name) == str, "dataset_name must be a string"
    assert type(epochs) == int, "epochs must be an integer"

    weight_str = "weighted" if weighted else ""
    balance_str = "balanced" if balanced else ""
    weight_balence_str = weight_str + balance_str
    dataset_name = dataset_name.lower().replace(' ', '_')

    # Generate a clean, snake_case filename
    filename_list = [dataset_name, weight_balence_str, str(epochs)]
    if notes is not None:
        assert type(notes) == str, "notes must be a string or None"
        notes = notes.replace(" ", "_")
        filename_list.append(notes)
    filename = "_".join(filename_list)

    return filename
