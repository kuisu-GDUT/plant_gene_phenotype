from matplotlib import pyplot


def show_fig(y_dict: dict, y_label: str = None, title: str = None):
    fig, ax = pyplot.subplots()
    for label, y in y_dict.items():
        epochs = len(y)
        x_axis = range(0, epochs)
        ax.plot(x_axis, y, label=label)
        ax.legend()
    pyplot.ylabel(y_label)
    pyplot.title(title)
    pyplot.show()

