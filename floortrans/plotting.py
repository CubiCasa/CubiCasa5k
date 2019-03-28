import matplotlib.pyplot as plt
from matplotlib import colors, cm
import matplotlib.path as mplp
import numpy as np
import torch
from skimage import draw
from shapely.geometry import Polygon, Point


def discrete_cmap_furukawa():
    """create a colormap with N (N<15) discrete colors and register it"""
    # define individual colors as hex values
    cpool = ['#696969', '#b3de69', '#ffffb3', '#8dd3c7', '#fdb462',
             '#fccde5', '#80b1d3', '#d9d9d9', '#fb8072', '#577a4d',
             'white', '#000000', '#e31a1c']
    cmap3 = colors.ListedColormap(cpool, 'rooms_furukawa')
    cm.register_cmap(cmap=cmap3)

    cpool = ['#ede676', '#8dd3c7', '#b15928', '#fdb462', '#ffff99',
             '#fccde5', '#80b1d3', '#d9d9d9', '#fb8072', '#696969',
             '#577a4d', '#e31a1c', '#42ef59', '#8c595a', '#3131e5',
             '#48e0e6', 'white']
    cmap3 = colors.ListedColormap(cpool, 'icons_furukawa')
    cm.register_cmap(cmap=cmap3)


def drawJunction(h, point, point_type, width, height):
    lineLength = 15
    lineWidth = 10
    x, y = point
    # plt.text(x,y,str(index),fontsize=25,color='r')
    if point_type == -1:
        h.scatter(x, y, color='#6488ea')
    ###########################
    # o
    # | #6488ea soft blue
    # | drawcode = [1,1]
    #
    ###########################
    if point_type == 0:
        h.plot([x, x], [y, min(y + lineLength, height - 1)],
               linewidth=lineWidth, color='#6488ea')
        #plt.scatter(x, y-10, c='k')
    ###########################
    #
    #  ---o #6241c7 bluey purple
    #     drawcode = [1,2]
    #
    ###########################
    elif point_type == 1:
        h.plot([x, max(x - lineLength, 0)], [y, y],
               linewidth=lineWidth, color='#6241c7')
        #plt.scatter(x+10, y, c='k')
    ###########################
    #    |
    #    | drawcode = [1,3]
    #    o #056eee cerulean blue
    #
    ###########################
    elif point_type == 2:
        h.plot([x, x], [y, max(y - lineLength, 0)],
               linewidth=lineWidth, color='#056eee')
        #plt.scatter(x, y+10, c='k')
    ###########################
    #
    #  drawcode = [1,4]
    #
    #  o--- #004577 prussian blue
    #
    ###########################
    elif point_type == 3:
        h.plot([x, min(x + lineLength, width - 1)], [y, y],
               linewidth=lineWidth, color='#004577')
        #plt.scatter(x-10, y, c='k')
    ###########################
    #
    # |--- drawcode = [2,3]
    # |
    #
    ###########################
    elif point_type == 6:
        h.plot([x, min(x + lineLength, width - 1)], [y, y],
               linewidth=lineWidth, color='#04d8b2')
        h.plot([x, x], [y, min(y + lineLength, height - 1)],
               linewidth=lineWidth, color='#04d8b2')
    ###########################
    #
    #  ---|
    #     | drawcode = [2,4]
    #
    ###########################
    elif point_type == 7:
        h.plot([x, max(x - lineLength, 0)], [y, y],
               linewidth=lineWidth, color='#cdfd02')
        h.plot([x, x], [y, min(y + lineLength, height - 1)],
               linewidth=lineWidth, color='#cdfd02')
    ###########################
    #    |
    # ---| drawcode = [2,1]
    #
    #
    ###########################
    elif point_type == 4:
        h.plot([x, max(x - lineLength, 0)], [y, y],
               linewidth=lineWidth, color='#ff81c0')
        h.plot([x, x], [y, max(y - lineLength, 0)],
               linewidth=lineWidth, color='#ff81c0')
    ###########################
    #
    #  |
    #  | drawcode = [2,2]
    #  --
    #
    ###########################
    elif point_type == 5:
        h.plot([x, min(x + lineLength, width - 1)], [y, y],
               linewidth=lineWidth, color='#f97306')
        h.plot([x, x], [y, max(y - lineLength, 0)],
               linewidth=lineWidth, color='#f97306')
    ###########################
    #
    # |
    # |--- drawcode = [3,4]
    # |
    #
    ###########################
    elif point_type == 11:
        h.plot([x, min(x + lineLength, width - 1)],
               [y, y], linewidth=lineWidth, color='b')
        h.plot([x, x], [y, max(y - lineLength, 0)],
               linewidth=lineWidth, color='b')
        h.plot([x, x], [y, min(y + lineLength, height - 1)],
               linewidth=lineWidth, color='b')
    ###########################
    #
    # ---
    #  |  drawcode = [3,1]
    #  |
    #
    ###########################
    elif point_type == 8:
        h.plot([x, min(x + lineLength, width - 1)],
               [y, y], linewidth=lineWidth, color='y')
        h.plot([x, max(x - lineLength, 0)], [y, y],
               linewidth=lineWidth, color='y')
        h.plot([x, x], [y, min(y + lineLength, height - 1)],
               linewidth=lineWidth, color='y')
    ###########################
    #
    #    |
    # ---| drawcode = [3,2]
    #    |
    #
    ###########################
    elif point_type == 9:
        h.plot([x, max(x - lineLength, 0)], [y, y],
               linewidth=lineWidth, color='r')
        h.plot([x, x], [y, max(y - lineLength, 0)],
               linewidth=lineWidth, color='r')
        h.plot([x, x], [y, min(y + lineLength, height - 1)],
               linewidth=lineWidth, color='r')
    ###########################
    #
    #  |
    #  | drawcode = [3,3]
    # ---
    #
    ###########################
    elif point_type == 10:
        h.plot([x, min(x + lineLength, width - 1)],
               [y, y], linewidth=lineWidth, color='m')
        h.plot([x, max(x - lineLength, 0)], [y, y],
               linewidth=lineWidth, color='m')
        h.plot([x, x], [y, max(y - lineLength, 0)],
               linewidth=lineWidth, color='m')
    ###########################
    #
    #  |
    # --- drawcode = [4,1]
    #  |
    #
    ###########################
    elif point_type == 12:
        h.plot([x, min(x + lineLength, width - 1)],
               [y, y], linewidth=lineWidth, color='k')
        h.plot([x, max(x - lineLength, 0)], [y, y],
               linewidth=lineWidth, color='k')
        h.plot([x, x], [y, max(y - lineLength, 0)],
               linewidth=lineWidth, color='k')
        h.plot([x, x], [y, min(y + lineLength, height - 1)],
               linewidth=lineWidth, color='k')

    lineLength = 10
    lineWidth = 5

    ###########################
    # o--- opening left
    ###########################
    if point_type == 13:
        h.plot([x], [y], 'o', markersize=30, color='red')
        h.plot([x], [y], 'o', markersize=25, color='white')
        h.text(x, y, 'OL', fontsize=30, color='magenta')
    ###########################
    # ---o opening right
    ###########################
    elif point_type == 14:
        h.plot([x], [y], 'o', markersize=30, color='red')
        h.plot([x], [y], 'o', markersize=25, color='white')
        h.text(x, y, 'OR', fontsize=30, color='magenta')
    ###########################
    # o opening up
    # |
    # |
    ###########################
    elif point_type == 15:
        h.plot([x], [y], 'o', markersize=30, color='red')
        h.plot([x], [y], 'o', markersize=25, color='white')
        h.text(x, y, 'OU', fontsize=30, color='mediumblue')
    ###########################
    # | opening down
    # |
    # o
    ###########################
    elif point_type == 16:
        h.plot([x], [y], 'o', markersize=30, color='red')
        h.plot([x], [y], 'o', markersize=25, color='white')
        h.text(x, y, 'OD', fontsize=30, color='mediumblue')

    ###########################
    #
    # |--- drawcode = [2,3]
    # |
    #
    ###########################
    elif point_type == 17:
        h.plot([x, min(x + lineLength, width - 1)], [y, y],
               linewidth=lineWidth, color='indianred')
        h.plot([x, x], [y, min(y + lineLength, height - 1)],
               linewidth=lineWidth, color='indianred')
    ###########################
    #
    #  ---|
    #     | drawcode = [2,4]
    #
    ###########################
    elif point_type == 18:
        h.plot([x, max(x - lineLength, 0)], [y, y],
               linewidth=lineWidth, color='darkred')
        h.plot([x, x], [y, min(y + lineLength, height - 1)],
               linewidth=lineWidth, color='darkred')
    ###########################
    #
    #  |
    #  | drawcode = [2,2]
    #  --
    #
    ###########################
    elif point_type == 19:
        h.plot([x, min(x + lineLength, width - 1)],
               [y, y], linewidth=lineWidth, color='salmon')
        h.plot([x, x], [y, max(y - lineLength, 0)],
               linewidth=lineWidth, color='salmon')
    ###########################
    #    |
    # ---| drawcode = [2,1]
    #
    #
    ###########################
    elif point_type == 20:
        h.plot([x, max(x - lineLength, 0)], [y, y],
               linewidth=lineWidth, color='orangered')
        h.plot([x, x], [y, max(y - lineLength, 0)],
               linewidth=lineWidth, color='orangered')


def draw_junction_from_dict(point_dict, width, height, size=1, fontsize=30):
    index = 0
    markersize_large = 20 * size
    markersize_small = 15 * size
    for point_type, locations in point_dict.items():
        for loc in locations:

            x, y = loc
            lineLength = 20 * size
            lineWidth = 20 * size
            # plt.text(x,y,str(index),fontsize=25,color='r')
            ###########################
            # o
            # | #6488ea soft blue
            # | drawcode = [1,1]
            #
            ###########################
            if point_type == 0:
                plt.plot([x, x], [y, min(y + lineLength, height - 1)],
                         linewidth=lineWidth, color='#6488ea')
                #plt.scatter(x, y-10, c='k')
            ###########################
            #
            #  ---o #6241c7 bluey purple
            #     drawcode = [1,2]
            #
            ###########################
            elif point_type == 1:
                plt.plot([x, max(x - lineLength, 0)], [y, y],
                         linewidth=lineWidth, color='#6241c7')
                #plt.scatter(x+10, y, c='k')
            ###########################
            #    |
            #    | drawcode = [1,3]
            #    o #056eee cerulean blue
            #
            ###########################
            elif point_type == 2:
                plt.plot([x, x], [y, max(y - lineLength, 0)],
                         linewidth=lineWidth, color='#056eee')
                #plt.scatter(x, y+10, c='k')
            ###########################
            #
            #  drawcode = [1,4]
            #
            #  o--- #004577 prussian blue
            #
            ###########################
            elif point_type == 3:
                plt.plot([x, min(x + lineLength, width - 1)], [y, y],
                         linewidth=lineWidth, color='#004577')
                #plt.scatter(x-10, y, c='k')
            ###########################
            #
            # |--- drawcode = [2,3]
            # |
            #
            ###########################
            elif point_type == 6:
                plt.plot([x, min(x + lineLength, width - 1)], [y, y],
                         linewidth=lineWidth, color='#04d8b2')
                plt.plot([x, x], [y, min(y + lineLength, height - 1)],
                         linewidth=lineWidth, color='#04d8b2')
            ###########################
            #
            #  ---|
            #     | drawcode = [2,4]
            #
            ###########################
            elif point_type == 7:
                plt.plot([x, max(x - lineLength, 0)], [y, y],
                         linewidth=lineWidth, color='#cdfd02')
                plt.plot([x, x], [y, min(y + lineLength, height - 1)],
                         linewidth=lineWidth, color='#cdfd02')
            ###########################
            #    |
            # ---| drawcode = [2,1]
            #
            #
            ###########################
            elif point_type == 4:
                plt.plot([x, max(x - lineLength, 0)], [y, y],
                         linewidth=lineWidth, color='#ff81c0')
                plt.plot([x, x], [y, max(y - lineLength, 0)],
                         linewidth=lineWidth, color='#ff81c0')
            ###########################
            #
            #  |
            #  | drawcode = [2,2]
            #  --
            #
            ###########################
            elif point_type == 5:
                plt.plot([x, min(x + lineLength, width - 1)], [y, y],
                         linewidth=lineWidth, color='#f97306')
                plt.plot([x, x], [y, max(y - lineLength, 0)],
                         linewidth=lineWidth, color='#f97306')
            ###########################
            #
            # |
            # |--- drawcode = [3,4]
            # |
            #
            ###########################
            elif point_type == 11:
                plt.plot([x, min(x + lineLength, width - 1)],
                         [y, y], linewidth=lineWidth, color='b')
                plt.plot([x, x], [y, max(y - lineLength, 0)],
                         linewidth=lineWidth, color='b')
                plt.plot([x, x], [y, min(y + lineLength, height - 1)],
                         linewidth=lineWidth, color='b')
            ###########################
            #
            # ---
            #  |  drawcode = [3,1]
            #  |
            #
            ###########################
            elif point_type == 8:
                plt.plot([x, min(x + lineLength, width - 1)],
                         [y, y], linewidth=lineWidth, color='y')
                plt.plot([x, max(x - lineLength, 0)], [y, y],
                         linewidth=lineWidth, color='y')
                plt.plot([x, x], [y, min(y + lineLength, height - 1)],
                         linewidth=lineWidth, color='y')
            ###########################
            #
            #    |
            # ---| drawcode = [3,2]
            #    |
            #
            ###########################
            elif point_type == 9:
                plt.plot([x, max(x - lineLength, 0)], [y, y],
                         linewidth=lineWidth, color='r')
                plt.plot([x, x], [y, max(y - lineLength, 0)],
                         linewidth=lineWidth, color='r')
                plt.plot([x, x], [y, min(y + lineLength, height - 1)],
                         linewidth=lineWidth, color='r')
            ###########################
            #
            #  |
            #  | drawcode = [3,3]
            # ---
            #
            ###########################
            elif point_type == 10:
                plt.plot([x, min(x + lineLength, width - 1)],
                         [y, y], linewidth=lineWidth, color='m')
                plt.plot([x, max(x - lineLength, 0)], [y, y],
                         linewidth=lineWidth, color='m')
                plt.plot([x, x], [y, max(y - lineLength, 0)],
                         linewidth=lineWidth, color='m')
            ###########################
            #
            #  |
            # --- drawcode = [4,1]
            #  |
            #
            ###########################
            elif point_type == 12:
                plt.plot([x, min(x + lineLength, width - 1)],
                         [y, y], linewidth=lineWidth, color='k')
                plt.plot([x, max(x - lineLength, 0)], [y, y],
                         linewidth=lineWidth, color='k')
                plt.plot([x, x], [y, max(y - lineLength, 0)],
                         linewidth=lineWidth, color='k')
                plt.plot([x, x], [y, min(y + lineLength, height - 1)],
                         linewidth=lineWidth, color='k')

            lineLength = 15 * size
            lineWidth = 15 * size

            ###########################
            # o--- opening left
            ###########################
            if point_type == 13:
                plt.plot([x], [y], 'o',
                         markersize=markersize_large, color='red')
                plt.plot([x], [y], 'o',
                         markersize=markersize_small, color='white')
                plt.text(x, y, 'OL', fontsize=fontsize, color='magenta')
            ###########################
            # ---o opening right
            ###########################
            elif point_type == 14:
                plt.plot([x], [y], 'o',
                         markersize=markersize_large, color='red')
                plt.plot([x], [y], 'o',
                         markersize=markersize_small, color='white')
                plt.text(x, y, 'OR', fontsize=fontsize, color='magenta')
            ###########################
            # o opening up
            # |
            # |
            ###########################
            elif point_type == 15:
                plt.plot([x], [y], 'o',
                         markersize=markersize_large, color='red')
                plt.plot([x], [y], 'o',
                         markersize=markersize_small, color='white')
                plt.text(x, y, 'OU', fontsize=fontsize, color='mediumblue')
            ###########################
            # | opening down
            # |
            # o
            ###########################
            elif point_type == 16:
                plt.plot([x], [y], 'o',
                         markersize=markersize_large, color='red')
                plt.plot([x], [y], 'o',
                         markersize=markersize_small, color='white')
                plt.text(x, y, 'OD', fontsize=fontsize, color='mediumblue')

            ###########################
            #
            # |--- drawcode = [2,3]
            # |
            #
            ###########################
            elif point_type == 17:
                plt.plot([x, min(x + lineLength, width - 1)], [y, y],
                         linewidth=lineWidth, color='indianred')
                plt.plot([x, x], [y, min(y + lineLength, height - 1)],
                         linewidth=lineWidth, color='indianred')
            ###########################
            #
            #  ---|
            #     | drawcode = [2,4]
            #
            ###########################
            elif point_type == 18:
                plt.plot([x, max(x - lineLength, 0)], [y, y],
                         linewidth=lineWidth, color='darkred')
                plt.plot([x, x], [y, min(y + lineLength, height - 1)],
                         linewidth=lineWidth, color='darkred')
            ###########################
            #
            #  |
            #  | drawcode = [2,2]
            #  --
            #
            ###########################
            elif point_type == 19:
                plt.plot([x, min(x + lineLength, width - 1)],
                         [y, y], linewidth=lineWidth, color='salmon')
                plt.plot([x, x], [y, max(y - lineLength, 0)],
                         linewidth=lineWidth, color='salmon')
            ###########################
            #    |
            # ---| drawcode = [2,1]
            #
            #
            ###########################
            elif point_type == 20:
                plt.plot([x, max(x - lineLength, 0)], [y, y],
                         linewidth=lineWidth, color='orangered')
                plt.plot([x, x], [y, max(y - lineLength, 0)],
                         linewidth=lineWidth, color='orangered')

            index += 1


def plot_pre_rec_4(instances, classes):
    walls = ['Wall', 'Railing']
    openings = ['Window', 'Door']
    rooms = ['Outdoor', 'Kitchen',
             'Living Room', 'Bed Room', 'Entry',
             'Dining', 'Storage', 'Garage',
             'Undefined Room', 'Sauna', 'Fire Place',
             'Bathtub', 'Chimney']
    icons = ['Bath',  'Closet',
             'Electrical Appliance', 'Toilet',
             'Shower', 'Sink', 'Sauna', 'Fire Place',
             'Bathtub', 'Chimney']

    def make_sub_plot(classes_to_plot):
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        indx = [classes.index(i) for i in classes_to_plot]
        ins = instances[:, indx].sum(axis=1)

        correct = ins[:, 0]
        false_positive = ins[:, 2]
        false_negatives = ins[:, 1]
        precision = correct / (correct + false_positive)
        recall = correct / (correct + false_negatives)

        plt.step(recall[::-1], precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall[::-1], precision,
                         step='post', alpha=0.2, color='b')

    plt.subplot(2, 2, 1)
    plt.title("Walls")
    make_sub_plot(walls)
    plt.subplot(2, 2, 2)
    plt.title("Openings")
    make_sub_plot(openings)
    plt.subplot(2, 2, 3)
    plt.title("Rooms")
    make_sub_plot(rooms)
    plt.subplot(2, 2, 4)
    plt.title("Icons")
    make_sub_plot(icons)


def discrete_cmap():
    """create a colormap with N (N<15) discrete colors and register it"""
    # define individual colors as hex values
    cpool = ['#DCDCDC', '#b3de69', '#000000', '#8dd3c7', '#fdb462',
             '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969',
             '#577a4d', '#ffffb3']
    cmap3 = colors.ListedColormap(cpool, 'rooms')
    cm.register_cmap(cmap=cmap3)

    cpool = ['#DCDCDC', '#8dd3c7', '#b15928', '#fdb462', '#ffff99',
             '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969',
             '#577a4d']
    cmap3 = colors.ListedColormap(cpool, 'icons')
    cm.register_cmap(cmap=cmap3)

    """create a colormap with N (N<15) discrete colors and register it"""
    # define individual colors as hex values
    cpool = ['#DCDCDC', '#b3de69', '#000000', '#8dd3c7', '#fdb462',
             '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969',
             '#577a4d', '#ffffb3', 'd3d5d7']
    cmap3 = colors.ListedColormap(cpool, 'rooms_furu')
    cm.register_cmap(cmap=cmap3)

    cpool = ['#DCDCDC', '#8dd3c7', '#b15928', '#fdb462', '#ffff99',
             '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969',
             '#577a4d']
    cmap3 = colors.ListedColormap(cpool, 'rooms_furu')
    cm.register_cmap(cmap=cmap3)


def segmentation_plot(rooms_pred, icons_pred, rooms_label, icons_label):
    room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room",
                    "Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
    icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Applience",
                    "Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]
    discrete_cmap()  # custom colormap

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))
    axes[0].set_title('Room Ground Truth')
    axes[0].imshow(rooms_label, cmap='rooms', vmin=0,
                   vmax=len(room_classes) - 1)

    axes[1].set_title('Room Prediction')
    im = axes[1].imshow(rooms_pred, cmap='rooms', vmin=0,
                        vmax=len(room_classes) - 1)

    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=np.arange(12) + 0.5)

    fig.subplots_adjust(right=0.8)
    cbar.ax.set_yticklabels(room_classes)
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))
    axes[0].set_title('Icon Ground Truth')
    axes[0].imshow(icons_label, cmap='icons', vmin=0,
                   vmax=len(icon_classes) - 1)

    axes[1].set_title('Icon Prediction')
    im = axes[1].imshow(icons_pred, cmap='icons', vmin=0,
                        vmax=len(icon_classes) - 1)

    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=np.arange(11) + 0.5)

    fig.subplots_adjust(right=0.8)
    cbar.ax.set_yticklabels(icon_classes)
    plt.show()


def polygons_to_image(polygons, types, room_polygons, room_types, height, width):
    pol_room_seg = np.zeros((height, width))
    pol_icon_seg = np.zeros((height, width))

    for i, pol in enumerate(room_polygons):

        mask = shp_mask(pol, np.arange(width), np.arange(height))

#         jj, ii = draw.polygon(pol[:, 1], pol[:, 0])
        pol_room_seg[mask] = room_types[i]['class']

    for i, pol in enumerate(polygons):
        jj, ii = draw.polygon(pol[:, 1], pol[:, 0])
        if types[i]['type'] == 'wall':
            pol_room_seg[jj, ii] = types[i]['class']
        else:
            pol_icon_seg[jj, ii] = types[i]['class']

    return pol_room_seg, pol_icon_seg


def plot_room(r, name, n_classes=12):
    discrete_cmap()  # custom colormap
    plt.figure(figsize=(40, 30))
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(r, cmap='rooms', vmin=0, vmax=n_classes - 1)
    plt.savefig(name + ".png", format="png")
    plt.show()


def plot_icon(i, name, n_classes=11):
    discrete_cmap()  # custom colormap
    plt.figure(figsize=(40, 30))
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(i, cmap='icons', vmin=0, vmax=n_classes - 1)
    plt.savefig(name + ".png", format="png")
    plt.show()


def plot_heatmaps(h, name):
    for index, i in enumerate(h):
        plt.figure(figsize=(40, 30))
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(i, cmap='Reds', vmin=0, vmax=1)
        plt.savefig(name + str(index) + ".png", format="png")
        plt.show()

def outline_to_mask(line, x, y):
    """Create mask from outline contour

    Parameters
    ----------
    line: array-like (N, 2)
    x, y: 1-D grid coordinates (input for meshgrid)

    Returns
    -------
    mask : 2-D boolean array (True inside)

    Examples
    --------
    >>> from shapely.geometry import Point
    >>> poly = Point(0,0).buffer(1)
    >>> x = np.linspace(-5,5,100)
    >>> y = np.linspace(-5,5,100)
    >>> mask = outline_to_mask(poly.boundary, x, y)
    """
    mpath = mplp.Path(line)
    X, Y = np.meshgrid(x, y)
    points = np.array((X.flatten(), Y.flatten())).T
    mask = mpath.contains_points(points).reshape(X.shape)
    return mask


def _grid_bbox(x, y):
    dx = dy = 0
    return x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2


def _bbox_to_rect(bbox):
    l, r, b, t = bbox
    return Polygon([(l, b), (r, b), (r, t), (l, t)])


def shp_mask(shp, x, y, m=None):
    """
    Adapted from code written by perrette
    form: https://gist.github.com/perrette/a78f99b76aed54b6babf3597e0b331f8
    Use recursive sub-division of space and shapely contains method to create a raster mask on a regular grid.

    Parameters
    ----------
    shp : shapely's Polygon (or whatever with a "contains" method and intersects method)
    x, y : 1-D numpy arrays defining a regular grid
    m : mask to fill, optional (will be created otherwise)

    Returns
    -------
    m : boolean 2-D array, True inside shape.

    Examples
    --------
    >>> from shapely.geometry import Point
    >>> poly = Point(0,0).buffer(1)
    >>> x = np.linspace(-5,5,100)
    >>> y = np.linspace(-5,5,100)
    >>> mask = shp_mask(poly, x, y)
    """
    rect = _bbox_to_rect(_grid_bbox(x, y))

    if m is None:
        m = np.zeros((y.size, x.size), dtype=bool)

    if not shp.intersects(rect):
        m[:] = False

    elif shp.contains(rect):
        m[:] = True

    else:
        k, l = m.shape

        if k == 1 and l == 1:
            m[:] = shp.contains(Point(x[0], y[0]))

        elif k == 1:
            m[:, :l // 2] = shp_mask(shp, x[:l // 2], y, m[:, :l // 2])
            m[:, l // 2:] = shp_mask(shp, x[l // 2:], y, m[:, l // 2:])

        elif l == 1:
            m[:k // 2] = shp_mask(shp, x, y[:k // 2], m[:k // 2])
            m[k // 2:] = shp_mask(shp, x, y[k // 2:], m[k // 2:])

        else:
            m[:k // 2, :l //
                2] = shp_mask(shp, x[:l // 2], y[:k // 2], m[:k // 2, :l // 2])
            m[:k // 2, l //
                2:] = shp_mask(shp, x[l // 2:], y[:k // 2], m[:k // 2, l // 2:])
            m[k // 2:, :l //
                2] = shp_mask(shp, x[:l // 2], y[k // 2:], m[k // 2:, :l // 2])
            m[k // 2:, l //
                2:] = shp_mask(shp, x[l // 2:], y[k // 2:], m[k // 2:, l // 2:])

    return m
