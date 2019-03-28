import torch
import torch.nn.functional as F
import numpy as np
import copy
from itertools import combinations
from scipy import stats
from skimage import draw
from scipy.ndimage import measurements
from shapely.geometry import Polygon
from shapely.ops import unary_union
from collections.abc import Iterable


def get_wall_polygon(wall_heatmaps, room_segmentation, threshold, wall_classes, point_orientations, orientation_ranges):
    wall_lines, wall_points, wall_point_orientation_lines_map = get_wall_lines(wall_heatmaps, room_segmentation, threshold, wall_classes, point_orientations, orientation_ranges)

    walls = np.empty([0, 4, 2], int)
    types = [] 
    wall_lines_new = []
    
    for indx, i in enumerate(wall_lines):
        res = extract_wall_polygon(i, wall_points, room_segmentation, wall_classes)
        if res is not None:
            wall_width, polygon = res
            walls = np.append(walls, [polygon], axis=0)
            wall_type = {'type': 'wall', 'class': i[2]}
            types.append(wall_type)
            wall_lines_new.append(i)

    walls = fix_wall_corners(walls, wall_points, wall_lines_new)
    res = remove_overlapping_walls(walls, types, wall_lines_new)
    walls, types, wall_lines_new = res

    return walls, types, wall_points, wall_lines_new, wall_point_orientation_lines_map


def polygon_intersection(x_min, x_max, y_min, y_max, x_min_label, x_max_label, y_min_label, y_max_label):
    if (x_max > x_min_label and x_max_label > x_min and
       y_max > y_min_label and y_max_label > y_min):
        x_minn = max(x_min, x_min_label)
        x_maxx = min(x_max, x_max_label)
        y_minn = max(y_min, y_min_label)
        y_maxx = min(y_max, y_max_label)
        area = np.sqrt((x_maxx-x_minn)**2+(y_maxx-y_minn)**2)

        return area
    else:
        return 0


def remove_overlapping_walls(walls, types, wall_lines):
    threshold = 0.4
    to_be_removed = set()
    for i, wall1 in enumerate(walls):
        y_min_wall1 = min(wall1[:, 1])
        y_max_wall1 = max(wall1[:, 1])
        x_min_wall1 = min(wall1[:, 0])
        x_max_wall1 = max(wall1[:, 0])
        label_area = np.sqrt((x_max_wall1-x_min_wall1)**2+(y_max_wall1-y_min_wall1)**2)
        for j in range(i+1, len(walls)):
            wall2 = walls[j]
            wall1_dim = calc_polygon_dim(wall1)
            wall2_dim = calc_polygon_dim(wall2)
            if wall1_dim == wall2_dim:
                y_min_wall2 = min(wall2[:, 1])
                y_max_wall2 = max(wall2[:, 1])
                x_min_wall2 = min(wall2[:, 0])
                x_max_wall2 = max(wall2[:, 0])
                intersection = polygon_intersection(x_min_wall1, x_max_wall1, y_min_wall1, y_max_wall1, x_min_wall2, x_max_wall2, y_min_wall2, y_max_wall2)
                pred_area = np.sqrt((x_max_wall2-x_min_wall2)**2+(y_max_wall2-y_min_wall2)**2)
                union = pred_area + label_area - intersection

                iou = intersection / union
                if iou > threshold:
                    if label_area > pred_area:
                        to_be_removed.add(i)
                    else:
                        to_be_removed.add(j)

    walls_new = np.empty([0, 4, 2], int)
    types_new = []
    wall_lines_new = []
    for i in range(len(walls)):
        if i not in to_be_removed:
            walls_new = np.append(walls_new, [walls[i]], axis=0)
            types_new.append(types[i])
            wall_lines_new.append(wall_lines[i])

    return walls_new, types_new, wall_lines_new


def remove_overlapping_openings(polygons, types, classes):
    opening_types = classes['window'] + classes['door']
    good_openings = []
    for i, t in enumerate(types):
        keep = True
        if t['type'] == 'icon' and int(t['class']) in opening_types:
            for j, tt in enumerate(types):
                if not (polygons[j] == polygons[i]).all() and tt['type'] == 'icon' and int(tt['class']) in opening_types:
                    # Different opening
                    if rectangles_overlap(polygons[j], polygons[i]):
                        # The other must be removed.
                        size_i = rectangle_size(polygons[i])
                        size_j = rectangle_size(polygons[j])
                        if size_i == size_j and tt['prob'] > t['prob']:
                            # Fail
                            keep = False
                            break
                        elif size_i < size_j:
                            keep = False
                            break

        good_openings.append(keep)

    new_polygons = polygons[np.array(good_openings)]
    new_types = [t for (t, good) in zip(types, good_openings) if good]

    return new_polygons, new_types


def rectangles_overlap(r1, r2):
        return (range_overlap(min(r1[:, 0]), max(r1[:, 0]), min(r2[:, 0]), max(r2[:, 0]))
                and range_overlap(min(r1[:, 1]), max(r1[:, 1]), min(r2[:, 1]), max(r2[:, 1])))


def range_overlap(a_min, a_max, b_min, b_max):
    '''Neither range is completely greater than the other
    '''
    return (a_min <= b_max) and (b_min <= a_max)


def rectangle_size(r):
    x = max(r[:, 0]) - min(r[:, 0])
    y = max(r[:, 1]) - min(r[:, 1])
    return x*y


def fix_wall_corners(walls, wall_points, wall_lines):
    for i, point in enumerate(wall_points):
        x, y, t1, t2, prob = point
        left = None
        right = None
        up = None
        down = None
        for j, line in enumerate(wall_lines):
            p1, p2, wall_type = line
            dim = calc_line_dim(wall_points, line)
            
            if dim == 0:
                # horizontal
                if p1 == i:
                    right = walls[j], j
                elif p2 == i: 
                    left = walls[j], j
            else:
                # vertical
                if p1 == i:
                    down = walls[j], j
                elif p2 == i: 
                    up = walls[j], j

        # expand right wall to left
        if right and (down or up):
            x1 = np.inf
            x2 = np.inf
            if down:
                x1 = down[0][0, 0]
            if up:
                x2 = up[0][0, 0]

            new_x = min(x1, x2)

            walls[right[1], 0, 0] = new_x
            walls[right[1], 3, 0] = new_x
        
        # expand left to right
        if left and (down or up):
            x1 = 0
            x2 = 0
            if down:
                x1 = down[0][1, 0]
            if up:
                x2 = up[0][1, 0]

            new_x = max(x1, x2)

            walls[left[1], 1, 0] = new_x
            walls[left[1], 2, 0] = new_x

        # expand up to down
        if up and (left or right):
            y1 = np.inf
            y2 = np.inf
            if left:
                y1 = left[0][3, 1]
            if right:
                y2 = right[0][0, 1]

            new_y = min(y1, y2)

            walls[up[1], 2, 1] = new_y
            walls[up[1], 3, 1] = new_y

        # expand up to down
        if down and (left or right):
            y1 = 0
            y2 = 0
            if left:
                y1 = left[0][2, 1]
            if right:
                y2 = right[0][0, 1]

            new_y = max(y1, y2)

            walls[down[1], 0, 1] = new_y
            walls[down[1], 1, 1] = new_y

    return walls


def get_wall_lines(wall_heatmaps, room_segmentation, threshold, wall_classes, point_orientations, orientation_ranges, max_num_points=100):
    _, height, width = room_segmentation.shape
    gap = 10

    wall_points = []
    for i in range(len(wall_heatmaps)):
        info = [int(i / 4), int(i % 4)]
        p = extract_local_max(wall_heatmaps[i], max_num_points, info, threshold, close_point_suppression=True)
        wall_points += p

    point_info = calc_point_info(wall_points, gap, point_orientations, orientation_ranges, height, width)
    wall_lines, wall_point_orientation_lines_map, wall_point_neighbors = point_info

    good_wall_lines = []
    for i1, i2 in wall_lines:
        point1 = wall_points[i1]
        x1 = point1[0]
        y1 = point1[1]
        point2 = wall_points[i2]
        x2 = point2[0]
        y2 = point2[1]

        line_pxls = bresenham_line(x1, y1, x2, y2)
        rooms_on_line = np.array([room_segmentation[:, i[0], i[1]] for i in line_pxls])
        segment = np.argmax(rooms_on_line.sum(axis=0))
        if segment in wall_classes:
            good_wall_lines.append((i1, i2, segment))

    wall_lines = drop_long_walls(good_wall_lines, wall_points)
    v_walls = [line for line in wall_lines if calc_line_dim(wall_points, line)]
    h_walls = [line for line in wall_lines if not calc_line_dim(wall_points, line)]

    connected_walls_v = get_connected_walls(v_walls)
    wall_points = points_to_manhantan(connected_walls_v, wall_points, 0)
    connected_walls_h = get_connected_walls(h_walls)
    wall_points = points_to_manhantan(connected_walls_h, wall_points, 1)

    return wall_lines, wall_points, wall_point_orientation_lines_map

def get_rectangle_polygons(junction_points, size):
    max_x = size[1] - 1
    max_y = size[0] - 1
    x = np.sort(np.concatenate(([0, max_x], np.unique(junction_points[:, 0]))))
    y = np.sort(np.concatenate(([0, max_y], np.unique(junction_points[:, 1]))))

    # number of rectangle polygons
    polygon_count_x = (len(x)-1)
    polygon_count_y = (len(y)-1)
    num_pol = polygon_count_x * polygon_count_y

    polygons = np.zeros((num_pol, 4, 2))

    # we first set the upper left x
    x_up_left = x[:polygon_count_x]
    polygons[:, 0, 0] = np.repeat(x_up_left, polygon_count_y)
    # set upper left y
    y_up_left = y[:polygon_count_y]
    polygons[:, 0, 1] = np.tile(y_up_left, polygon_count_x)

    # set upper right x
    x_up_left = x[1:]
    polygons[:, 1, 0] = np.repeat(x_up_left, polygon_count_y)
    # set upper right y
    y_up_left = y[:polygon_count_y]
    polygons[:, 1, 1] = np.tile(y_up_left, polygon_count_x)

    # set lower right x
    x_up_left = x[1:]
    polygons[:, 2, 0] = np.repeat(x_up_left, polygon_count_y)
    # set lower right y
    y_up_left = y[1:]
    polygons[:, 2, 1] = np.tile(y_up_left, polygon_count_x)

    # set lower left x
    x_up_left = x[:polygon_count_x]
    polygons[:, 3, 0] = np.repeat(x_up_left, polygon_count_y)
    # set lower left y
    y_up_left = y[1:]
    polygons[:, 3, 1] = np.tile(y_up_left, polygon_count_x)

    return polygons

def merge_rectangles(rectangles, room_types):
    # Room polygons to shapely Polygon type
    shapely_polygons = [Polygon(p) for p in rectangles]

    # We initialize array for each classes polygons.
    # polygon_indexes contain n arrays that contain the indexes
    # of the polygons that are of the same class.
    # The class number is the index inner array.
    num_classes = 0
    for r in room_types:
        if r['class'] > num_classes:
            num_classes = r['class']

    polygon_indexes = [[] for i in range(num_classes+1)]

    for i, t in enumerate(room_types):
        polygon_indexes[t['class']].append(i)

    room_polygons = []
    new_room_types = []
    for pol_class, pol_i in enumerate(polygon_indexes):
        if pol_class != 0:  # index 0 is the background and we can ignore it.
            pol_type = {'type': 'room', 'class': pol_class}
            same_cls_pols = []
            for indx in pol_i:
                same_cls_pols.append(shapely_polygons[indx])

            polygon_union = unary_union(same_cls_pols)

            # If there are multiple polygons we split them.
            if isinstance(polygon_union, Iterable):
                for pol in polygon_union:
#                     x, y = pol.boundary.coords.xy
#                     numpy_pol = np.array([np.array(x), np.array(y)]).T
#                     room_polygons.append(numpy_pol)
                    room_polygons.append(pol)
                    new_room_types.append(pol_type)
                    
            else:
#                 x, y = polygon_union.boundary.coords.xy
#                 numpy_pol = np.array([np.array(x), np.array(y)]).T
#                 room_polygons.append(numpy_pol)
                room_polygons.append(polygon_union)
                new_room_types.append(pol_type)

    return room_polygons, new_room_types

def get_polygons(predictions, threshold, all_opening_types):
    heatmaps, room_seg, icon_seg = predictions
    height = icon_seg.shape[1]
    width = icon_seg.shape[2]

    point_orientations = [[(2, ), (3, ), (0, ), (1, )],
                          [(0, 3), (0, 1), (1, 2), (2, 3)],
                          [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)],
                          [(0, 1, 2, 3)]]
    orientation_ranges = [[width, 0, 0, 0],
                          [width, height, width, 0],
                          [width, height, 0, height],
                          [0, height, 0, 0]]

    wall_heatmaps = heatmaps[:13]
    walls = np.empty([0, 4, 2], int)
    wall_layers = [2, 8]
    walls, wall_types, wall_points, wall_lines, wall_point_orientation_lines_map = get_wall_polygon(wall_heatmaps, room_seg, threshold, wall_layers, point_orientations, orientation_ranges)

    icons = np.empty([0, 4, 2], int)
    icons, icon_types = get_icon_polygon(heatmaps, icon_seg, threshold, point_orientations, orientation_ranges)

    openings, opening_types = get_opening_polygon(heatmaps, walls, icon_seg, wall_points, wall_lines, wall_point_orientation_lines_map, threshold, point_orientations, orientation_ranges, all_opening_types)

    # junction_points shape n, 2, coordinate order x, y
    junction_points = get_junction_points(wall_points, wall_lines)
    grid_polygons = get_rectangle_polygons(junction_points, (height, width))

    c, h, w = room_seg.shape
    for i in range(c):
        if i in [2, 8]: # we ignore walls (2) and railings (8)
            room_seg[i] = np.zeros((h, w))
  
    room_seg_2D = np.argmax(room_seg, axis=0)
    room_types = []
    grid_polygons_new = []
    for i, pol in enumerate(grid_polygons):
        room_class = get_polygon_class(pol, room_seg_2D)
        if room_class is not None:
            grid_polygons_new.append(pol)
            room_types.append({'type': 'room', 'class': room_class})

    room_polygons, room_types = merge_rectangles(grid_polygons_new, room_types)

    polygons = np.concatenate([walls, icons, openings])
    types = wall_types + icon_types + opening_types

    classes = {'door': [2], 'window': [1]}

    if len(polygons) > 0:
        polygons, types = remove_overlapping_openings(polygons, types, classes)

    return polygons, types, room_polygons, room_types



def split_by_value(arr, max_val, skip=[]):
    res = np.zeros((max_val, arr.shape[0], arr.shape[1]), dtype=int)
    for i in range(max_val):
        if i not in skip:
            res[i] = np.isin(arr, [i])

    return res


def get_junction_points(wall_points, wall_lines):
    junction_points = np.empty([0, 2], int)
    for wall in wall_lines:
        indx1 = wall[0]
        indx2 = wall[1]
        p1 = np.array(wall_points[indx1][:2])
        junction_points = np.append(junction_points, [p1], axis=0)
        p2 = np.array(wall_points[indx2][:2])
        junction_points = np.append(junction_points, [p2], axis=0)
    
    if len(junction_points) > 0:
        junction_points = np.unique(junction_points, axis=0)

    return junction_points


def get_opening_polygon(heatmaps, wall_polygons, icons_seg, wall_points, wall_lines, wall_point_orientation_lines_map, threshold, point_orientations, orientation_ranges, all_opening_types, max_num_points=100, gap=10):
    height, width = heatmaps.shape[1], heatmaps.shape[2]
    size = height, width
    wall_mask = draw_line_mask(wall_points, wall_lines, height, width)
    # Layer order switch. Must be done to make calc_point_info work.
    door_points = []
    for index, i in enumerate([2, 1, 3, 0]):
        info = [0, index]
        heatmap = heatmaps[i+13]
        heatmap *= wall_mask
        p = extract_local_max(heatmap, max_num_points, info, threshold)
        door_points += p

    point_info = calc_point_info(door_points, gap, point_orientations, orientation_ranges, height, width, True)
    door_lines, door_point_orientation_lines_map, door_point_neighbors = point_info
    
    label_votes_map = np.zeros(icons_seg.shape)
    label_map = np.zeros((30, height, width))
    for segment_index, segmentation_img in enumerate(icons_seg):
        label_votes_map[segment_index] = segmentation_img
        label_map[segment_index] = segmentation_img

    door_types = []
    num_door_types = 2
    door_offset = 23
    for line_index, line in enumerate(door_lines):
        point = door_points[line[0]]
        neighbor_point = door_points[line[1]]
        line_dim = calc_line_dim(door_points, line)
        fixed_value = int(
            round((neighbor_point[1 - line_dim] + point[1 - line_dim]) / 2))
        door_evidence_sums = [0 for type_index in range(num_door_types)]
        for delta in range(int(abs(neighbor_point[line_dim] - point[line_dim]) + 1)):
            intermediate_point = [0, 0]
            intermediate_point[line_dim] = int(
                min(neighbor_point[line_dim], point[line_dim]) + delta)
            intermediate_point[1 - line_dim] = fixed_value
            for type_index in range(num_door_types):
                door_evidence_sums[type_index] += label_map[door_offset + type_index][min(max(
                    intermediate_point[1], 0), height - 1)][min(max(intermediate_point[0], 0), width - 1)]

        door_types.append((line_index, np.argmax(
            door_evidence_sums), np.max(door_evidence_sums)))

    door_types_ori = copy.deepcopy(door_types)
    door_types.sort(key=lambda door_type: door_type[2], reverse=True)

    invalid_doors = {}
    door_conflict_map = {}
    conflict_door_line_pairs = find_conflict_line_pairs(door_points, door_lines, gap)
    for conflict_pair in conflict_door_line_pairs:
        if conflict_pair[0] not in door_conflict_map:
            door_conflict_map[conflict_pair[0]] = []
            pass
        door_conflict_map[conflict_pair[0]].append(conflict_pair[1])

        if conflict_pair[1] not in door_conflict_map:
            door_conflict_map[conflict_pair[1]] = []
            pass
        door_conflict_map[conflict_pair[1]].append(conflict_pair[0])
        continue

    for index, door_type in enumerate(door_types):
        break
        door_index = door_type[0]
        if door_index in invalid_doors:
            continue
        if door_index not in door_conflict_map:
            continue
        for other_index, other_door_type in enumerate(door_types):
            if other_index <= index:
                continue
            other_door_index = other_door_type[0]
            if other_door_index in door_conflict_map[door_index]:
                invalid_doors[other_door_index] = True
                pass
            continue
        continue

    filtered_door_lines = []
    filtered_door_types = []
    for door_index, door in enumerate(door_lines):
        if door_index not in invalid_doors:
            filtered_door_lines.append(door)
            filtered_door_types.append(door_types_ori[door_index][1])
            pass
        continue

    filtered_wall_points = []
    valid_point_mask = {}
    for point_index, orientation_lines_map in enumerate(wall_point_orientation_lines_map):
        if len(orientation_lines_map) == wall_points[point_index][2] + 1:
            filtered_wall_points.append(wall_points[point_index])
            valid_point_mask[point_index] = True

    filtered_wall_lines = []
    for wall_line in wall_lines:
        if wall_line[0] in valid_point_mask and wall_line[1] in valid_point_mask:
            filtered_wall_lines.append(wall_line)

    filtered_door_wall_map = find_line_map_single(door_points, filtered_door_lines,
                                                  wall_points, filtered_wall_lines,
                                                  gap / 2, height, width)
    adjust_door_points(door_points, filtered_door_lines, wall_points,
                       filtered_wall_lines, filtered_door_wall_map)

    opening_polygons = extract_opening_polygon(wall_polygons, door_points, door_lines, size)
    opening_types = get_opening_types(opening_polygons, icons_seg, all_opening_types)

    return opening_polygons, opening_types


def get_opening_types(opening_polygons, icons_seg, all_opening_classes):
    opening_types = []
    for pol in opening_polygons:
        y_1 = min(pol[:, 1])
        y_2 = max(pol[:, 1])
        x_1 = min(pol[:, 0])
        x_2 = max(pol[:, 0])
        
        opening_evidence_sums = icons_seg[all_opening_classes, y_1:y_2+1, x_1:x_2+1].sum(axis=(1, 2))
        opening_class = np.argmax(opening_evidence_sums)
        # if opening_class in all_opening_types:
        opening_area = abs(y_2-y_1)*abs(x_2-x_1)
        opening_types.append({'type': 'icon',
                              'class': all_opening_classes[opening_class],
                              'prob': np.max(opening_evidence_sums) / opening_area})

    return opening_types

def get_icon_polygon(heatmaps, icons_seg, threshold, point_orientations, orientation_ranges, max_num_points=100):
    _, height, width = icons_seg.shape

    icon_points = []
    # Layer order switch. Must be done to make calc_point_info work.
    for index, i in enumerate([3, 2, 0, 1]):
        info = [1, index]
        point = extract_local_max(heatmaps[i+17], max_num_points, info, threshold,
                                  close_point_suppression=True)
        icon_points += point

    gap = 10
    icons = find_icons(icon_points, gap, point_orientations, orientation_ranges, height, width, False)
    icons_good = drop_big_icons(icons, icon_points)
    icon_types_good = []
    icon_polygons = np.empty((0, 4, 2), dtype=int)
    for icon_index, icon in enumerate(icons_good):
        icon_evidence_sums = []
        point_1 = icon_points[icon[0]]
        point_2 = icon_points[icon[1]]
        point_3 = icon_points[icon[2]]
        point_4 = icon_points[icon[3]]
        
        x1 = int((point_1[0] + point_3[0]) / 2)
        x2 = int((point_2[0] + point_4[0]) / 2)
        y1 = int((point_1[1] + point_2[1]) / 2)
        y2 = int((point_3[1] + point_4[1]) / 2)

        icon_area = get_icon_area(icon, icon_points)
        icon_evidence_sums = icons_seg[:, y1:y2+1, x1:x2+1].sum(axis=(1, 2))
        icon_class = np.argmax(icon_evidence_sums)
        icon_polygon = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]])
        if icon_class != 0:
            icon_types_good.append({'type': 'icon',
                                    'class': icon_class,
                                    'prob': np.max(icon_evidence_sums) / icon_area})
            icon_polygons = np.append(icon_polygons, icon_polygon, axis=0)

    return icon_polygons, icon_types_good


def get_connected_walls(walls):
    connected_walls = []
    while walls:
        wall = walls.pop(0)
        wall_inx = set(wall[:2])
        i = 0
        walls_len = len(walls)
        while i < walls_len:
            con_wall_inx = set(walls[i][:2])
            if wall_inx & con_wall_inx:
                wall_inx = wall_inx | con_wall_inx
                walls.pop(i)
                walls_len -= 1
                i = 0
            else:
                i += 1

        connected_walls.append(wall_inx)
        
    return connected_walls

    
def points_to_manhantan(connected_walls, wall_points, line_dim):
    new_wall_points = copy.deepcopy(wall_points)
    for walls in connected_walls:
        summ = 0
        for i in walls:
            summ += wall_points[i][line_dim]

        new_coord = int(np.round(float(summ)/len(walls)))
        for i in walls:
            new_wall_points[i][line_dim] = new_coord

    return new_wall_points


def extract_opening_polygon(wall_polygons, door_points, door_lines, size):
    height = size[0]
    width = size[1]

    opening_polygons = np.empty([0, 4, 2], dtype=int)
    for i, pol in enumerate(wall_polygons):
        polygon_dim = calc_polygon_dim(pol)
        for door_line in door_lines:
            indx1 = door_line[0]
            indx2 = door_line[1]
            point1 = door_points[indx1]
            point2 = door_points[indx2]
            dim = calc_line_dim(door_points, door_line)
            if polygon_dim == dim and points_in_polygon(point1, point2, pol):
                if dim == 0:
                    # horizontal openings
                    p11 = pol[0]
                    p12 = pol[1]
                    p21 = point1[:2]
                    p22 = [point1[0], 0]
                    up_left = get_intersect(p11, p12, p21, p22)

                    p21 = point2[:2]
                    p22 = [point2[0], 0]
                    up_right = get_intersect(p11, p12, p21, p22)

                    p11 = pol[3]
                    p12 = pol[2]
                    p21 = point2[:2]
                    p22 = [point2[0], height-1]
                    down_right = get_intersect(p11, p12, p21, p22)

                    p21 = point1[:2]
                    p22 = [point1[0], height-1]
                    down_left = get_intersect(p11, p12, p21, p22)
                else:
                    # vertical openings
                    p11 = pol[0]
                    p12 = pol[3]
                    p21 = point1[:2]
                    p22 = [0, point1[1]]
                    up_left = get_intersect(p11, p12, p21, p22)

                    p11 = pol[1]
                    p12 = pol[2]
                    p21 = point1[:2]
                    p22 = [width - 1, point1[1]]
                    up_right = get_intersect(p11, p12, p21, p22)

                    p11 = pol[1]
                    p12 = pol[2]
                    p21 = point2[:2]
                    p22 = [width - 1, point2[1]]
                    down_right = get_intersect(p11, p12, p21, p22)

                    p11 = pol[0]
                    p12 = pol[3]
                    p21 = point2[:2]
                    p22 = [0, point2[1]]
                    down_left = get_intersect(p11, p12, p21, p22)

                op_pol = np.array([[up_left, up_right, down_right, down_left]], dtype=int)
                opening_polygons = np.append(opening_polygons, op_pol, axis=0)

    return opening_polygons

def get_polygon_class(polygon, segmentation, remove_layers=[]):
    seg_copy = np.copy(segmentation)
    size = seg_copy.shape

    jj, ii = draw.polygon(polygon[:, 1], polygon[:, 0], shape=size)
    area = seg_copy[jj, ii]
    values, counts = np.unique(area, return_counts=True)
    if len(counts) != 0:
        ind = np.argmax(counts)
        winner_class = values[ind]

        return winner_class
    else:
        return None

def get_intersect(p11, p12, p21, p22):
    # If door point is the same as wall point
    # we do not have to calculate the intersect.
    assert len(p11) == 2
    assert len(p12) == 2
    assert len(p21) == 2
    assert len(p22) == 2
    if np.array_equal(p21, p22):
        return np.array(p21, dtype=int)

    x1 = float(p11[0])
    y1 = float(p11[1])
    x2 = float(p12[0])
    y2 = float(p12[1])
    x3 = float(p21[0])
    y3 = float(p21[1])
    x4 = float(p22[0])
    y4 = float(p22[1])
    a = (x1*y2-y1*x2)
    b = (x3*y4-y3*x4)
    c = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    px = np.round((a * (x3-x4)-(x1-x2) * b) / c)
    py = np.round((a * (y3-y4)-(y1-y2) * b) / c)

    return np.array([px, py], dtype=int)


def points_in_polygon(p1, p2, polygon):
    if point_inside_polygon(p1, polygon) and point_inside_polygon(p2, polygon):
        return True

    return False


def point_inside_polygon(p, polygon):
    x = p[0]
    y = p[1]
    if (x >= polygon[0, 0] and x >= polygon[3, 0] and x <= polygon[1, 0] and x <= polygon[2, 0] and
       y >= polygon[0, 1] and y >= polygon[1, 1] and y <= polygon[2, 1] and y <= polygon[3, 1]):
        return True
    
    return False


def get_wall_seg(wall_polygons, size):
    res = np.zeros(size)
    for pol in wall_polygons:
        jj, ii = draw.polygon(pol[:, 1], pol[:, 0])
        j = []
        i = []
        for indx in range(len(jj)):
            if jj[indx] < size[0] and ii[indx] < size[1]:
                j.append(jj[indx])
                i.append(ii[indx])
        res[j, i] = 1

    return res


def drop_big_icons(icons, icon_points):
    remaining_icons = icons
    bad_icons = []
    remaining_icons = []
    for icon1, icon2 in combinations(icons, 2):
        if icon1 not in bad_icons and icon2 not in bad_icons:
            if icons_same_corner(icon1, icon2):
                area1 = get_icon_area(icon1, icon_points)
                area2 = get_icon_area(icon2, icon_points)
                if area1 <= area2:
                    good_icon = icon1
                    bad_icons.append(icon2)
                else:
                    good_icon = icon2
                    bad_icons.append(icon1)

                if good_icon not in remaining_icons:
                    remaining_icons.append(good_icon)
        else:
            if icon1 not in remaining_icons and icon1 not in bad_icons:
                remaining_icons.append(icon1)
            if icon2 not in remaining_icons and icon2 not in bad_icons:
                remaining_icons.append(icon2)

    res = []
    for icon in remaining_icons:
        if icon not in bad_icons:
            res.append(icon)

    return res


def icons_same_corner(icon1, icon2):
    for i in range(4):
        if icon1[i] == icon2[i]:
            return True

    return False


def drop_long_walls(walls, wall_points):
    bad_walls = []
    remaining_walls = []
    for wall1, wall2 in combinations(walls, 2):
        if wall1 not in bad_walls and wall2 not in bad_walls and walls_same_corner(wall1, wall2, wall_points):
            # if walls_same_corner(wall1, wall2, wall_points):
            length1 = get_wall_length(wall1, wall_points)
            length2 = get_wall_length(wall2, wall_points)
            if length1 <= length2:
                good_wall = wall1
                bad_walls.append(wall2)
            else:
                good_wall = wall2
                bad_walls.append(wall1)

            if good_wall not in remaining_walls:
                remaining_walls.append(good_wall)
        else:
            if wall1 not in remaining_walls and wall1 not in bad_walls:
                remaining_walls.append(wall1)
            if wall2 not in remaining_walls and wall2 not in bad_walls:
                remaining_walls.append(wall2)

    res = []
    for wall in remaining_walls:
        if wall not in bad_walls:
            res.append(wall)

    return res


def walls_same_corner(wall1, wall2, wall_points):
    w1_dim = calc_line_dim(wall_points, wall1)
    w2_dim = calc_line_dim(wall_points, wall2)
    if w1_dim != w2_dim:
        return False
    for i in range(2):
        if wall1[i] == wall2[i]:
            return True

    return False


def extract_wall_polygon(wall, wall_points, segmentation, seg_class):
    _, max_height, max_width = segmentation.shape
    x1 = wall_points[wall[0]][0]
    x2 = wall_points[wall[1]][0]
    y1 = wall_points[wall[0]][1]
    y2 = wall_points[wall[1]][1]
    line_pxls = bresenham_line(x1, y1, x2, y2)
    w_dim = calc_line_dim(wall_points, wall)

    widths = np.array([])

    line_pxls = bresenham_line(x1, y1, x2, y2)
    # strait vertical line
    if w_dim == 1:
        for i in line_pxls:
            w_pos = 0
            w_neg = 0
            j0, i0 = i[0], i[1]
            con = True
            while con and i0 < max_width-1:
                i1 = i0 + 1
                j1 = j0
                pxl_class = get_pxl_class(int(np.floor(i1)), int(np.floor(j1)), segmentation)
                if pxl_class in seg_class:
                    w_pos += 1
                else:
                    con = False
                j0 = j1
                i0 = i1

            j0, i0 = i[0], i[1]
            con = True
            while con and i0 > 0:
                i1 = i0 - 1
                j1 = j0
                pxl_class = get_pxl_class(int(np.floor(i1)), int(np.floor(j1)), segmentation)
                if pxl_class in seg_class:
                    w_neg += 1
                else:
                    con = False
                j0 = j1
                i0 = i1

            widths = np.append(widths, w_pos + w_neg + 1)

        # widths = reject_outliers(widths)
        # if len(widths) == 0:
            # return None
        wall_width = stats.mode(widths).mode[0]
        if wall_width > y2 - y1:
            wall_width = y2 - y1
        w_delta = int(wall_width / 2.0)

        if w_delta == 0:
            return None
        up_left = np.array([x1 - w_delta, y1])
        up_right = np.array([x1 + w_delta, y1])
        down_left = np.array([x2 - w_delta, y2])
        down_right = np.array([x2 + w_delta, y2])
        polygon = np.array([up_left,
                            up_right,
                            down_right,
                            down_left])
        
        polygon[:, 0] = np.clip(polygon[:, 0], 0, max_width)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, max_height)

        return wall_width, polygon

    else:
        for i in line_pxls:
            w_pos = 0
            w_neg = 0
            j0, i0 = i[0], i[1]
            con = True
            while con and j0 < max_height-1:
                i1 = i0
                j1 = j0 + 1
                pxl_class = get_pxl_class(int(np.floor(i1)), int(np.floor(j1)), segmentation)
                if pxl_class in seg_class:
                    w_pos += 1
                else:
                    con = False
                j0 = j1
                i0 = i1

            j0, i0 = i[0], i[1]
            con = True
            while con and j0 > 0:
                i1 = i0
                j1 = j0 - 1
                pxl_class = get_pxl_class(int(np.floor(i1)), int(np.floor(j1)), segmentation)
                if pxl_class in seg_class:
                    w_neg += 1
                else:
                    con = False
                j0 = j1
                i0 = i1

            widths = np.append(widths, w_pos + w_neg + 1)

        # widths = reject_outliers(widths)
        # if len(widths) == 0:
            # return None
        wall_width = stats.mode(widths).mode[0]
        if wall_width > x2 - x1:
            wall_width = x2 - x1
        w_delta = int(wall_width / 2.0)
        if w_delta == 0:
            return None

        down_left = np.array([x1, y1+w_delta])
        down_right = np.array([x2, y2+w_delta])
        up_left = np.array([x1, y1-w_delta])
        up_right = np.array([x2, y2-w_delta])
        polygon = np.array([up_left,
                            up_right,
                            down_right,
                            down_left])

        polygon[:, 0] = np.clip(polygon[:, 0], 0, max_width)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, max_height)

        return wall_width, polygon


def reject_outliers(data, m=0.5):
    data = data[data < 70]
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def get_pxl_class(i, j, segmentation):
    return np.argmax(segmentation[:, j, i])


def get_wall_length(wall, wall_points):
    point1 = wall_points[wall[0]]
    x1 = point1[0]
    y1 = point1[1]
    point2 = wall_points[wall[1]]
    x2 = point2[0]
    y2 = point2[1]

    return np.sqrt((x1-x2)**2+(y1-y2)**2)


def get_icon_area(icon, icon_points):
    point_1 = icon_points[icon[0]]
    point_2 = icon_points[icon[1]]
    point_3 = icon_points[icon[2]]
    point_4 = icon_points[icon[3]]
    
    x_1 = int((point_1[0] + point_3[0]) / 2)
    x_2 = int((point_2[0] + point_4[0]) / 2)
    y_1 = int((point_1[1] + point_2[1]) / 2)
    y_2 = int((point_3[1] + point_4[1]) / 2)

    icon_area = (x_2 - x_1) * (y_2 - y_1)

    return icon_area


def split_validation(tensor, shape, split):
    height = shape[0]
    width = shape[1]
    heatmaps, rooms, icons = torch.split(tensor, [split[0], 1, 1], 1)

    heatmaps = F.interpolate(heatmaps, size=shape, mode='bilinear', align_corners=False).squeeze().data.numpy()
    icons = F.interpolate(icons, size=shape, mode='nearest').squeeze().data.numpy()
    rooms = F.interpolate(rooms, size=shape, mode='nearest').squeeze().data.numpy()

    rooms_new = np.empty([split[1], height, width], float)
    icons_new = np.empty([split[2], height, width], float)

    for i, e in enumerate(icons_new):
        icons_new[i] = np.isin(icons, [float(i)]).astype(float)

    for i, e in enumerate(rooms_new):
        rooms_new[i] = np.isin(rooms, [float(i)]).astype(float)

    return heatmaps, rooms_new, icons_new


def split_prediction(tensor, shape, split):
    tensor = F.interpolate(tensor, size=shape, mode='bilinear', align_corners=False).squeeze()
    heatmaps, rooms, icons = torch.split(tensor, split, 0)

    icons = F.softmax(icons, 0)
    rooms = F.softmax(rooms, 0)

    heatmaps = heatmaps.data.numpy()
    icons = icons.data.numpy()
    rooms = rooms.data.numpy()

    return heatmaps, rooms, icons


def extract_local_max(mask_img, num_points, info, heatmap_value_threshold=0.5,
                      close_point_suppression=False, line_width=5,
                      mask_index=-1, gap=10):
    mask = copy.deepcopy(mask_img)
    height, width = mask.shape
    points = []

    for point_index in range(num_points):
        index = np.argmax(mask)
        y, x = np.unravel_index(index, mask.shape)
        max_value = mask[y, x]
        if max_value <= heatmap_value_threshold:
            return points

        points.append([int(x), int(y)] + info + [max_value, ])

        maximum_suppression(mask, x, y, heatmap_value_threshold)
        if close_point_suppression:
            mask[max(y - gap, 0):min(y + gap, height - 1),
                 max(x - gap, 0):min(x + gap, width - 1)] = 0

    return points


def maximum_suppression(mask, x, y, heatmap_value_threshold):
    height, width = mask.shape
    value = mask[y][x]
    mask[y][x] = -1
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for delta in deltas:
        neighbor_x = x + delta[0]
        neighbor_y = y + delta[1]
        if neighbor_x < 0 or neighbor_y < 0 or neighbor_x >= width or neighbor_y >= height:
            continue
        neighbor_value = mask[neighbor_y][neighbor_x]
        if neighbor_value <= value and neighbor_value > heatmap_value_threshold:
            maximum_suppression(mask, neighbor_x, neighbor_y,
                               heatmap_value_threshold)
            pass
        continue


def calc_point_info(points, gap, point_orientations, orientation_ranges, 
                    height, width, min_distance_only=False,
                    double_direction=False):
    lines = []
    point_orientation_lines_map = []
    point_neighbors = [[] for point in points]

    for point_index, point in enumerate(points):
        point_type = point[2]
        orientations = point_orientations[point_type][point[3]]
        orientation_lines = {}
        for orientation in orientations:
            orientation_lines[orientation] = []

        point_orientation_lines_map.append(orientation_lines)

    for point_index, point in enumerate(points):
        point_type = point[2]
        orientations = point_orientations[point_type][point[3]]
        for orientation in orientations:
            opposite_orientation = (orientation + 2) % 4
            ranges = copy.deepcopy(orientation_ranges[orientation])
            line_dim = -1
            # line_dim 1 is horizontal and line_dim 2 is vertical.
            if orientation == 0 or orientation == 2:
                line_dim = 1
            else:
                line_dim = 0
                pass
            deltas = [0, 0]

            if line_dim == 1:
                deltas[0] = gap
            else:
                deltas[1] = gap
                pass

            for c in range(2):
                ranges[c] = min(ranges[c], point[c] - deltas[c])
                ranges[c + 2] = max(ranges[c + 2], point[c] + deltas[c])
                continue

            neighbor_points = []
            min_distance = max(width, height)
            min_distance_neighbor_point = -1

            for neighbor_point_index, neighbor_point in enumerate(points):
                if (neighbor_point_index <= point_index and not double_direction) or neighbor_point_index == point_index:
                    continue

                neighbor_orientations = point_orientations[neighbor_point[2]][neighbor_point[3]]
                if opposite_orientation not in neighbor_orientations:
                    continue

                in_range = True
                for c in range(2):
                    if neighbor_point[c] < ranges[c] or neighbor_point[c] > ranges[c + 2]:
                        in_range = False
                        break
                    continue

                if not in_range or abs(neighbor_point[line_dim] - point[line_dim]) < max(abs(neighbor_point[1 - line_dim] - point[1 - line_dim]), 1):
                    continue

                if min_distance_only:
                    distance = abs(neighbor_point[line_dim] - point[line_dim])
                    if distance < min_distance:
                        min_distance = distance
                        min_distance_neighbor_point = neighbor_point_index
                        pass
                else:
                    neighbor_points.append(neighbor_point_index)
                    pass
                continue

            if min_distance_only and min_distance_neighbor_point >= 0:
                neighbor_points.append(min_distance_neighbor_point)
                pass

            for neighbor_point_index in neighbor_points:
                neighbor_point = points[neighbor_point_index]

                if double_direction and ((point_index, neighbor_point_index) in lines or (neighbor_point_index, point_index) in lines):
                    continue

                line_index = len(lines)
                point_orientation_lines_map[point_index][orientation].append(
                    line_index)
                point_orientation_lines_map[neighbor_point_index][opposite_orientation].append(
                    line_index)
                point_neighbors[point_index].append(neighbor_point_index)
                point_neighbors[neighbor_point_index].append(point_index)

                if points[point_index][0] + points[point_index][1] < points[neighbor_point_index][0] + points[neighbor_point_index][1]:
                    lines.append((point_index, neighbor_point_index))
                else:
                    lines.append((neighbor_point_index, point_index))
                    pass
                continue
            continue
        continue

    return lines, point_orientation_lines_map, point_neighbors


def draw_line_mask(points, lines, height, width, line_width=5, background_image=None):
    line_mask = np.zeros((height, width))

    for line_index, line in enumerate(lines):
        point_1 = points[line[0]]
        point_2 = points[line[1]]
        line_dim = calc_line_dim(points, line)

        fixed_value = int(
            round((point_1[1 - line_dim] + point_2[1 - line_dim]) / 2))
        min_value = int(min(point_1[line_dim], point_2[line_dim]))
        max_value = int(max(point_1[line_dim], point_2[line_dim]))
        if line_dim == 0:
            line_mask[max(fixed_value - line_width, 0):min(fixed_value + line_width, height), min_value:max_value + 1] = 1
        else:
            line_mask[min_value:max_value + 1, max(fixed_value - line_width, 0):min(fixed_value + line_width, width)] = 1
            pass
        continue

    return line_mask


def find_conflict_line_pairs(points, lines, gap):
    conflict_line_pairs = []
    for line_index_1, line_1 in enumerate(lines):
        point_1 = points[line_1[0]]
        point_2 = points[line_1[1]]
        if point_2[0] - point_1[0] > point_2[1] - point_1[1]:
            line_dim_1 = 0
        else:
            line_dim_1 = 1
            pass

        fixed_value_1 = int(
            round((point_1[1 - line_dim_1] + point_2[1 - line_dim_1]) / 2))
        min_value_1 = int(min(point_1[line_dim_1], point_2[line_dim_1]))
        max_value_1 = int(max(point_1[line_dim_1], point_2[line_dim_1]))

        for line_index_2, line_2 in enumerate(lines):
            if line_index_2 <= line_index_1:
                continue

            point_1 = points[line_2[0]]
            point_2 = points[line_2[1]]
            if point_2[0] - point_1[0] > point_2[1] - point_1[1]:
                line_dim_2 = 0
            else:
                line_dim_2 = 1
                pass

            if (line_1[0] == line_2[0] or line_1[1] == line_2[1]) and line_dim_2 == line_dim_1:
                conflict_line_pairs.append((line_index_1, line_index_2))
                continue

            fixed_value_2 = int(
                round((point_1[1 - line_dim_2] + point_2[1 - line_dim_2]) / 2))
            min_value_2 = int(min(point_1[line_dim_2], point_2[line_dim_2]))
            max_value_2 = int(max(point_1[line_dim_2], point_2[line_dim_2]))

            if line_dim_1 == line_dim_2:
                if abs(fixed_value_2 - fixed_value_1) > gap / 2 or min_value_1 > max_value_2 - gap or min_value_2 > max_value_1 - gap:
                    continue
                conflict_line_pairs.append((line_index_1, line_index_2))
                #draw_lines('test/lines_' + str(line_index_1) + "_" + str(line_index_2) + '.png', width, height, points, [line_1, line_2])
            else:
                if min_value_1 > fixed_value_2 - gap or max_value_1 < fixed_value_2 + gap or min_value_2 > fixed_value_1 - gap or max_value_2 < fixed_value_1 + gap:
                    continue
                conflict_line_pairs.append((line_index_1, line_index_2))
                pass
            continue
        continue

    return conflict_line_pairs


def find_conflict_rectangle_pairs(points, rectangles, gap):
    conflict_rectangle_pairs = []
    for rectangle_index_1, rectangle_1 in enumerate(rectangles):
        for rectangle_index_2, rectangle_2 in enumerate(rectangles):
            if rectangle_index_2 <= rectangle_index_1:
                continue

            conflict = False
            for corner_index in range(4):
                if rectangle_1[corner_index] == rectangle_2[corner_index]:
                    conflict_rectangle_pairs.append(
                        (rectangle_index_1, rectangle_index_2))
                    conflict = True
                    break
                continue

            if conflict:
                continue

            min_x = max(points[rectangle_1[0]][0], points[rectangle_1[2]]
                       [0], points[rectangle_2[0]][0], points[rectangle_2[2]][0])
            max_x = min(points[rectangle_1[1]][0], points[rectangle_1[3]]
                       [0], points[rectangle_2[1]][0], points[rectangle_2[3]][0])
            if min_x > max_x - gap:
                continue
            min_y = max(points[rectangle_1[0]][1], points[rectangle_1[1]]
                       [1], points[rectangle_2[0]][1], points[rectangle_2[1]][1])
            max_y = min(points[rectangle_1[2]][1], points[rectangle_1[3]]
                       [1], points[rectangle_2[2]][1], points[rectangle_2[3]][1])
            if min_y > max_y - gap:
                continue
            conflict_rectangle_pairs.append((rectangle_index_1, rectangle_index_2))
            continue
        continue

    return conflict_rectangle_pairs


def find_icons(points, gap, point_orientations, orientation_ranges,
               height, width, min_distance_only=False,
               max_lengths=(10000, 10000)):
    point_orientation_neighbors_map = []

    for point_index, point in enumerate(points):
        point_type = point[2]
        orientations = point_orientations[point_type][point[3]]
        orientation_neighbors = {}
        for orientation in orientations:
            orientation_neighbors[orientation] = []
            continue
        point_orientation_neighbors_map.append(orientation_neighbors)
        continue

    for point_index, point in enumerate(points):
        point_type = point[2]
        orientations = point_orientations[point_type][point[3]]
        for orientation in orientations:
            opposite_orientation = (orientation + 2) % 4
            ranges = copy.deepcopy(orientation_ranges[orientation])
            line_dim = -1
            if orientation == 0 or orientation == 2:
                line_dim = 1
            else:
                line_dim = 0
                pass
            deltas = [0, 0]

            if line_dim == 1:
                deltas[0] = gap
            else:
                deltas[1] = gap
                pass

            for c in range(2):
                ranges[c] = min(ranges[c], point[c] - deltas[c])
                ranges[c + 2] = max(ranges[c + 2], point[c] + deltas[c])
                continue

            neighbor_points = []
            min_distance = max(width, height)
            min_distance_neighbor_point = -1

            for neighbor_point_index, neighbor_point in enumerate(points):
                if neighbor_point_index <= point_index:
                    continue
                neighbor_orientations = point_orientations[neighbor_point[2]
                                                         ][neighbor_point[3]]
                if opposite_orientation not in neighbor_orientations:
                    continue

                in_range = True
                for c in range(2):
                    if neighbor_point[c] < ranges[c] or neighbor_point[c] > ranges[c + 2]:
                        in_range = False
                        break
                    continue

                if not in_range or abs(neighbor_point[line_dim] - point[line_dim]) < max(abs(neighbor_point[1 - line_dim] - point[1 - line_dim]), gap):
                    continue

                distance = abs(neighbor_point[line_dim] - point[line_dim])
                if distance > max_lengths[line_dim]:
                    continue

                if min_distance_only:
                    if distance < min_distance:
                        min_distance = distance
                        min_distance_neighbor_point = neighbor_point_index
                        pass
                    pass
                else:
                    neighbor_points.append(neighbor_point_index)
                    pass
                continue

            if min_distance_only and min_distance_neighbor_point >= 0:
                neighbor_points.append(min_distance_neighbor_point)
                pass

            for neighbor_point_index in neighbor_points:
                point_orientation_neighbors_map[point_index][orientation].append(
                    neighbor_point_index)
                point_orientation_neighbors_map[neighbor_point_index][opposite_orientation].append(
                    point_index)
                continue
            continue
        continue

    icons = []
    ordered_orientations = (1, 2, 3, 0)
    for point_index_1, orientation_neighbors in enumerate(point_orientation_neighbors_map):
        if ordered_orientations[0] not in orientation_neighbors or ((ordered_orientations[3] + 2) % 4) not in orientation_neighbors:
            continue
        point_indices_4 = orientation_neighbors[(ordered_orientations[3] + 2) % 4]
        for point_index_2 in orientation_neighbors[ordered_orientations[0]]:
            if ordered_orientations[1] not in point_orientation_neighbors_map[point_index_2]:
                continue
            for point_index_3 in point_orientation_neighbors_map[point_index_2][ordered_orientations[1]]:
                if ordered_orientations[2] not in point_orientation_neighbors_map[point_index_3]:
                    continue
                for point_index_4 in point_orientation_neighbors_map[point_index_3][ordered_orientations[2]]:
                    if point_index_4 in point_indices_4:
                        icons.append((point_index_1, point_index_2, point_index_4, point_index_3, (
                            points[point_index_1][4] + points[point_index_2][4] + points[point_index_3][4] + points[point_index_4][4]) / 4))
                        pass
                    continue
                continue
            continue
        continue

    return icons


def calc_line_dim(points, line):
    point_1 = points[line[0]]
    point_2 = points[line[1]]
    if point_2[0] - point_1[0] > point_2[1] - point_1[1]:
        # horizontal
        line_dim = 0
    else:
        # vertical
        line_dim = 1
    return line_dim


def calc_polygon_dim(polygon):
    # polygons are in manhattan world
    # corners are in the order up left, up right, down right, down left
    # first is x and then y coordinate
    x1 = polygon[0, 0]
    x2 = polygon[1, 0]
    y1 = polygon[0, 1]
    y2 = polygon[2, 1]

    if abs(x2 - x1) > abs(y2 - y1):
        # horizontal
        return 0
    else:
        # vertical
        return 1


def find_line_map_single(points, lines, points_2, lines_2, gap, height, width):
    line_map = []
    for line_index, line in enumerate(lines):
        line_dim = calc_line_dim(points, line)
        min_distance = max(width, height)
        min_distance_line_index = -1
        for neighbor_line_index, neighbor_line in enumerate(lines_2):
            neighbor_line_dim = calc_line_dim(points_2, neighbor_line)
            if line_dim != neighbor_line_dim:
                continue

            min_value = max(points[line[0]][line_dim],
                           points_2[neighbor_line[0]][line_dim])
            max_value = min(points[line[1]][line_dim],
                           points_2[neighbor_line[1]][line_dim])
            if max_value - min_value < gap:
                continue
            fixed_value_1 = (points[line[0]][1 - line_dim] +
                            points[line[1]][1 - line_dim]) / 2
            fixed_value_2 = (points_2[neighbor_line[0]][1 - line_dim] +
                            points_2[neighbor_line[1]][1 - line_dim]) / 2

            distance = abs(fixed_value_2 - fixed_value_1)
            if distance < min_distance:
                min_distance = distance
                min_distance_line_index = neighbor_line_index
                pass
            continue

        line_map.append(min_distance_line_index)
        continue

    return line_map


def adjust_door_points(door_points, door_lines, wall_points, wall_lines, door_wall_map):
    for door_line_index, door_line in enumerate(door_lines):
        line_dim = calc_line_dim(door_points, door_line)
        wall_line = wall_lines[door_wall_map[door_line_index]]
        wall_point_1 = wall_points[wall_line[0]]
        wall_point_2 = wall_points[wall_line[1]]
        fixed_value = (wall_point_1[1 - line_dim] + wall_point_2[1 - line_dim]) / 2
        for end_point_index in range(2):
            door_points[door_line[end_point_index]][1 - line_dim] = fixed_value
            continue
        continue


def bresenham_line(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1
    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0
    res = []
    for x in range(dx + 1):
        res.append((y0 + x*xy + y*yy, x0 + x*xx + y*yx))
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy

    return res
