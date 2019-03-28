import math
import numpy as np
from floortrans.loaders.svg_utils import PolygonWall, get_polygon, calc_distance, get_room_number, get_icon, get_icon_number, get_points, get_direction, get_gaussian2D
from xml.dom import minidom
from skimage.draw import polygon
import cv2


all_rooms = {"Background": 0,  # Not in data. The default outside label
             "Alcove": 1,
             "Attic": 2,
             "Ballroom": 3,
             "Bar": 4,
             "Basement": 5,
             "Bath": 6,
             "Bedroom": 7,
             "Below150cm": 8,
             "CarPort": 9,
             "Church": 10,
             "Closet": 11,
             "ConferenceRoom": 12,
             "Conservatory": 13,
             "Counter": 14,
             "Den": 15,
             "Dining": 16,
             "DraughtLobby": 17,
             "DressingRoom": 18,
             "EatingArea": 19,
             "Elevated": 20,
             "Elevator": 21,
             "Entry": 22,
             "ExerciseRoom": 23,
             "Garage": 24,
             "Garbage": 25,
             "Hall": 26,
             "HallWay": 27,
             "HotTub": 28,
             "Kitchen": 29,
             "Library": 30,
             "LivingRoom": 31,
             "Loft": 32,
             "Lounge": 33,
             "MediaRoom": 34,
             "MeetingRoom": 35,
             "Museum": 36,
             "Nook": 37,
             "Office": 38,
             "OpenToBelow": 39,
             "Outdoor": 40,
             "Pantry": 41,
             "Reception": 42,
             "RecreationRoom": 43,
             "RetailSpace": 44,
             "Room": 45,
             "Sanctuary": 46,
             "Sauna": 47,
             "ServiceRoom": 48,
             "ServingArea": 49,
             "Skylights": 50,
             "Stable": 51,
             "Stage": 52,
             "StairWell": 53,
             "Storage": 54,
             "SunRoom": 55,
             "SwimmingPool": 56,
             "TechnicalRoom": 57,
             "Theatre": 58,
             "Undefined": 59,
             "UserDefined": 60,
             "Utility": 61,
             "Wall": 62,
             "Railing": 63,
             "Stairs": 64}

rooms_selected = {"Alcove": 11,
                  "Attic": 11,
                  "Ballroom": 11,
                  "Bar": 11,
                  "Basement": 11,
                  "Bath": 6,
                  "Bedroom": 5,
                  "CarPort": 10,
                  "Church": 11,
                  "Closet": 9,
                  "ConferenceRoom": 11,
                  "Conservatory": 11,
                  "Counter": 11,
                  "Den": 11,
                  "Dining": 4,
                  "DraughtLobby": 7,
                  "DressingRoom": 9,
                  "EatingArea": 4,
                  "Elevated": 11,
                  "Elevator": 11,
                  "Entry": 7,
                  "ExerciseRoom": 11,
                  "Garage": 10,
                  "Garbage": 11,
                  "Hall": 11,
                  "HallWay": 7,
                  "HotTub": 11,
                  "Kitchen": 3,
                  "Library": 11,
                  "LivingRoom": 4,
                  "Loft": 11,
                  "Lounge": 4,
                  "MediaRoom": 11,
                  "MeetingRoom": 11,
                  "Museum": 11,
                  "Nook": 11,
                  "Office": 11,
                  "OpenToBelow": 11,
                  "Outdoor": 1,
                  "Pantry": 11,
                  "Reception": 11,
                  "RecreationRoom": 11,
                  "RetailSpace": 11,
                  "Room": 11,
                  "Sanctuary": 11,
                  "Sauna": 6,
                  "ServiceRoom": 11,
                  "ServingArea": 11,
                  "Skylights": 11,
                  "Stable": 11,
                  "Stage": 11,
                  "StairWell": 11,
                  "Storage": 9,
                  "SunRoom": 11,
                  "SwimmingPool": 11,
                  "TechnicalRoom": 11,
                  "Theatre": 11,
                  "Undefined": 11,
                  "UserDefined": 11,
                  "Utility": 11,
                  "Background": 0,  # Not in data. The default outside label
                  "Wall": 2,
                  "Railing": 8}

room_name_map = {"Alcove": "Room",
                 "Attic": "Room",
                 "Ballroom": "Room",
                 "Bar": "Room",
                 "Basement": "Room",
                 "Bath": "Bath",
                 "Bedroom": "Bedroom",
                 "Below150cm": "Room",
                 "CarPort": "Garage",
                 "Church": "Room",
                 "Closet": "Storage",
                 "ConferenceRoom": "Room",
                 "Conservatory": "Room",
                 "Counter": "Room",
                 "Den": "Room",
                 "Dining": "Dining",
                 "DraughtLobby": "Entry",
                 "DressingRoom": "Storage",
                 "EatingArea": "Dining",
                 "Elevated": "Room",
                 "Elevator": "Room",
                 "Entry": "Entry",
                 "ExerciseRoom": "Room",
                 "Garage": "Garage",
                 "Garbage": "Room",
                 "Hall": "Room",
                 "HallWay": "Entry",
                 "HotTub": "Room",
                 "Kitchen": "Kitchen",
                 "Library": "Room",
                 "LivingRoom": "LivingRoom",
                 "Loft": "Room",
                 "Lounge": "LivingRoom",
                 "MediaRoom": "Room",
                 "MeetingRoom": "Room",
                 "Museum": "Room",
                 "Nook": "Room",
                 "Office": "Room",
                 "OpenToBelow": "Room",
                 "Outdoor": "Outdoor",
                 "Pantry": "Room",
                 "Reception": "Room",
                 "RecreationRoom": "Room",
                 "RetailSpace": "Room",
                 "Room": "Room",
                 "Sanctuary": "Room",
                 "Sauna": "Bath",
                 "ServiceRoom": "Room",
                 "ServingArea": "Room",
                 "Skylights": "Room",
                 "Stable": "Room",
                 "Stage": "Room",
                 "StairWell": "Room",
                 "Storage": "Storage",
                 "SunRoom": "Room",
                 "SwimmingPool": "Room",
                 "TechnicalRoom": "Room",
                 "Theatre": "Room",
                 "Undefined": "Room",
                 "UserDefined": "Room",
                 "Utility": "Room",
                 "Wall": "Wall",
                 "Railing": "Railing",
                 "Background": "Background"}  # Not in data. The default outside label

all_icons = {"Empty": 0,
             "Window": 1,
             "Door": 2,
             "BaseCabinet": 3,
             "BaseCabinetRound": 4,
             "BaseCabinetTriangle": 5,
             "Bathtub": 6,
             "BathtubRound": 7,
             "Chimney": 8,
             "Closet": 9,
             "ClosetRound": 10,
             "ClosetTriangle": 11,
             "CoatCloset": 12,
             "CoatRack": 13,
             "CornerSink": 14,
             "CounterTop": 15,
             "DoubleSink": 16,
             "DoubleSinkRight": 17,
             "ElectricalAppliance": 18,
             "Fireplace": 19,
             "FireplaceCorner": 20,
             "FireplaceRound": 21,
             "GasStove": 22,
             "Housing": 23,
             "Jacuzzi": 24,
             "PlaceForFireplace": 25,
             "PlaceForFireplaceCorner": 26,
             "PlaceForFireplaceRound": 27,
             "RoundSink": 28,
             "SaunaBenchHigh": 29,
             "SaunaBenchLow": 30,
             "SaunaBenchMid": 31,
             "Shower": 32,
             "ShowerCab": 33,
             "ShowerScreen": 34,
             "ShowerScreenRoundLeft": 35,
             "ShowerScreenRoundRight": 36,
             "SideSink": 37,
             "Sink": 38,
             "Toilet": 39,
             "Urinal": 40,
             "WallCabinet": 41,
             "WaterTap": 42,
             "WoodStove": 43,
             "Misc": 44,
             "SaunaBench": 45,
             "SaunaStove": 46,
             "WashingMachine": 47,
             "IntegratedStove": 48,
             "Dishwasher": 49,
             "GeneralAppliance": 50,
             "ShowerPlatform": 51}

icons_selected = {"Window": 1,
                  "Door": 2,
                  "Closet": 3,
                  "ClosetRound": 3,
                  "ClosetTriangle": 3,
                  "CoatCloset": 3,
                  "CoatRack": 3,
                  "CounterTop": 3,
                  "Housing": 3,
                  "ElectricalAppliance": 4,
                  "WoodStove": 4,
                  "GasStove": 4,
                  "Toilet": 5,
                  "Urinal": 5,
                  "SideSink": 6,
                  "Sink": 6,
                  "RoundSink": 6,
                  "CornerSink": 6,
                  "DoubleSink": 6,
                  "DoubleSinkRight": 6,
                  "WaterTap": 6,
                  "SaunaBenchHigh": 7,
                  "SaunaBenchLow": 7,
                  "SaunaBenchMid": 7,
                  "SaunaBench": 7,
                  "Fireplace": 8,
                  "FireplaceCorner": 8,
                  "FireplaceRound": 8,
                  "PlaceForFireplace": 8,
                  "PlaceForFireplaceCorner": 8,
                  "PlaceForFireplaceRound": 8,
                  "Bathtub": 9,
                  "BathtubRound": 9,
                  "Chimney": 10,
                  "Misc": None,
                  "BaseCabinetRound": None,
                  "BaseCabinetTriangle": None,
                  "BaseCabinet": None,
                  "WallCabinet": None,
                  "Shower": None,
                  "ShowerCab": None,
                  "ShowerPlatform": None,
                  "ShowerScreen": None,
                  "ShowerScreenRoundRight": None,
                  "ShowerScreenRoundLeft": None,
                  "Jacuzzi": None}

icon_name_map = {"Window": "Window",
                  "Door": "Door",
                  "Closet": "Closet",
                  "ClosetRound": "Closet",
                  "ClosetTriangle": "Closet",
                  "CoatCloset": "Closet",
                  "CoatRack": "Closet",
                  "CounterTop": "Closet",
                  "Housing": "Closet",
                  "ElectricalAppliance": "ElectricalAppliance",
                  "WoodStove": "ElectricalAppliance",
                  "GasStove": "ElectricalAppliance",
                  "SaunaStove": "ElectricalAppliance",
                  "Toilet": "Toilet",
                  "Urinal": "Toilet",
                  "SideSink": "Sink",
                  "Sink": "Sink",
                  "RoundSink": "Sink",
                  "CornerSink": "Sink",
                  "DoubleSink": "Sink",
                  "DoubleSinkRight": "Sink",
                  "WaterTap": "Sink",
                  "SaunaBenchHigh": "SaunaBench",
                  "SaunaBenchLow": "SaunaBench",
                  "SaunaBenchMid": "SaunaBench",
                  "SaunaBench": "SaunaBench",
                  "Fireplace": "Fireplace",
                  "FireplaceCorner": "Fireplace",
                  "FireplaceRound": "Fireplace",
                  "PlaceForFireplace": "Fireplace",
                  "PlaceForFireplaceCorner": "Fireplace",
                  "PlaceForFireplaceRound": "Fireplace",
                  "Bathtub": "Bathtub",
                  "BathtubRound": "Bathtub",
                  "Chimney": "Chimney",
                  "Misc": None,
                  "BaseCabinetRound": None,
                  "BaseCabinetTriangle": None,
                  "BaseCabinet": None,
                  "WallCabinet": None,
                  "Shower": "None",
                  "ShowerCab": "None",
                  "ShowerPlatform": "None",
                  "ShowerScreen": None,
                  "ShowerScreenRoundRight": None,
                  "ShowerScreenRoundLeft": None,
                  "Jacuzzi": None,
                  "WashingMachine": None,
                  "IntegratedStove": "ElectricalAppliance",
                  "Dishwasher": "ElectricalAppliance",
                  "GeneralAppliance": "ElectricalAppliance"}


class House:
    def __init__(self, path, height, width, icon_list=icons_selected, room_list=rooms_selected):
        self.height = height
        self.width = width
        shape = height, width
        svg = minidom.parse(path)
        self.walls = np.empty((height, width), dtype=np.uint8)
        self.walls.fill(0)
        self.wall_ids = np.empty((height, width), dtype=np.uint8)
        self.wall_ids.fill(0)
        self.icons = np.zeros((height, width), dtype=np.uint8)
        # junction_id = 0
        wall_id = 1
        self.wall_ends = []
        self.wall_objs = []
        self.icon_types = []
        self.room_types = []
        self.icon_corners = {'upper_left': [],
                             'upper_right': [],
                             'lower_left': [],
                             'lower_right': []}
        self.opening_corners = {'left': [],
                                'right': [],
                                'up': [],
                                'down': []}
        self.representation = {'doors': [],
                               'icons': [],
                               'labels': [],
                               'walls': []}

        self.icon_areas = []

        for e in svg.getElementsByTagName('g'):
            try: 
                if e.getAttribute("id") == "Wall":
                    wall = PolygonWall(e, wall_id, shape)
                    wall.rr, wall.cc = self._clip_outside(wall.rr, wall.cc)
                    self.wall_objs.append(wall)
                    self.walls[wall.rr, wall.cc] = room_list["Wall"]
                    self.wall_ids[wall.rr, wall.cc] = wall_id
                    self.wall_ends.append(wall.end_points)

                    wall_id += 1

                if e.getAttribute("id") == "Railing":
                    wall = PolygonWall(e, wall_id, shape)
                    wall.rr, wall.cc = self._clip_outside(wall.rr, wall.cc)
                    self.wall_objs.append(wall)
                    self.walls[wall.rr, wall.cc] = room_list["Railing"]
                    self.wall_ids[wall.rr, wall.cc] = wall_id
                    self.wall_ends.append(wall.end_points)

                    wall_id += 1

            except ValueError as k:
                if str(k) != 'small wall':
                    raise k
                continue

            if e.getAttribute("id") == "Window":
                X, Y = get_points(e)
                rr, cc = polygon(X, Y)
                cc, rr = self._clip_outside(cc, rr)
                direction = get_direction(X, Y)
                locs = np.column_stack((X, Y))
                if direction == 'H':
                    left_index = np.argmin(locs[:, 0])
                    left1 = locs[left_index]
                    locs = np.delete(locs, left_index, axis=0)
                    left_index = np.argmin(locs[:, 0])
                    left2 = locs[left_index]
                    right = np.delete(locs, left_index, axis=0)
                    left = np.array([left1, left2])

                    point_left = left.mean(axis=0)
                    point_right = right.mean(axis=0)
                    self.opening_corners['left'].append(point_left)
                    self.opening_corners['right'].append(point_right)

                    door_rep = [[list(point_left), list(point_right)], ['door', 1, 1]]
                    self.representation['doors'].append(door_rep)
                else:
                    up_index = np.argmin(locs[:, 1])
                    up1 = locs[up_index]
                    locs = np.delete(locs, up_index, axis=0)
                    up_index = np.argmin(locs[:, 1])
                    up2 = locs[up_index]
                    down = np.delete(locs, up_index, axis=0)
                    up = np.array([up1, up2])

                    point_up = up.mean(axis=0)
                    point_down = down.mean(axis=0)
                    self.opening_corners['up'].append(point_up)
                    self.opening_corners['down'].append(point_down)

                    door_rep = [[list(point_up), list(point_down)], ['door', 1, 1]]
                    self.representation['doors'].append(door_rep)

                self.icons[cc, rr] = 1
                self.icon_types.append(1)

            if e.getAttribute("id") == "Door":
                # How to reperesent empty door space
                X, Y = get_points(e)
                rr, cc = polygon(X, Y)
                cc, rr = self._clip_outside(cc, rr)
                direction = get_direction(X, Y)
                locs = np.column_stack((X, Y))
                if direction == 'H':
                    left_index = np.argmin(locs[:, 0])
                    left1 = locs[left_index]
                    locs = np.delete(locs, left_index, axis=0)
                    left_index = np.argmin(locs[:, 0])
                    left2 = locs[left_index]
                    right = np.delete(locs, left_index, axis=0)
                    left = np.array([left1, left2])

                    point_left = left.mean(axis=0)
                    point_right = right.mean(axis=0)
                    self.opening_corners['left'].append(left.mean(axis=0))
                    self.opening_corners['right'].append(right.mean(axis=0))

                    door_rep = [[list(point_left), list(point_right)], ['door', 1, 1]]
                    self.representation['doors'].append(door_rep)
                else:
                    up_index = np.argmin(locs[:, 1])
                    up1 = locs[up_index]
                    locs = np.delete(locs, up_index, axis=0)
                    up_index = np.argmin(locs[:, 1])
                    up2 = locs[up_index]
                    down = np.delete(locs, up_index, axis=0)
                    up = np.array([up1, up2])

                    point_up = up.mean(axis=0)
                    point_down = down.mean(axis=0)
                    self.opening_corners['up'].append(up.mean(axis=0))
                    self.opening_corners['down'].append(down.mean(axis=0))

                    door_rep = [[list(point_up), list(point_down)], ['door', 1, 1]]
                    self.representation['doors'].append(door_rep)

                self.icons[cc, rr] = 2
                self.icon_types.append(2)

            if "FixedFurniture " in e.getAttribute("class"):
                num = get_icon_number(e, icon_list)
                if num is not None:
                    rr, cc, X, Y = get_icon(e)
                    # only four corner icons
                    if len(X) == 4:
                        locs = np.column_stack((X, Y))
                        up_left_index = locs.sum(axis=1).argmin()
                        self.icon_corners['upper_left'].append(locs[up_left_index])
                        up_left = list(locs[up_left_index])
                        locs = np.delete(locs, up_left_index, axis=0)
                        down_right_index = locs.sum(axis=1).argmax()
                        self.icon_corners['lower_right'].append(locs[down_right_index])
                        down_right = list(locs[down_right_index])
                        locs = np.delete(locs, down_right_index, axis=0)
                        up_right_index = locs[:, 1].argmin()
                        self.icon_corners['upper_right'].append(locs[up_right_index])
                        locs = np.delete(locs, up_right_index, axis=0)
                        self.icon_corners['lower_left'].append(locs[0])

                        icon_name = e.getAttribute('class').replace('FixedFurniture ', '').split(' ')[0]
                        icon_name = icon_name_map[icon_name]

                        icon_rep = [[up_left, down_right], [icon_name, 1, 1]]
                        self.representation['icons'].append(icon_rep)

                        rr, cc = self._clip_outside(rr, cc)
                        self.icon_areas.append(len(rr))
                        self.icons[rr, cc] = num
                        self.icon_types.append(num)

            if "Space " in e.getAttribute("class"):
                num = get_room_number(e, room_list)
                rr, cc = get_polygon(e)
                if len(rr) != 0:
                    rr, cc = self._clip_outside(rr, cc)
                    if len(rr) != 0 and len(cc) != 0:
                        self.walls[rr, cc] = num
                        self.room_types.append(num)

                        rr_mean = int(round(np.mean(rr)))
                        cc_mean = int(round(np.mean(cc)))
                        center_box = [[rr_mean-10, cc_mean-10], [rr_mean+10, cc_mean+10]]
                        room_name = e.getAttribute('class').replace('Space ', '').split(' ')[0]
                        room_name = room_name_map[room_name]
                        self.representation['labels'].append([center_box, [room_name, 1, 1]])

            # if "Stairs" in e.getAttribute("class"):
                # for c in e.childNodes:
                    # if c.getAttribute("class") in ["Flight", "Winding"]:
                        # num = room_list["Stairs"]
                        # rr, cc = get_polygon(c)
                        # if len(rr) != 0:
                            # rr, cc = self._clip_outside(rr, cc)
                            # if len(rr) != 0 and len(cc) != 0:
                                # self.walls[rr, cc] = num
                                # self.room_types.append(num)

                                # rr_mean = int(round(np.mean(rr)))
                                # cc_mean = int(round(np.mean(cc)))
                                # center_box = [[rr_mean-10, cc_mean-10], [rr_mean+10, cc_mean+10]]
                                # room_name = "Stairs"
                                # # room_name = room_name_map[room_name]
                                # self.representation['labels'].append([center_box, [room_name, 1, 1]])

        self.avg_wall_width = self.get_avg_wall_width()

        self.new_walls = self.connect_walls(self.wall_objs)

        for w in self.new_walls:
            w.change_end_points()

        for w in self.pillar_walls:
            self.new_walls.append(w)

        self.points = self.lines_to_points(
            self.width, self.height, self.new_walls, self.avg_wall_width)
        self.points = self.merge_joints(self.points, self.avg_wall_width)

        # walls to representation
        for w in self.new_walls:
            end_points = w.end_points.round().astype('int').tolist()
            if w.name == "Wall":
                self.representation['walls'].append([end_points, ['wall', 1, 1]])
            else:
                self.representation['walls'].append([end_points, ['wall', 2, 1]])



    def get_tensor(self):
        heatmaps = self.get_heatmaps()
        wall_t = np.expand_dims(self.walls, axis=0)
        icon_t = np.expand_dims(self.icons, axis=0)
        tensor = np.concatenate((heatmaps, wall_t, icon_t), axis=0)

        return tensor

    def get_segmentation_tensor(self):
        wall_t = np.expand_dims(self.walls, axis=0)
        icon_t = np.expand_dims(self.icons, axis=0)
        tensor = np.concatenate((wall_t, icon_t), axis=0)

        return tensor

    def get_heatmap_dict(self):
        # init dict
        heatmaps = {}
        for i in range(21):
            heatmaps[i] = []

        for p in self.points:
            cord, _, p_type = p
            x = int(np.round(cord[0]))
            y = int(np.round(cord[1]))
            channel = self.get_number(p_type)
            if y < self.height and x < self.width:
                heatmaps[channel-1] = heatmaps[channel-1] + [(x, y)]

        channel = 13
        for i in self.opening_corners['left']:
            y = int(i[1])
            x = int(i[0])
            if y < self.height and x < self.width:
                heatmaps[channel] = heatmaps[channel] + [(x, y)]
        channel += 1
        for i in self.opening_corners['right']:
            y = int(i[1])
            x = int(i[0])
            if y < self.height and x < self.width:
                heatmaps[channel] = heatmaps[channel] + [(x, y)]
        channel += 1
        for i in self.opening_corners['up']:
            y = int(i[1])
            x = int(i[0])
            if y < self.height and x < self.width:
                heatmaps[channel] = heatmaps[channel] + [(x, y)]
        channel += 1
        for i in self.opening_corners['down']:
            y = int(i[1])
            x = int(i[0])
            if y < self.height and x < self.width:
                heatmaps[channel] = heatmaps[channel] + [(x, y)]
        channel += 1

        for i in self.icon_corners['upper_left']:
            y = int(i[1])
            x = int(i[0])
            if y < self.height and x < self.width:
                heatmaps[channel] = heatmaps[channel] + [(x, y)]
        channel += 1
        for i in self.icon_corners['upper_right']:
            y = int(i[1])
            x = int(i[0])
            if y < self.height and x < self.width:
                heatmaps[channel] = heatmaps[channel] + [(x, y)]
        channel += 1
        for i in self.icon_corners['lower_left']:
            y = int(i[1])
            x = int(i[0])
            if y < self.height and x < self.width:
                heatmaps[channel] = heatmaps[channel] + [(x, y)]
        channel += 1
        for i in self.icon_corners['lower_right']:
            y = int(i[1])
            x = int(i[0])
            if y < self.height and x < self.width:
                heatmaps[channel] = heatmaps[channel] + [(x, y)]

        return heatmaps

    def get_heatmaps(self):
        heatmaps = np.zeros((21, self.height, self.width))
        for p in self.points:
            cord, _, p_type = p
            x = int(np.round(cord[0]))
            y = int(np.round(cord[1]))
            channel = self.get_number(p_type)
            if y < self.height and x < self.width:
                heatmaps[channel-1, y, x] = 1

        channel = 13
        for i in self.opening_corners['left']:
            y = int(i[1])
            x = int(i[0])
            if y < self.height and x < self.width:
                heatmaps[channel, y, x] = 1
        channel += 1
        for i in self.opening_corners['right']:
            y = int(i[1])
            x = int(i[0])
            if y < self.height and x < self.width:
                heatmaps[channel, y, x] = 1
        channel += 1
        for i in self.opening_corners['up']:
            y = int(i[1])
            x = int(i[0])
            if y < self.height and x < self.width:
                heatmaps[channel, y, x] = 1
        channel += 1
        for i in self.opening_corners['down']:
            y = int(i[1])
            x = int(i[0])
            if y < self.height and x < self.width:
                heatmaps[channel, y, x] = 1
        channel += 1

        for i in self.icon_corners['upper_left']:
            y = int(i[1])
            x = int(i[0])
            if y < self.height and x < self.width:
                heatmaps[channel, y, x] = 1
        channel += 1
        for i in self.icon_corners['upper_right']:
            y = int(i[1])
            x = int(i[0])
            if y < self.height and x < self.width:
                heatmaps[channel, y, x] = 1
        channel += 1
        for i in self.icon_corners['lower_left']:
            y = int(i[1])
            x = int(i[0])
            if y < self.height and x < self.width:
                heatmaps[channel, y, x] = 1
        channel += 1
        for i in self.icon_corners['lower_right']:
            y = int(i[1])
            x = int(i[0])
            if y < self.height and x < self.width:
                heatmaps[channel, y, x] = 1

        kernel = get_gaussian2D(13)
        for i, h in enumerate(heatmaps):
            heatmaps[i] = cv2.filter2D(h, -1, kernel)

        return heatmaps

    def _clip_outside(self, rr, cc):
        s = np.column_stack((rr, cc))
        s = s[s[:, 0] < self.height]
        s = s[s[:, 1] < self.width]

        return s[:, 0], s[:, 1]

    def lines_to_points(self, width, height, walls, lineWidth):
        lines = [h.end_points for h in walls]

        points = []
        usedLinePointMask = []

        for lineIndex, line in enumerate(lines):
            usedLinePointMask.append([False, False])

        for lineIndex_1, wall_1 in enumerate(walls):
            line_1 = wall_1.end_points

            lineDim_1 = self.get_lineDim(line_1, 1)
            if lineDim_1 <= -1:
                # If wall is diagonal we skip
                continue

            fixedValue_1 = (line_1[0][1 - lineDim_1] +
                            line_1[1][1 - lineDim_1]) / 2
            for lineIndex_2, wall_2 in enumerate(walls):
                line_2 = wall_2.end_points

                if lineIndex_2 <= lineIndex_1:
                    continue

                lineDim_2 = self.get_lineDim(line_2, 1)
                if lineDim_2 + lineDim_1 != 1:
                    # if walls have the same direction we skip
                    continue

                fixedValue_2 = (
                    line_2[0][1 - lineDim_2] + line_2[1][1 - lineDim_2]) / 2
                lineWidth = max(wall_1.max_width, wall_2.max_width)
                nearestPair, minDistance = self.findNearestJunctionPair(
                    line_1, line_2, lineWidth)

                if minDistance <= lineWidth:
                    pointIndex_1 = nearestPair[0]
                    pointIndex_2 = nearestPair[1]
                    if pointIndex_1 > -1 and pointIndex_2 > -1:

                        point = [None, None]
                        point[lineDim_1] = fixedValue_2
                        point[lineDim_2] = fixedValue_1
                        side = [None, None]
                        side[lineDim_1] = line_1[1 -
                                                 pointIndex_1][lineDim_1] - fixedValue_2
                        side[lineDim_2] = line_2[1 -
                                                 pointIndex_2][lineDim_2] - fixedValue_1

                        if side[0] < 0 and side[1] < 0:
                            points.append(
                                [point, point, ['point', 2, 1]])
                        elif side[0] > 0 and side[1] < 0:
                            points.append(
                                [point, point, ['point', 2, 2]])
                        elif side[0] > 0 and side[1] > 0:
                            points.append(
                                [point, point, ['point', 2, 3]])
                        elif side[0] < 0 and side[1] > 0:
                            points.append(
                                [point, point, ['point', 2, 4]])

                        usedLinePointMask[lineIndex_1][pointIndex_1] = True
                        usedLinePointMask[lineIndex_2][pointIndex_2] = True
                    elif (pointIndex_1 > -1 and pointIndex_2 == -1) or (pointIndex_1 == -1 and pointIndex_2 > -1):

                        if pointIndex_1 > -1:
                            lineDim = lineDim_1
                            pointIndex = pointIndex_1
                            fixedValue = fixedValue_2
                            pointValue = line_1[pointIndex_1][1 - lineDim_1]
                            usedLinePointMask[lineIndex_1][pointIndex_1] = True
                        else:
                            lineDim = lineDim_2
                            pointIndex = pointIndex_2
                            fixedValue = fixedValue_1
                            pointValue = line_2[pointIndex_2][1 - lineDim_2]
                            usedLinePointMask[lineIndex_2][pointIndex_2] = True

                        point = [None, None]
                        point[lineDim] = fixedValue
                        point[1 - lineDim] = pointValue

                        if pointIndex == 0:
                            if lineDim == 0:
                                points.append(
                                    [point, point, ['point', 3, 4]])
                            else:
                                points.append(
                                    [point, point, ['point', 3, 1]])
                        else:
                            if lineDim == 0:
                                points.append(
                                    [point, point, ['point', 3, 2]])
                            else:
                                points.append(
                                    [point, point, ['point', 3, 3]])

                elif line_1[0][lineDim_1] < fixedValue_2 and \
                        line_1[1][lineDim_1] > fixedValue_2 and \
                        line_2[0][lineDim_2] < fixedValue_1 and \
                        line_2[1][lineDim_2] > fixedValue_1:
                    point = [None, None]
                    point[lineDim_1] = fixedValue_2
                    point[lineDim_2] = fixedValue_1
                    points.append([point, point, ['point', 4, 1]])

        for lineIndex, pointMask in enumerate(usedLinePointMask):
            lineDim = self.get_lineDim(lines[lineIndex], 1)
            for pointIndex in range(2):
                if pointMask[pointIndex] is True:
                    continue
                point = [lines[lineIndex][pointIndex]
                         [0], lines[lineIndex][pointIndex][1]]
                if pointIndex == 0:
                    if lineDim == 0:
                        points.append([point, point, ['point', 1, 4]])
                    elif lineDim == 1:
                        points.append([point, point, ['point', 1, 1]])
                else:
                    if lineDim == 0:
                        points.append([point, point, ['point', 1, 2]])
                    elif lineDim == 1:
                        points.append([point, point, ['point', 1, 3]])

        return points

    def _pointId2index(self, g, t):
        g_ = g - 1
        t_ = t - 1
        k = g_ * 4 + t_
        return k

    def _index2pointId(self, k):
        g = k // 4 + 1
        t = k % 4 + 1
        return [g, t]

    def _are_close(self, p1, p2, width):
        return calc_distance(p1, p2) < width

    def merge_joints(self, points, wall_width):
        lookuptable = {}
        lookuptable[0] = {0: 0, 1: 7, 2: None, 3: 6, 4: 9,
                          5: 11, 6: 6, 7: 7, 8: 8, 9: 9, 10: 12, 11: 11, 12: 12}
        lookuptable[1] = {0: 7, 1: 1, 2: 4, 3: None, 4: 4,
                          5: 10, 6: 8, 7: 7, 8: 8, 9: 9, 10: 10, 11: 12, 12: 12}
        lookuptable[2] = {0: None, 1: 4, 2: 2, 3: 5, 4: 4, 5: 5,
                          6: 11, 7: 9, 8: 12, 9: 9, 10: 10, 11: 11, 12: 12}
        lookuptable[3] = {0: 6, 1: None, 2: 5, 3: 3, 4: 10,
                          5: 5, 6: 6, 7: 8, 8: 8, 9: 12, 10: 10, 11: 11, 12: 12}
        lookuptable[4] = {0: 9, 1: 4, 2: 4, 3: 10, 4: 4, 5: 10,
                          6: 12, 7: 9, 8: 12, 9: 9, 10: 10, 11: 12, 12: 12}
        lookuptable[5] = {0: 11, 1: 10, 2: 5, 3: 5, 4: 10, 5: 5,
                          6: 11, 7: 12, 8: 12, 9: 12, 10: 10, 11: 11, 12: 12}
        lookuptable[6] = {0: 6, 1: 8, 2: 11, 3: 6, 4: 12, 5: 11,
                          6: 6, 7: 8, 8: 8, 9: 12, 10: 12, 11: 11, 12: 12}
        lookuptable[7] = {0: 7, 1: 7, 2: 9, 3: 8, 4: 9, 5: 12,
                          6: 8, 7: 7, 8: 8, 9: 9, 10: 12, 11: 12, 12: 12}
        lookuptable[8] = {0: 8, 1: 8, 2: 12, 3: 8, 4: 12, 5: 12,
                          6: 8, 7: 8, 8: 8, 9: 12, 10: 12, 11: 12, 12: 12}
        lookuptable[9] = {0: 9, 1: 9, 2: 9, 3: 12, 4: 9, 5: 12,
                          6: 12, 7: 9, 8: 12, 9: 9, 10: 12, 11: 12, 12: 12}
        lookuptable[10] = {0: 12, 1: 10, 2: 10, 3: 10, 4: 10,
                           5: 10, 6: 12, 7: 12, 8: 12, 9: 12, 10: 10, 11: 12, 12: 12}
        lookuptable[11] = {0: 11, 1: 12, 2: 11, 3: 11, 4: 12,
                           5: 11, 6: 11, 7: 12, 8: 12, 9: 12, 10: 12, 11: 11, 12: 12}
        lookuptable[12] = {0: 12, 1: 12, 2: 12, 3: 12, 4: 12,
                           5: 12, 6: 12, 7: 12, 8: 12, 9: 12, 10: 12, 11: 12, 12: 12}

        newPoints = []
        merged = [False] * len(points)
        for i, point1 in enumerate(points):
            if merged[i] is False:
                pool = [point1]
                for j, point2 in enumerate(points):
                    if j != i and merged[j] is False and self._are_close(point1[0], point2[0], wall_width):
                        merged[j] = True
                        pool.append(point2)

                if len(pool) == 1:
                    newPoints.append(point1)
                    merged[i] = True
                else:
                    p_ = pool[0]
                    for point_id in range(1, len(pool)):
                        merge_to_p = pool[point_id]

                        k_ = self._pointId2index(p_[2][1], p_[2][2])
                        k_merge_to_p = self._pointId2index(
                            merge_to_p[2][1], merge_to_p[2][2])

                        knew = lookuptable[k_][k_merge_to_p]
                        if knew is None:
                            continue

                        typenew = self._index2pointId(knew)
                        p_ = [p_[0], p_[1], ['point', typenew[0], typenew[1]]]

                        newPoints.append(p_)

        return newPoints

    def get_avg_wall_width(self):
        res = 0
        for i, w in enumerate(self.wall_objs):
            res += w.max_width
        res = res / float(i)

        return res

    def connect_walls(self, walls):
        new_walls = []
        num_walls = len(walls)
        remaining_walls = list(range(1, num_walls + 1))

        # getting pillars
        remaining_pillar_ids = []
        for p_id in range(1, num_walls + 1):
            p_wall = self.find_wall_by_id(p_id, walls)
            if p_wall.wall_is_pillar(self.avg_wall_width):
                for wall_id in range(1, num_walls + 1):
                    wall = self.find_wall_by_id(wall_id, walls)
                    if p_wall.merge_possible(wall):
                        break
                else:
                    remaining_walls.pop(remaining_walls.index(p_wall.id))
                    remaining_pillar_ids.append(p_wall.id)

        while (len(remaining_walls) > 0):
            new_wall_id = remaining_walls.pop(0)
            new_wall = self.find_wall_by_id(new_wall_id, walls)

            found = True
            while (found):
                found = False
                for merge_wall_id in remaining_walls:
                    merged = self.find_wall_by_id(merge_wall_id, walls)
                    temp_wall = new_wall.merge_walls(merged)

                    if temp_wall is not None:
                        remaining_walls.pop(remaining_walls.index(merged.id))
                        new_wall = temp_wall
                        found = True

            new_walls.append(new_wall)

        # connect pillars to walls
        new_wall_id = num_walls + 1
        self.pillar_walls = []
        for id in remaining_pillar_ids:
            w = self.find_wall_by_id(id, walls)
            pws = w.split_pillar_wall(new_wall_id, self.avg_wall_width)
            new_wall_id += 4
            for pw in pws:
                self.pillar_walls.append(pw)

        return new_walls

    def get_number(self, x):
        return (x[1] - 1) * 4 + x[2]

    def get_lineDim(self, line, lineWidth):
        lineWidth = lineWidth or 1
        if abs(line[0][0] - line[1][0]) > abs(line[0][1] - line[1][1]) and \
                abs(line[0][1] - line[1][1]) <= lineWidth:
            return 0
        elif abs(line[0][1] - line[1][1]) > abs(line[0][0] - line[1][0]) and \
                abs(line[0][0] - line[1][0]) <= lineWidth:
            return 1
        else:
            return -1

    def findNearestJunctionPair(self, line_1, line_2, gap):

        minDistance = None
        for index_1 in range(0, 2):
            for index_2 in range(0, 2):

                distance = calc_distance(line_1[index_1], line_2[index_2])
                if minDistance is None or distance < minDistance:
                    nearestPair = [index_1, index_2]
                    minDistance = distance

        if minDistance > gap:
            lineDim_1 = self.get_lineDim(line_1, 1)
            lineDim_2 = self.get_lineDim(line_2, 1)

            if lineDim_1 + lineDim_2 == 1:
                fixedValue_1 = (line_1[0][1 - lineDim_1] +
                                line_1[1][1 - lineDim_1]) / 2
                fixedValue_2 = (line_2[0][1 - lineDim_2] +
                                line_2[1][1 - lineDim_2]) / 2

                if line_2[0][lineDim_2] < fixedValue_1 and line_2[1][lineDim_2] > fixedValue_1:
                    for index in range(2):
                        distance = abs(line_1[index][lineDim_1] - fixedValue_2)
                        if distance < minDistance:
                            nearestPair = [index, -1]
                            minDistance = distance

                if line_1[0][lineDim_1] < fixedValue_2 and line_1[1][lineDim_1] > fixedValue_2:
                    for index in range(2):
                        distance = abs(line_2[index][lineDim_2] - fixedValue_1)
                        if distance < minDistance:
                            nearestPair = [-1, index]
                            minDistance = distance

        return nearestPair, minDistance

    def find_wall_by_id(self, id, walls):
        for wall in walls:
            if wall.id == id:
                return wall

        return None
