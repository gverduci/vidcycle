import unittest
from unittest.mock import MagicMock
from render import PanelRenderer
from coordinate import GarminSegment, GarminCoordinate
from video import GoProVideo
from filecmp import cmp
import os
import json

class TestStringMethods(unittest.TestCase):

    def test_map(self):
        dirname = os.path.dirname(__file__)
        filenameGeoJson = os.path.join(dirname, '../testResources/path.geojson')
        # read coords.geojson to get coordinates anc use them to create a GarminSegment
        with open(filenameGeoJson, 'r') as file:
            data = json.load(file)

        coordinates = []
        for feature in data['features']:
            coords = feature['geometry']['coordinates']
            for coord in coords[0]:
                coordinates.append(GarminCoordinate(position_lat=coord[0], position_long=coord[1], distance=0, temperature=0, timestamp=0, heart_rate=120))


        segment = GarminSegment(coordinates=coordinates)
        subsegment = GarminSegment(coordinates=[GarminCoordinate(position_lat=-74.0147156, position_long=40.6409009, distance=0, temperature=0, timestamp=0, heart_rate=120)])

        # Mock data for testing
        GoProVideo.get_resolution = MagicMock(return_value=(1920, 1080))
        video = GoProVideo(video_paths=["video.mp4"], video_hours_delta=0)
        
        output_folder = "panel"
        thread_number = 1
        panel_width = 0.2
        map_height = 0.3
        map_opacity = 0.9
        map_marker_inner_size = 15
        map_marker_inner_opacity = 1.0
        map_marker_outer_size = 30
        map_marker_outer_opacity = 0.5  
        stat_keys_and_labels = []
        stats_x_position = 0.15
        stats_y_range = (0.2, 0.7)
        stat_label_y_position_delta = 0.12
        font_size = 80
        label_font_size = 50
        stats_opacity = 0.9
        
        PanelRenderer.file_name = MagicMock(return_value="test_map")

        renderer = PanelRenderer(
            segment,
            subsegment,
            video,
            output_folder,
            thread_number,
            panel_width,
            map_height,
            map_opacity,
            map_marker_inner_size,
            map_marker_inner_opacity,
            map_marker_outer_size,
            map_marker_outer_opacity,
            stat_keys_and_labels,
            stats_x_position,
            stats_y_range,
            stat_label_y_position_delta,
            font_size,
            label_font_size,
            stats_opacity
        )

        renderer.render()
        
        filenameReference = os.path.join(dirname, '..', 'testResources', 'test_map.png')
        filenameoutput = os.path.join(dirname, "..", "panel", "test_map.png")
        self.assertTrue(cmp(filenameReference, filenameoutput), 'The images are not the same')

    def test_stats(self):
        dirname = os.path.dirname(__file__)
        coordinates = [
            GarminCoordinate(position_lat=-74.0143669, position_long=40.641194, distance=0, temperature=0, timestamp=0, heart_rate=120),
            GarminCoordinate(position_lat=-74.0144956, position_long=40.6410759, distance=0, temperature=0, timestamp=0, heart_rate=120),
            GarminCoordinate(position_lat=-74.0147156, position_long=40.6409009, distance=0, temperature=0, timestamp=0, heart_rate=120)
        ]


        segment = GarminSegment(coordinates=coordinates)
        subsegment = GarminSegment(coordinates=[GarminCoordinate(position_lat=-74.0147156, position_long=40.6409009, distance=0, temperature=0, timestamp=0, heart_rate=120)])

        # Mock data for testing
        GoProVideo.get_resolution = MagicMock(return_value=(1920, 1080))
        video = GoProVideo(video_paths=["video.mp4"], video_hours_delta=0)
        
        output_folder = "panel"
        thread_number = 1
        panel_width = 0.2
        map_height = 0.3
        map_opacity = 0.9
        map_marker_inner_size = 15
        map_marker_inner_opacity = 1.0
        map_marker_outer_size = 30
        map_marker_outer_opacity = 0.5  
        stat_keys_and_labels = [["heart_rate", "bpm", []],[
                "enhanced_speed",
                "km/h",
                []
            ],
            [
                "enhanced_altitude",
                "m",
                []
            ]]
        stats_x_position = 0.15
        stats_y_range = (0.2, 0.7)
        stat_label_y_position_delta = 0.12
        font_size = 80
        label_font_size = 50
        stats_opacity = 0.9

        PanelRenderer.file_name = MagicMock(return_value="test_stats")
        
        renderer = PanelRenderer(
            segment,
            subsegment,
            video,
            output_folder,
            thread_number,
            panel_width,
            map_height,
            map_opacity,
            map_marker_inner_size,
            map_marker_inner_opacity,
            map_marker_outer_size,
            map_marker_outer_opacity,
            stat_keys_and_labels,
            stats_x_position,
            stats_y_range,
            stat_label_y_position_delta,
            font_size,
            label_font_size,
            stats_opacity
        )

        renderer.render()

        filenameReference = os.path.join(dirname, '..', 'testResources', 'test_stats.png')
        filenameoutput = os.path.join(dirname, "..", "panel", "test_stats.png")
        self.assertTrue(cmp(filenameReference, filenameoutput), 'The images are not the same')

    def test_stats_range(self):
        dirname = os.path.dirname(__file__)
        coordinates = [
            GarminCoordinate(position_lat=-74.0143669, position_long=40.641194, distance=0, temperature=0, timestamp=0, heart_rate=120),
            GarminCoordinate(position_lat=-74.0144956, position_long=40.6410759, distance=0, temperature=0, timestamp=0, heart_rate=120),
            GarminCoordinate(position_lat=-74.0147156, position_long=40.6409009, distance=0, temperature=0, timestamp=0, heart_rate=120)
        ]


        segment = GarminSegment(coordinates=coordinates, laps =[], hr_zone_high_boundary=[84, 101, 118, 134, 151, 168])
        subsegment = GarminSegment(coordinates=[GarminCoordinate(position_lat=-74.0147156, position_long=40.6409009, distance=0, temperature=0, timestamp=0, heart_rate=120)], laps =[], hr_zone_high_boundary=[84, 101, 118, 134, 151, 168])

        # Mock data for testing
        GoProVideo.get_resolution = MagicMock(return_value=(1920, 1080))
        video = GoProVideo(video_paths=["video.mp4"], video_hours_delta=0)
        
        output_folder = "panel"
        thread_number = 1
        panel_width = 0.2
        map_height = 0.3
        map_opacity = 0.9
        map_marker_inner_size = 15
        map_marker_inner_opacity = 1.0
        map_marker_outer_size = 30
        map_marker_outer_opacity = 0.5  
        stat_keys_and_labels = [
            [
                "heart_rate",
                "bpm", 
                [
                    {"color": "#d6d6d6", "label": "Rest" },
                    {"color": "#a6a6a6", "label": "Warm Up" },
                    {"color": "#3b97f3", "label": "Easy"},
                    {"color": "#82c91e", "label": "Aerobic"},
                    {"color": "#f98925", "label": "Threshold"},
                    {"color": "#d32020", "label": "Maximum"}
                ]
            ],
            [
                "enhanced_speed",
                "km/h",
                []
            ],
            [
                "enhanced_altitude",
                "m",
                []
            ]
        ]
        stats_x_position = 0.15
        stats_y_range = (0.2, 0.7)
        stat_label_y_position_delta = 0.12
        font_size = 80
        label_font_size = 50
        stats_opacity = 0.9

        PanelRenderer.file_name = MagicMock(return_value="test_stats_range")
        
        renderer = PanelRenderer(
            segment,
            subsegment,
            video,
            output_folder,
            thread_number,
            panel_width,
            map_height,
            map_opacity,
            map_marker_inner_size,
            map_marker_inner_opacity,
            map_marker_outer_size,
            map_marker_outer_opacity,
            stat_keys_and_labels,
            stats_x_position,
            stats_y_range,
            stat_label_y_position_delta,
            font_size,
            label_font_size,
            stats_opacity
        )

        renderer.render()

        filenameReference = os.path.join(dirname, '..', 'testResources', 'test_stats_range.png')
        filenameoutput = os.path.join(dirname, "..", "panel", "test_stats_range.png")
        self.assertTrue(cmp(filenameReference, filenameoutput), 'The images are not the same')        
        

if __name__ == '__main__':
    unittest.main()