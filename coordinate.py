from datetime import datetime, timedelta, timezone
import json
from typing import List, Optional, Dict, Any, Tuple
import geopy.distance
import functools
from garmin_fit_sdk import Decoder, Stream
import subprocess
import gpxpy
import csv
from copy import copy


class Coordinate:
    def __init__(
        self,
        timestamp: datetime,
        latitude: Optional[float],
        longitude: Optional[float],
    ) -> None:
        self.timestamp = timestamp
        self.latitude = latitude
        self.longitude = longitude

    def __copy__(self) -> "Coordinate":
        return type(self)(self.timestamp, self.latitude, self.longitude)

    def __str__(self) -> str:
        return json.dumps(self.__dict__, indent=4, default=str)

    def set_timestamp(self, timestamp: datetime):
        self.timestamp = timestamp

    def weighted_average(
        self, other_coordinate: "Coordinate", other_weight: float
    ) -> "Coordinate":
        assert 0.0 <= other_weight <= 1.0
        self_weight = 1.0 - other_weight

        return Coordinate(
            timestamp=datetime.fromtimestamp(
                (self.timestamp.timestamp() * self_weight)
                + (other_coordinate.timestamp.timestamp() * other_weight),
                tz=self.timestamp.tzinfo,
            ),
            latitude=(self.latitude * self_weight)
            + (other_coordinate.latitude * other_weight),
            longitude=(self.longitude * self_weight)
            + (other_coordinate.longitude * other_weight),
        )

    def distance(self, other: "Coordinate") -> "Coordinate":
        return geopy.distance.geodesic(
            (self.latitude, self.longitude), (other.latitude, other.longitude)
        ).km

    @staticmethod
    def load_coordinates_from_video_file(video_file_path: str) -> List["Coordinate"]:
        output = subprocess.run(
            [
                "exiftool",
                "-ee",
                "-p",
                "gpx.fmt",
                "-api",
                "largefilesupport=1",
                video_file_path,
            ],
            capture_output=True,
        )
        out = output.stdout
        gpx = gpxpy.parse(out)
        points = gpx.tracks[0].segments[0].points
        return [
            Coordinate(
                timestamp=point.time,
                latitude=point.latitude,
                longitude=point.longitude,
            )
            for point in points
            if point.latitude != 0 and point.longitude != 0
        ]


class GarminCoordinate(Coordinate):
    INT_TO_FLOAT_LAT_LONG_CONST = 11930465

    def __init__(
        self,
        timestamp: datetime,
        distance: float,
        temperature: int,
        altitude: Optional[float] = None,
        enhanced_altitude: Optional[float] = None,
        heart_rate: Optional[int] = None,
        speed: Optional["Speed"] = None,
        enhanced_speed: Optional["Speed"] = None,
        position_lat: Optional[int] = None,
        position_long: Optional[int] = None,
        power: Optional[int] = None,
        cadence: Optional[int] = None,
        slope: Optional["Slope"] = None,
        **_kwargs: Dict[str, Any]
    ):
        self.position_lat = position_lat
        self.position_long = position_long

        if position_lat is not None:
            position_lat /= self.INT_TO_FLOAT_LAT_LONG_CONST
        if position_long is not None:
            position_long /= self.INT_TO_FLOAT_LAT_LONG_CONST

        super().__init__(timestamp, position_lat, position_long)

        self.distance = distance
        self.altitude = altitude
        self.enhanced_altitude = enhanced_altitude
        self.speed = speed
        self.enhanced_speed = enhanced_speed
        self.heart_rate = heart_rate
        self.temperature = temperature
        self.power = power
        self.cadence = cadence
        self.slope = slope

    def __copy__(self) -> "GarminCoordinate":
        return type(self)(
            self.timestamp,
            self.distance,
            self.temperature,
            self.altitude,
            self.enhanced_altitude,
            self.heart_rate,
            self.speed,
            self.enhanced_speed,
            self.position_lat,
            self.position_long,
            self.power,
            self.cadence,
            self.slope,
        )

    def set_slope(self, slope):
        self.slope = slope

    def __str__(self) -> str:
        return json.dumps(self.__dict__, indent=4, default=str)

    def weighted_average(
        self, other_coordinate: "GarminCoordinate", other_weight: float
    ) -> "GarminCoordinate":
        super_coordinate = super().weighted_average(other_coordinate, other_weight)
        kwargs = {
            "timestamp": super_coordinate.timestamp,
            "position_long": super_coordinate.longitude
            * self.INT_TO_FLOAT_LAT_LONG_CONST,
            "position_lat": super_coordinate.latitude
            * self.INT_TO_FLOAT_LAT_LONG_CONST,
        }

        self_weight = 1.0 - other_weight

        other_dict = other_coordinate.__dict__
        self_dict = self.__dict__
        for key in self_dict:
            if (
                key in ["latitude", "longitude", "timestamp"]
                or other_dict[key] is None
                or self_dict[key] is None
            ):
                continue
            kwargs[key] = (self_dict[key] * self_weight) + (
                other_dict[key] * other_weight
            )

        garmin_coordinate = GarminCoordinate(**kwargs)

        return garmin_coordinate


class Segment:
    def __init__(self, coordinates: List[Coordinate]) -> None:
        self.coordinates: List[Coordinate] = self._get_filtered_coordinates(coordinates)

    def _get_filtered_coordinates(
        self, coordinates: List[Coordinate]
    ) -> List[Coordinate]:
        reversed_coordinates = []
        for coordinate in coordinates[::-1]:
            if (
                coordinate is not None
                and coordinate.latitude is not None
                and coordinate.longitude is not None
                and (
                    len(reversed_coordinates) == 0
                    or Coordinate.distance(coordinate, reversed_coordinates[-1]) < 1
                )
            ):
                reversed_coordinates.append(coordinate)
        return reversed_coordinates[::-1]

    # TODO: optimize using binary search
    @functools.lru_cache(maxsize=None)
    def get_coordinate(self, time: datetime) -> Optional[Coordinate]:
        result = None

        for a, b in zip(self.coordinates[:-1], self.coordinates[1:]):
            if a.timestamp <= time <= b.timestamp:
                a_timestamp = a.timestamp.timestamp()
                b_timestamp = b.timestamp.timestamp()

                time_delta = b_timestamp - a_timestamp
                # why care if there is a gap > 1.5 secs?
                # because this indicates gps stopped recording
                if time_delta < 0.0001 or time_delta > 1.5:
                    result = copy(a)
                    break

                weight = (b_timestamp - time.timestamp()) / (b_timestamp - a_timestamp)
                result = a.weighted_average(b, 1.0 - weight)
                break

        if result is not None:
            result.set_timestamp(time)

        return result

    def get_start_time(self) -> datetime:
        return self.coordinates[0].timestamp

    def get_end_time(self) -> datetime:
        return self.coordinates[-1].timestamp

    def get_length(self) -> timedelta:
        return self.get_end_time() - self.get_start_time()

    def get_iterator(self, iterator_step_length: timedelta):
        return SegmentIterator(self, iterator_step_length)

    def _get_coordinates(
        self, start_time: datetime, end_time: datetime, step_length: timedelta
    ) -> List[Coordinate]:
        new_coordinates = []
        while start_time <= end_time:
            new_coordinates.append(self.get_coordinate(start_time))
            start_time += step_length
        return new_coordinates
    
    def _extend_first_coordinates(
        self, garmin_start_time: datetime, start_time: datetime, end_time: datetime, step_length: timedelta
    ) -> List[Coordinate]:
        new_coordinates = []
        new_coordinate = self.get_coordinate(garmin_start_time)
        while start_time <= end_time:
            new_coordinates.append(new_coordinate)
            start_time += step_length
        return new_coordinates

    def get_subsegment(
        self, start_time: datetime, end_time: datetime, step_length: timedelta
    ) -> "Segment":
        new_coordinates: List[Coordinate] = self._get_coordinates(
            start_time, end_time, step_length
        )
        return Segment(new_coordinates)

    def write_to_csv(self, file_path):
        with open(file_path, "w") as csvfile:
            writer = csv.writer(csvfile)

            for coordinate in self.coordinates:
                writer.writerow(
                    [
                        int(coordinate.timestamp.timestamp()),
                        coordinate.latitude,
                        coordinate.longitude,
                    ]
                )

        csvfile.close()

    def get_xy_pair(self) -> Tuple[float, float]:
        return 0.0, 0.0

class Slope():
    def __init__(self, percentage: float) -> None:
        self.percentage = percentage

    def __str__(self) -> str:
        return f"{self.percentage:.0f}"
    
    def __repr__(self) -> str:
        return f"Slope(percentage={self.percentage})"
    
    def __add__(self, other: "Slope") -> "Slope":
        return Slope(self.percentage + other.percentage)

    def __sub__(self, other: "Slope") -> "Slope":
        return Slope(self.percentage - other.percentage)
    
    def __mul__(self, other_speed: Any) -> "Slope":
        if type(other_speed) == float:
            return Slope(self.percentage * other_speed)
        elif type(other_speed) == Slope:
            return Slope(self.percentage * other_speed.percentage)


    def __div__(self, other: "Slope") -> "Slope":
        if other.percentage == 0:
            raise ZeroDivisionError("Cannot divide by zero slope")
        return Slope(self.percentage / other.percentage)

    def __lt__(self, other: "Slope") -> bool:
        return self.percentage < other.percentage

    def __gt__(self, other: "Slope") -> bool:
        return self.percentage > other.percentage   
    


class SlopeCalculator():
    def __init__(self) -> None:
        self.d1 = 0.0
        self.d2 = 0.0
        self.h_fixed = 0.0
        self.last_slope = Slope(0.0)
        self.buffer = []
        self.max_buffer = 4
        self.min_x = 0.1
        self.max_y = 50.0
        self.count = 0
        # altitude filter coefficient
        self.kf = 0.0 # 0.97
        self.least_square = True  # if True, use least square to calculate the slope, otherwise use the last two points in the buffer



    def calculate(self, current_h, start_distance, end_distance) -> float:
        
        # current height fixed is the height that is used to calculate the slope
        # it is calculated as a weighted average of the current height and the fixed height

        # if the delta diference is too small, return the last slope
        if (end_distance - start_distance) < self.min_x:
            return self.last_slope
        
        if current_h is None:
            return self.last_slope

        self.count += 1

        if (self.count <= self.max_buffer * 2):
            self.h_fixed = current_h
            return self.last_slope
        

        # avoid big jumps in altitude
        current_h_fixed = ((self.h_fixed - current_h) * self.kf) + current_h
        # current_h_fixed = current_h
        
        delta_d = end_distance - start_distance
        delta_h = current_h_fixed - self.h_fixed

        self.h_fixed = current_h_fixed

        if (len(self.buffer) < self.max_buffer):
            self.buffer.append([delta_d, delta_h])
        else:
            # remove the first element from the buffer
            self.buffer.pop(0)
            # add the new element to the buffer
            self.buffer.append([delta_d, delta_h])

            # fix delta h (max_y): 
            for i in range(len(self.buffer) -1):
                if self.buffer[i+1][1] - self.buffer[i][1] > self.max_y:
                    self.buffer[i+1][1] = self.buffer[i][1] + self.max_y
                if self.buffer[i+1][1] - self.buffer[i][1] < -self.max_y:
                    self.buffer[i+1][1] = self.buffer[i][1] - self.max_y

            slope = self.last_slope

            if self.least_square:
                # calculate the slope using least square on the buffer
                sum_x = 0.0
                sum_y = 0.0
                sum_xy = 0.0
                sum_xx = 0.0
                n = len(self.buffer)
                for x, y in self.buffer:
                    sum_x += x
                    sum_y += y
                    sum_xy += x * y
                    sum_xx += x * x
                if n == 0:
                    return self.last_slope
                # calculate the slope using least square
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
            else:
                # calculate the slope using the last two points in the buffer
                if (self.buffer[-1][0] - self.buffer[-2][0]) != 0:
                    slope = (self.buffer[-1][1] - self.buffer[-2][1]) / (self.buffer[-1][0] - self.buffer[-2][0])
                else:
                    slope = self.last_slope.percentage

            self.last_slope = Slope(slope * 100.0)  # convert to percentage

        return self.last_slope
    
    def reset(self):
        self.h_fixed = 0.0
        self.last_slope = Slope(0.0)
        self.count = 0


class GarminSegment(Segment):
    def get_coordinate(self, time: datetime) -> Optional[GarminCoordinate]:
        return super().get_coordinate(time)

    def __init__(
        self, coordinates: List[GarminCoordinate], laps: List["GarminLap"] = [], hr_zone_high_boundary: List[int] = []
    ) -> None:
        super().__init__(coordinates)
        self.coordinates: List[GarminCoordinate] = self.coordinates
        self.laps = laps
        self.hr_zone_high_boundary = hr_zone_high_boundary

    def write_to_csv(self, file_path):
        with open(file_path, "w") as csvfile:
            writer = csv.writer(csvfile)

            for coordinate in self.coordinates:
                writer.writerow(
                    [
                        int(coordinate.timestamp.timestamp()),
                        coordinate.latitude,
                        coordinate.longitude,
                        coordinate.speed,
                    ]
                )

        csvfile.close()

    def get_iterator(self, iterator_step_length: timedelta):
        return GarminSegmentIterator(self, iterator_step_length)

    def get_subsegment(
        self, start_time: datetime, end_time: datetime, step_length: timedelta
    ) -> "GarminSegment":
        new_coordinates: List[GarminCoordinate] = self._get_coordinates(
            start_time, end_time, step_length
        )
        return GarminSegment(new_coordinates, [], self.hr_zone_high_boundary)
    
    def get_subsegment_filled(
        self, garmin_start_time: datetime, fill_start_time: datetime, fill_end_time: datetime, start_time: datetime, end_time: datetime, step_length: timedelta
    ) -> "GarminSegment":
        fill_coordinates: List[GarminCoordinate] = self._extend_first_coordinates(
            garmin_start_time, fill_start_time, fill_end_time, step_length
        )

        new_coordinates: List[GarminCoordinate] = self._get_coordinates(
            start_time, end_time, step_length
        )

        return GarminSegment(fill_coordinates + new_coordinates, [], self.hr_zone_high_boundary)

    def get_first_lap(
        self, start_time: datetime, end_time: datetime
    ) -> Optional["GarminLap"]:
        for lap in self.laps:
            if (
                (lap.lap_trigger == "manual" or lap.lap_trigger == "session_end")
                and start_time < lap.start_time < end_time
            ):
                return lap

        return None

    def get_manual_laps(self) -> List["GarminLap"]:
        return [lap for lap in self.laps 
                if lap.lap_trigger == "manual" or lap.lap_trigger == "session_end"]

    @staticmethod
    def slope_percentage_calculator(slope_calculator: SlopeCalculator, coordinates: List[GarminCoordinate], end_coordinate: GarminCoordinate) -> float:
        if len(coordinates) > 0:
            return slope_calculator.calculate(
                end_coordinate.enhanced_altitude or end_coordinate.altitude,
                coordinates[-1].distance,
                end_coordinate.distance
            )
        else:
            return slope_calculator.last_slope

    @staticmethod
    def load_from_fit_file(path: str) -> "GarminSegment":
        stream = Stream.from_file(path)

        decoder = Decoder(stream)
        messages, _ = decoder.read()

        # save messages to json
        with open(path + ".json", "w") as json_file:
            json.dump(messages, json_file, indent=4, default=str)
        json_file.close()

        slope_calculator = SlopeCalculator()
        coordinates = []
        print("timestamp1,distance1,timestamp2,distance2,h_fixed,slope")
        for message in messages["record_mesgs"]:
            message = {key: message[key] for key in message if type(key) == str}
            message = {
                **message,
                "speed": Speed(meters_per_second=message.get("speed", None)),
                "enhanced_speed": Speed(meters_per_second=message.get("enhanced_speed", None))
            }
            if (message.get("temperature", None) is None):
                message = {
                **message,
                "temperature": 0,
                "slope": Slope(0.0)
            }

            coordinate = GarminCoordinate(**message)

            slope = GarminSegment.slope_percentage_calculator(slope_calculator, coordinates, coordinate)
            
            if (len(coordinates) > 0):
                print(f"{coordinates[-1].timestamp},{coordinates[-1].distance},{coordinate.timestamp},{coordinate.distance},{slope_calculator.h_fixed},{slope.percentage}")
            else:
                print(f",,{coordinate.timestamp},{coordinate.distance},{slope_calculator.h_fixed},{slope.percentage}")

            coordinate.set_slope(slope)
            coordinates.append(coordinate)

        slope_calculator.reset()

        laps = []
        for message in messages["lap_mesgs"]:
            message = {key: message[key] for key in message if type(key) == str}
            laps.append(GarminLap(**message))

        hr_zone_high_boundary = []
        if len(messages["time_in_zone_mesgs"]) > 0:
            message = messages["time_in_zone_mesgs"][0]
            if len(message["hr_zone_high_boundary"]) > 0:
                for h_boundary in message["hr_zone_high_boundary"]:
                    hr_zone_high_boundary.append(h_boundary)

        return GarminSegment(coordinates, laps=laps, hr_zone_high_boundary=hr_zone_high_boundary)


class SegmentIterator:
    def __init__(self, segment: Segment, iterator_step_length: timedelta) -> None:
        self.segment = segment
        self.iterator_step_length = iterator_step_length
        self.iterator_time = self.segment.get_start_time()

    def __iter__(self):
        self.iterator_time = self.segment.get_start_time()
        return self

    def __next__(self) -> Coordinate:
        coordinate = self.segment.get_coordinate(self.iterator_time)
        self.iterator_time += self.iterator_step_length

        if coordinate is None:
            raise StopIteration

        return coordinate


class GarminSegmentIterator(SegmentIterator):
    def __init__(self, segment: GarminSegment, iterator_step_length: timedelta):
        super().__init__(segment, iterator_step_length)

    def __iter__(self) -> "GarminSegmentIterator":
        return super().__iter__()

    def __next__(self) -> GarminCoordinate:
        return super().__next__()


class GarminLap:
    def __init__(self, start_time: datetime, lap_trigger: str, **_kwargs):
        self.start_time = start_time
        self.lap_trigger = lap_trigger
        self.total_elapsed_time = timedelta(seconds=_kwargs.get("total_elapsed_time", 0))


class Speed:
    METERS_IN_MILE = 1609.34
    SECONDS_IN_HOUR = 60 * 60

    def __init__(
        self,
        miles_per_hour: Optional[float] = None,
        meters_per_second: Optional[float] = None,
    ):
        if miles_per_hour is None and meters_per_second is None:
            miles_per_hour = 0.0
            meters_per_second = 0.0

        self.miles_per_hour = miles_per_hour
        self.meters_per_second = meters_per_second

    def get_miles_per_hour(self):
        if self.miles_per_hour is not None:
            return self.miles_per_hour
        elif self.meters_per_second is not None:
            return self.meters_per_second * (self.SECONDS_IN_HOUR / self.METERS_IN_MILE)

    def get_meters_per_second(self):
        if self.miles_per_hour is not None:
            return self.miles_per_hour / (self.SECONDS_IN_HOUR / self.METERS_IN_MILE)
        elif self.meters_per_second is not None:
            return self.meters_per_second

    def get_kilometers_per_hour(self):
        if self.miles_per_hour is not None:
            return (self.miles_per_hour / (self.SECONDS_IN_HOUR / self.METERS_IN_MILE)) * (self.SECONDS_IN_HOUR / 1000)
        elif self.meters_per_second is not None:
            return self.meters_per_second * (self.SECONDS_IN_HOUR / 1000)

    def __add__(self, other_speed: "Speed"):
        meters_per_second = (
            self.get_meters_per_second() + other_speed.get_meters_per_second()
        )
        return Speed(meters_per_second=meters_per_second)

    def __sub__(self, other_speed: "Speed"):
        meters_per_second = (
            self.get_meters_per_second() - other_speed.get_meters_per_second()
        )
        return Speed(meters_per_second=meters_per_second)

    def __div__(self, other_speed: "Speed"):
        meters_per_second = (
            self.get_meters_per_second() / other_speed.get_meters_per_second()
        )
        return Speed(meters_per_second=meters_per_second)

    def __mul__(self, other_speed: Any):
        if type(other_speed) == float:
            meters_per_second = self.get_meters_per_second() * other_speed
        elif type(other_speed) == Speed:
            meters_per_second = (
                self.get_meters_per_second() * other_speed.get_meters_per_second()
            )
        return Speed(meters_per_second=meters_per_second)

    def __lt__(self, other_speed: "Speed"):
        return self.get_meters_per_second() < other_speed.get_meters_per_second()

    def __gt__(self, other_speed: "Speed"):
        return self.get_meters_per_second() > other_speed.get_meters_per_second()
