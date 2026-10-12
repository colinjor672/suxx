import math
import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

import network.mqtt_navigation as mqtt_navigation_module
from network.mqtt_navigation import (
    DEFAULT_GNSS_TOPIC,
    DEFAULT_PREDICTION_TOPIC,
    MqttNavigationClient,
    MqttNavigationConfig,
    _normalise_prediction,
)
from utils.realtime_navigation import (
    EARTH_RADIUS_M,
    EnuMissionRoute,
    RealtimeNavigationState,
)


BASE_TS = 1_800_000_000.0
BASE_LAT = math.radians(22.75)
BASE_LON = math.radians(113.59)


def mission_point(north_m: float, east_m: float = 0.0) -> dict:
    return {
        "latitude": BASE_LAT + north_m / EARTH_RADIUS_M,
        "longitude": BASE_LON
        + east_m / (EARTH_RADIUS_M * math.cos(BASE_LAT)),
    }


def mission(points: list[dict]) -> dict:
    return {"count": 1, "tolerance": 5.0, "points": points}


def add_linear_gnss(
    state: RealtimeNavigationState,
    *,
    count: int,
    speed_mps: float,
    start_index: int = 0,
) -> None:
    for index in range(start_index, start_index + count):
        elapsed = index * 0.1
        state.add_gnss(
            BASE_TS + elapsed,
            BASE_LAT + speed_mps * elapsed / EARTH_RADIUS_M,
            BASE_LON,
            received_at=BASE_TS + elapsed,
        )


class EnuMissionRouteTests(unittest.TestCase):
    def test_full_route_is_enu_relative_to_command_position(self):
        route = EnuMissionRoute.from_mission(
            mission([mission_point(10.0), mission_point(40.0, 20.0)]),
            BASE_LAT,
            BASE_LON,
        )
        self.assertAlmostEqual(route.mission_north_m[0], 10.0, places=4)
        self.assertAlmostEqual(route.mission_east_m[1], 20.0, places=4)
        self.assertGreater(route.length_m, 40.0)

    def test_last_window_is_extended_to_exactly_sixty_metres(self):
        route = EnuMissionRoute.from_mission(
            mission([mission_point(50.0), mission_point(100.0)]),
            BASE_LAT,
            BASE_LON,
        )
        body = route.body_window(route.length_m - 10.0, 60.0, 0.25)
        length = float(np.sum(np.hypot(np.diff(body[:, 0]), np.diff(body[:, 1]))))
        self.assertEqual(len(body), 241)
        self.assertAlmostEqual(length, 60.0, places=3)
        np.testing.assert_allclose(body[0], [0.0, 0.0])


class RealtimeNavigationStateTests(unittest.TestCase):
    def make_state(self) -> RealtimeNavigationState:
        return RealtimeNavigationState(
            lookahead_m=60.0,
            sample_spacing_m=0.25,
            speed_window_points=15,
            speed_update_hz=10.0,
            moving_threshold_mps=1.0,
            timestamp_tolerance_s=1.0,
        )

    def test_route_progress_uses_fifteen_point_speed_window(self):
        state = self.make_state()
        add_linear_gnss(state, count=15, speed_mps=2.0)
        trigger = BASE_TS + 1.4
        state.activate_mission(
            mission([mission_point(20.0), mission_point(100.0)]),
            trigger,
            now_epoch=trigger,
            now_monotonic=10.0,
        )
        initial = state.snapshot()
        self.assertTrue(initial.moving)
        self.assertAlmostEqual(initial.speed_mps, 2.0, places=3)
        self.assertAlmostEqual(initial.time_to_60m_s, 30.0, places=3)

        state.update(now_epoch=trigger + 0.5, now_monotonic=10.5)
        self.assertAlmostEqual(state.snapshot().progress_m, 1.0, places=3)

    def test_speed_is_recomputed_only_after_one_hundred_ms(self):
        state = self.make_state()
        add_linear_gnss(state, count=15, speed_mps=2.0)
        trigger = BASE_TS + 1.4
        state.activate_mission(
            mission([mission_point(100.0)]),
            trigger,
            now_epoch=trigger,
            now_monotonic=10.0,
        )
        for index in range(15, 30):
            state.add_gnss(
                BASE_TS + index * 0.1,
                BASE_LAT + 2.8 / EARTH_RADIUS_M,
                BASE_LON,
                received_at=BASE_TS + index * 0.1,
            )

        state.update(now_epoch=BASE_TS + 2.9, now_monotonic=10.05)
        self.assertGreaterEqual(state.snapshot().speed_mps, 1.0)
        state.update(now_epoch=BASE_TS + 2.9, now_monotonic=10.1)
        self.assertFalse(state.snapshot().moving)

    def test_below_one_metre_per_second_pauses_route_and_draws_straight(self):
        state = self.make_state()
        add_linear_gnss(state, count=15, speed_mps=0.5)
        trigger = BASE_TS + 1.4
        state.activate_mission(
            mission([mission_point(20.0, 10.0), mission_point(100.0, 40.0)]),
            trigger,
            now_epoch=trigger,
            now_monotonic=10.0,
        )
        body = state.body_path(now_epoch=trigger, now_monotonic=10.0)
        snapshot = state.snapshot()
        self.assertTrue(snapshot.route_active)
        self.assertFalse(snapshot.moving)
        self.assertEqual(snapshot.progress_m, 0.0)
        np.testing.assert_allclose(body[:, 1], 0.0)
        self.assertAlmostEqual(float(body[-1, 0]), 60.0, places=5)

    def test_command_timestamp_requires_gnss_within_one_second(self):
        state = self.make_state()
        add_linear_gnss(state, count=15, speed_mps=2.0)
        with self.assertRaisesRegex(ValueError, "timestamp skew"):
            state.activate_mission(
                mission([mission_point(100.0)]),
                BASE_TS + 2.41,
                now_epoch=BASE_TS + 2.41,
                now_monotonic=10.0,
            )


class MqttNavigationClientTests(unittest.TestCase):
    def setUp(self):
        self.original_mqtt = mqtt_navigation_module.mqtt

    def tearDown(self):
        mqtt_navigation_module.mqtt = self.original_mqtt

    def test_default_topics_are_the_live_navigation_topics(self):
        config = MqttNavigationConfig()
        self.assertEqual(config.prediction_topic, "v1/11/prediction/result")
        self.assertEqual(config.gnss_topic, "v1/11/sensor/gnss/gnss_01")
        self.assertEqual(config.prediction_topic, DEFAULT_PREDICTION_TOPIC)
        self.assertEqual(config.gnss_topic, DEFAULT_GNSS_TOPIC)

    def test_prediction_shapes_are_normalised(self):
        expected = [mission_point(10.0), mission_point(20.0)]
        normalised = _normalise_prediction(
            {"result": {"path": [{"lat": p["latitude"], "lon": p["longitude"]} for p in expected]}}
        )
        self.assertEqual(normalised["count"], 1)
        self.assertEqual(normalised["tolerance"], 5.0)
        self.assertEqual(normalised["points"], expected)

    def test_start_creates_one_connection_and_subscribes_both_topics(self):
        class FakeClient:
            def __init__(self):
                self.connect_calls = []
                self.subscriptions = []

            def reconnect_delay_set(self, **_kwargs):
                pass

            def connect_async(self, *args, **kwargs):
                self.connect_calls.append((args, kwargs))

            def loop_start(self):
                pass

            def subscribe(self, topic, qos):
                self.subscriptions.append((topic, qos))

            def disconnect(self):
                pass

            def loop_stop(self):
                pass

        fake_client = FakeClient()

        class FakeMqtt:
            MQTTv5 = 5

            @staticmethod
            def Client(**_kwargs):
                return fake_client

        mqtt_navigation_module.mqtt = FakeMqtt()
        client = MqttNavigationClient(
            RealtimeNavigationState(),
            MqttNavigationConfig(host="broker.local"),
        )
        client.start()
        fake_client.on_connect(fake_client, None, None, 0)

        self.assertEqual(
            fake_client.connect_calls,
            [(('broker.local',), {"keepalive": 30})],
        )
        self.assertEqual(
            fake_client.subscriptions,
            [(DEFAULT_PREDICTION_TOPIC, 1), (DEFAULT_GNSS_TOPIC, 0)],
        )
        client.stop()

    def test_gnss_and_prediction_messages_activate_the_route(self):
        class FakeMqtt:
            MQTTv5 = 5

        mqtt_navigation_module.mqtt = FakeMqtt()
        state = RealtimeNavigationState(
            lookahead_m=30.0,
            sample_spacing_m=0.25,
            speed_window_points=15,
            speed_update_hz=10.0,
            moving_threshold_mps=1.0,
            timestamp_tolerance_s=1.0,
        )
        client = MqttNavigationClient(
            state,
            MqttNavigationConfig(),
            clock=lambda: BASE_TS + 1.4,
            monotonic_clock=lambda: 10.0,
        )
        for index in range(15):
            elapsed = index * 0.1
            client._handle_gnss(
                {
                    "ts": BASE_TS + elapsed,
                    "data": {
                        "lat": BASE_LAT + 2.0 * elapsed / EARTH_RADIUS_M,
                        "lon": BASE_LON,
                    },
                }
            )

        with redirect_stdout(StringIO()):
            client._handle_prediction(
                {
                    "ts": BASE_TS + 1.4,
                    "data": {
                        "path": [mission_point(20.0), mission_point(80.0)],
                    },
                }
            )
        snapshot = state.snapshot()
        self.assertTrue(snapshot.route_active)
        self.assertTrue(snapshot.moving)
        self.assertGreater(snapshot.route_length_m, 70.0)


if __name__ == "__main__":
    unittest.main()
