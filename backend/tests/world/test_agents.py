import math

import pytest

from parkpilot.core.config import AgentConfig
from parkpilot.core.geometry import Pose, rect_polygon
from parkpilot.world.agents import AgentSet, AgentSpec

CFG = AgentConfig()
FAR = Pose(100, 100, 0)


def test_pedestrian_walks_and_despawns() -> None:
    agents = AgentSet.from_specs(CFG, [AgentSpec("p", "pedestrian", ((0, 0), (2, 0)), 1.0)])
    a = agents.agents[0]
    t = 0.0
    for _ in range(10):
        agents.step(t, 0.1, FAR, [])
        t += 0.1
    assert math.isclose(a.pose.x, 1.0, abs_tol=1e-6)
    assert a.velocity[0] > 0
    for _ in range(20):
        agents.step(t, 0.1, FAR, [])
        t += 0.1
    assert not a.active and agents.active() == []


def test_start_time_and_trigger_distance() -> None:
    specs = [
        AgentSpec("late", "cart", ((0, 0), (5, 0)), 1.0, start_time=5.0),
        AgentSpec("trig", "pedestrian", ((10, 0), (10, 5)), 1.0, trigger_distance=3.0),
    ]
    agents = AgentSet.from_specs(CFG, specs)
    agents.step(1.0, 0.1, FAR, [])
    assert not agents.agents[0].started and not agents.agents[1].started
    assert (agents.agents[0].velocity == 0).all()
    agents.step(6.0, 0.1, Pose(9, 0, 0), [])
    assert agents.agents[0].started and agents.agents[1].started


def test_backing_car_has_reverse_lights_and_faces_away() -> None:
    spec = AgentSpec("c", "car", ((0, 0), (0, -5), (10, -5)), 1.0, reversing=True)
    agents = AgentSet.from_specs(CFG, [spec])
    car = agents.agents[0]
    assert math.isclose(car.pose.theta, math.pi / 2)  # nose points into the stall
    agents.step(0.0, 0.1, FAR, [])
    assert car.reverse_lights
    assert car.velocity[1] < 0  # moving backwards out of the stall
    for i in range(80):
        agents.step(0.1 * i, 0.1, FAR, [])
    assert not car.reverse_lights
    assert abs(car.pose.theta) < 1e-6  # driving forward along the aisle


def test_agent_yields_to_ego_in_front() -> None:
    agents = AgentSet.from_specs(CFG, [AgentSpec("p", "pedestrian", ((0, 0), (5, 0)), 1.0)])
    ego = [rect_polygon(Pose(1.0, 0, 0), 0.5, 2.0)]
    agents.step(0.0, 0.1, FAR, ego)
    assert agents.agents[0].waiting
    assert agents.agents[0].pose.x == 0.0


def test_loop_and_validation() -> None:
    agents = AgentSet.from_specs(CFG, [AgentSpec("l", "cart", ((0, 0), (0.5, 0)), 1.0, loop=True)])
    for i in range(20):
        agents.step(0.1 * i, 0.1, FAR, [])
    assert agents.agents[0].active
    assert agents.agents[0].polygon().shape == (4, 2)
    with pytest.raises(ValueError):
        AgentSet.from_specs(CFG, [AgentSpec("x", "cart", ((0, 0),), 1.0)])
