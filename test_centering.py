import pytest

from software.centering import (
    Detection,
    compute_velocity,
    is_centered,
    pick_target,
    pixel_error,
)

W, H = 640, 640


def det(cls="body tube", conf=0.9, box=(300, 300, 340, 340)):
    return Detection(cls, conf, *box)


def test_pick_target_chooses_highest_confidence():
    dets = [det(conf=0.5), det(conf=0.9), det("nosecone", conf=0.99)]
    assert pick_target(dets).conf == 0.9


def test_pick_target_ignores_low_confidence_and_other_classes():
    assert pick_target([det(conf=0.2), det("fin can")], min_conf=0.4) is None


def test_pick_target_class_name_formatting_is_forgiving():
    assert pick_target([det("body_tube")], target_class="Body Tube") is not None


def test_pick_target_empty():
    assert pick_target([]) is None


def test_pixel_error_centered_is_zero():
    ex, ey = pixel_error(det(box=(300, 300, 340, 340)), W, H)
    assert ex == pytest.approx(0.0)
    assert ey == pytest.approx(0.0)


def test_pixel_error_extremes():
    # box centered on the right edge, vertically centered
    ex, ey = pixel_error(Detection("body tube", 0.9, 630, 310, 650, 330), W, H)
    assert ex == pytest.approx(1.0, abs=0.05)
    assert ey == pytest.approx(0.0)


def test_is_centered():
    assert is_centered(0.02, -0.03)
    assert not is_centered(0.2, 0.0)


def test_velocity_directions():
    # target above center (ey < 0) -> forward; right of center (ex > 0) -> right
    vx, vy = compute_velocity(ex=0.5, ey=-0.5)
    assert vx > 0 and vy > 0
    vx, vy = compute_velocity(ex=-0.5, ey=0.5)
    assert vx < 0 and vy < 0


def test_velocity_deadband_gives_zero():
    assert compute_velocity(0.01, -0.02) == (0.0, 0.0)


def test_velocity_is_clamped():
    vx, vy = compute_velocity(ex=1.0, ey=-1.0, kp=10.0, max_speed=0.5)
    assert vx == 0.5 and vy == 0.5
