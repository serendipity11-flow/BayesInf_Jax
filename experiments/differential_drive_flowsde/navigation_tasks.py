"""Shared differential-drive navigation task definitions."""
import math


COMPLEX_TASKS = [
    ((-3.5, -2.5, .2), [(-2.0, 1.8), (0.0, -1.8), (2.0, 1.8), (3.5, -2.5)]),
    ((-3.8, 0.0, 0.0), [(-2.2, 2.3), (-.5, -2.2), (1.4, 2.2), (3.7, -.5)]),
    ((-3.0, -2.2, 0.0), [(3.0, -2.2), (3.0, 2.2), (-3.0, 2.2), (-3.0, -1.5)]),
]


def densify_tasks(tasks, max_spacing=.6):
    """Insert linear subgoals while preserving every original route vertex."""
    dense = []
    for start, waypoints in tasks:
        previous = start[:2]
        route = []
        for target in waypoints:
            distance = math.hypot(target[0] - previous[0], target[1] - previous[1])
            segments = max(1, math.ceil(distance / max_spacing))
            route.extend([
                (previous[0] + i / segments * (target[0] - previous[0]),
                 previous[1] + i / segments * (target[1] - previous[1]))
                for i in range(1, segments + 1)
            ])
            previous = target
        dense.append((start, route))
    return dense


DENSE_COMPLEX_TASKS = densify_tasks(COMPLEX_TASKS)


def path_lookahead_target(position, waypoints, waypoint_index, lookahead):
    """Return a point ``lookahead`` metres ahead on the remaining polyline."""
    start = (float(position[0]), float(position[1]))
    remaining = float(lookahead)
    for target in waypoints[waypoint_index:]:
        segment = math.dist(start, target)
        if segment >= remaining and segment > 0:
            fraction = remaining / segment
            return (start[0] + fraction * (target[0] - start[0]),
                    start[1] + fraction * (target[1] - start[1]))
        remaining -= segment
        start = target
    # Continue the virtual reference beyond the terminal checkpoint. This lets a
    # forward-only robot drive through the real endpoint instead of orbiting it
    # when the finite path makes the lookahead distance collapse to zero.
    if len(waypoints) > 1 and remaining > 0:
        previous, terminal = waypoints[-2], waypoints[-1]
        length = math.dist(previous, terminal)
        if length > 0:
            return (terminal[0] + remaining * (terminal[0] - previous[0]) / length,
                    terminal[1] + remaining * (terminal[1] - previous[1]) / length)
    return tuple(waypoints[-1])


def advance_path_index(position, waypoints, waypoint_index, tolerance, window=6):
    """Advance monotonically to the nearest local route checkpoint."""
    stop = min(len(waypoints), waypoint_index + window)
    distances = [math.dist(position, point) for point in waypoints[waypoint_index:stop]]
    nearest = waypoint_index + min(range(len(distances)), key=distances.__getitem__)
    if nearest > waypoint_index:
        return nearest
    return waypoint_index + 1 if distances[0] <= tolerance else waypoint_index
