#!/usr/bin/env python3

import argparse
import json
import math
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterable, List, Optional


COLLISION_THRESHOLD_SPECS = {
    "lower_torque_thresholds_nominal": (7, [40.0, 40.0, 35.0, 35.0, 30.0, 25.0, 20.0]),
    "upper_torque_thresholds_nominal": (7, [55.0, 55.0, 50.0, 50.0, 45.0, 40.0, 35.0]),
    "lower_torque_thresholds_acceleration": (7, [40.0, 40.0, 35.0, 35.0, 30.0, 25.0, 20.0]),
    "upper_torque_thresholds_acceleration": (7, [55.0, 55.0, 50.0, 50.0, 45.0, 40.0, 35.0]),
    "lower_force_thresholds_nominal": (6, [45.0, 45.0, 45.0, 35.0, 35.0, 35.0]),
    "upper_force_thresholds_nominal": (6, [60.0, 60.0, 60.0, 50.0, 50.0, 50.0]),
    "lower_force_thresholds_acceleration": (6, [45.0, 45.0, 45.0, 35.0, 35.0, 35.0]),
    "upper_force_thresholds_acceleration": (6, [60.0, 60.0, 60.0, 50.0, 50.0, 50.0]),
}

GRIPPER_MOVE_SPEED = 0.05
GRIPPER_GRASP_SPEED = 0.03
GRIPPER_GRASP_FORCE = 40.0
GRIPPER_EPSILON_INNER = 0.005
GRIPPER_EPSILON_OUTER = 0.005
GRIPPER_OPEN_WIDTH = 0.08
GRIPPER_CLOSED_WIDTH = 0.0


def _values(value: Any) -> List[float]:
    if value is None:
        return []
    if callable(value):
        value = value()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        result: List[float] = []
        for item in value:
            if isinstance(item, Iterable) and not isinstance(item, (str, bytes, dict)):
                result.extend(_values(item))
            else:
                result.append(float(item))
        return result
    return []


def _attr(obj: Any, *names: str) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            return value() if callable(value) else value
    return None


def _quat_from_matrix(matrix: List[float]) -> List[float]:
    m00, m01, m02 = matrix[0], matrix[1], matrix[2]
    m10, m11, m12 = matrix[4], matrix[5], matrix[6]
    m20, m21, m22 = matrix[8], matrix[9], matrix[10]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return [(m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale, 0.25 * scale]
    if m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return [0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale, (m21 - m12) / scale]
    if m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return [(m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale, (m02 - m20) / scale]
    scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return [(m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale, (m10 - m01) / scale]


def _affine_to_pose(affine: Any) -> List[float]:
    translation = _values(_attr(affine, "translation", "position", "translation_vector"))
    quaternion = _values(_attr(affine, "quaternion", "orientation", "rotation_quaternion"))
    if len(translation) >= 3 and len(quaternion) >= 4:
        return translation[:3] + quaternion[:4]

    raw = _values(affine)
    if len(raw) == 7:
        return raw
    if len(raw) == 16:
        return [raw[12], raw[13], raw[14]] + _quat_from_matrix(raw)
    raise RuntimeError(f"Could not convert franky Affine to pose; available attributes: {dir(affine)}")


def _reshape_jacobian(values: List[float]) -> List[List[float]]:
    if len(values) != 42:
        return [[0.0] * 7 for _ in range(6)]
    return [[float(values[row + 6 * col]) for col in range(7)] for row in range(6)]


def _valid_jacobian(values: List[float]) -> bool:
    return len(values) == 42 and any(abs(value) > 1e-12 for value in values)


def _column_major_matrix(values: List[float]) -> Any:
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel

        return np.array(values, dtype=float).reshape((4, 4), order="F")
    except ImportError:
        return [[values[row + 4 * col] for col in range(4)] for row in range(4)]


def _coerce_gripper_width(value: Any) -> float:
    width = float(value)
    if width < 0.0:
        raise ValueError("gripper width must be >= 0")
    return width


def _coerce_positive_float(value: Any, name: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return result


def _coerce_nonnegative_float(value: Any, name: str) -> float:
    result = float(value)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return result


class FrankyHTTPServer:
    def __init__(self, robot_ip: str, host: str, port: int, relative_dynamics_factor: float) -> None:
        try:
            import franky  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise RuntimeError(
                "Python package 'franky' is not installed. Install it with `pip install franky-control` "
                "or build TimSchneider42/franky from source."
            ) from exc

        self.franky = franky
        self.robot_ip = robot_ip
        self.host = host
        self.port = port
        self.lock = threading.RLock()
        self.robot = franky.Robot(robot_ip)
        self.gripper = None
        self.relative_dynamics_factor = float(relative_dynamics_factor)
        self.robot.relative_dynamics_factor = self.relative_dynamics_factor
        self.collision_thresholds = {
            name: list(default_values) for name, (_, default_values) in COLLISION_THRESHOLD_SPECS.items()
        }
        self._apply_collision_thresholds()

    def run(self) -> None:
        server = self

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                server.handle_request(self)

            def do_POST(self):  # noqa: N802
                server.handle_request(self)

            def log_message(self, fmt: str, *args) -> None:
                print(f"HTTP {self.command} - {fmt % args}", flush=True)

        http_server = ThreadingHTTPServer((self.host, self.port), RequestHandler)
        print(f"franky HTTP server listening on http://{self.host}:{self.port}", flush=True)
        http_server.serve_forever()

    def handle_request(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            payload = self._read_json(handler)
            response = self._dispatch(handler.path, payload)
            self._send_json(handler, HTTPStatus.OK, response)
        except ValueError as exc:
            print(f"HTTP {handler.path} bad request: {exc}", flush=True)
            self._send_json(handler, HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
        except Exception as exc:  # pylint: disable=broad-except
            print(f"HTTP {handler.path} failed: {exc}", flush=True)
            self._send_json(handler, HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})

    @staticmethod
    def _read_json(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
        length = int(handler.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        body = handler.rfile.read(length)
        return json.loads(body.decode("utf-8")) if body else {}

    @staticmethod
    def _send_json(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        handler.send_response(status.value)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    def _dispatch(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        routes = {
            "/health": self.health,
            "/startimp": self.noop,
            "/stopimp": self.noop,
            "/clearerr": self.clear_error,
            "/getstate": self.get_state,
            "/getpos": self.get_pose,
            "/pose": lambda: self.move_pose(payload),
            "/get_gripper": self.get_gripper,
            "/activate_gripper": self.activate_gripper,
            "/reset_gripper": self.reset_gripper,
            "/open_gripper": lambda: self.open_gripper(payload),
            "/close_gripper": lambda: self.close_gripper(payload),
            "/move_gripper": lambda: self.move_gripper(payload),
            "/get_collision": self.get_collision,
            "/set_collision": lambda: self.set_collision(payload),
        }
        if path not in routes:
            raise ValueError(f"Unknown endpoint: {path}")
        return routes[path]()

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "robot_ip": self.robot_ip,
            "backend": "franky",
            "relative_dynamics_factor": self.relative_dynamics_factor,
        }

    def noop(self) -> Dict[str, Any]:
        return {"ok": True, "message": "No-op in franky direct-control mode"}

    def clear_error(self) -> Dict[str, Any]:
        with self.lock:
            self.robot.recover_from_errors()
        return {"ok": True, "message": "Clear"}

    def get_pose(self) -> Dict[str, Any]:
        return {"ok": True, "pose": self.get_state()["pose"]}

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            cartesian_state = self.robot.current_cartesian_state
            joint_state = self.robot.current_joint_state
            robot_state = self.robot.state
            pose = _affine_to_pose(cartesian_state.pose.end_effector_pose)
            velocity = _values(cartesian_state.velocity.end_effector_twist)[:6]
            q = _values(joint_state.position)[:7]
            dq = _values(joint_state.velocity)[:7]
            jacobian_values = self._jacobian(robot_state)
            force_torque = self._force_torque(robot_state)
            gripper_state = self._gripper_state_or_none()
        response = {
            "ok": True,
            "pose": pose,
            "vel": velocity if len(velocity) == 6 else [0.0] * 6,
            "force": force_torque[:3],
            "torque": force_torque[3:],
            "q": q if len(q) == 7 else [0.0] * 7,
            "dq": dq if len(dq) == 7 else [0.0] * 7,
        }
        if jacobian_values is not None:
            response["jacobian"] = _reshape_jacobian(jacobian_values)
        if gripper_state is not None:
            response.update(gripper_state)
        return response

    def move_pose(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        pose = payload.get("arr")
        if not isinstance(pose, list) or len(pose) not in (3, 7):
            raise ValueError("arr must contain [x, y, z] or [x, y, z, qx, qy, qz, qw]")
        position = [float(value) for value in pose[:3]]
        quaternion = [float(value) for value in pose[3:]] if len(pose) == 7 else []
        relative_dynamics_factor = float(payload.get("relative_dynamics_factor", self.relative_dynamics_factor))
        asynchronous = bool(payload.get("asynchronous", False))

        with self.lock:
            self.robot.relative_dynamics_factor = relative_dynamics_factor
            self.relative_dynamics_factor = relative_dynamics_factor
            if not quaternion:
                quaternion = self.get_state()["pose"][3:7]
            quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
            if quaternion_norm < 1e-9:
                raise ValueError("Quaternion norm is zero")
            quaternion = [value / quaternion_norm for value in quaternion]
            affine = self.franky.Affine(position, quaternion)
            motion = self.franky.CartesianMotion(
                affine,
                self.franky.ReferenceType.Absolute,
                relative_dynamics_factor,
                True,
            )
            try:
                self.robot.move(motion, asynchronous=asynchronous)
            except TypeError:
                self.robot.move(motion)
        return {
            "ok": True,
            "message": "Moved" if not asynchronous else "Motion started",
            "target_pose": position + quaternion,
            "relative_dynamics_factor": relative_dynamics_factor,
            "asynchronous": asynchronous,
        }

    def get_collision(self) -> Dict[str, Any]:
        return {"ok": True, "collision_thresholds": self._collision_thresholds_response()}

    def get_gripper(self) -> Dict[str, Any]:
        with self.lock:
            state = self._gripper_state_or_none(create=True)
        if state is None:
            return {"ok": True, "gripper_pos": None, "have_gripper": False}
        return {"ok": True, **state}

    def activate_gripper(self) -> Dict[str, Any]:
        with self.lock:
            success = bool(self._get_gripper().homing())
            state = self._gripper_state_or_none()
        return {
            "ok": success,
            "message": "Gripper activated" if success else "Failed to activate gripper",
            **(state or {}),
        }

    def reset_gripper(self) -> Dict[str, Any]:
        with self.lock:
            try:
                self._get_gripper().stop()
            except Exception:  # pylint: disable=broad-except
                pass
            success = bool(self._get_gripper().homing())
            state = self._gripper_state_or_none()
        return {
            "ok": success,
            "message": "Gripper reset" if success else "Failed to reset gripper",
            **(state or {}),
        }

    def open_gripper(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        width = _coerce_gripper_width(payload.get("width", GRIPPER_OPEN_WIDTH))
        speed = _coerce_positive_float(payload.get("speed", GRIPPER_MOVE_SPEED), "gripper speed")
        with self.lock:
            gripper = self._get_gripper()
            if hasattr(gripper, "open") and abs(width - GRIPPER_OPEN_WIDTH) < 1e-9:
                success = bool(gripper.open(speed))
            else:
                success = bool(gripper.move(width, speed))
            state = self._gripper_state_or_none()
        return {
            "ok": success,
            "message": "Gripper opened" if success else "Failed to open gripper",
            **(state or {}),
        }

    def close_gripper(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        width = _coerce_gripper_width(payload.get("width", GRIPPER_CLOSED_WIDTH))
        speed = _coerce_positive_float(payload.get("speed", GRIPPER_GRASP_SPEED), "gripper speed")
        force = _coerce_positive_float(payload.get("force", GRIPPER_GRASP_FORCE), "gripper force")
        epsilon_inner = _coerce_nonnegative_float(
            payload.get("epsilon_inner", GRIPPER_EPSILON_INNER), "epsilon_inner"
        )
        epsilon_outer = _coerce_nonnegative_float(
            payload.get("epsilon_outer", GRIPPER_EPSILON_OUTER), "epsilon_outer"
        )
        with self.lock:
            success = bool(self._get_gripper().grasp(width, speed, force, epsilon_inner, epsilon_outer))
            state = self._gripper_state_or_none()
        return {
            "ok": success,
            "message": "Gripper closed" if success else "Failed to close gripper",
            **(state or {}),
        }

    def move_gripper(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        width = payload.get("width")
        if width is None and "arr" in payload:
            arr = payload.get("arr")
            if not isinstance(arr, list) or not arr:
                raise ValueError("move_gripper arr must be a non-empty list")
            width = arr[0]
        width = _coerce_gripper_width(width if width is not None else GRIPPER_OPEN_WIDTH)
        speed = _coerce_positive_float(payload.get("speed", GRIPPER_MOVE_SPEED), "gripper speed")
        with self.lock:
            success = bool(self._get_gripper().move(width, speed))
            state = self._gripper_state_or_none()
        return {
            "ok": success,
            "message": "Gripper moved" if success else "Failed to move gripper",
            **(state or {}),
        }

    def set_collision(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict) or not payload:
            raise ValueError("set_collision expects a JSON object with collision threshold arrays")
        unknown = sorted(set(payload.keys()) - set(COLLISION_THRESHOLD_SPECS.keys()))
        if unknown:
            raise ValueError(f"Unknown collision threshold fields: {unknown}")
        updates: Dict[str, List[float]] = {}
        for name, values in payload.items():
            expected_length, _ = COLLISION_THRESHOLD_SPECS[name]
            if not isinstance(values, list) or len(values) != expected_length:
                raise ValueError(f"{name} must contain {expected_length} numeric values")
            updates[name] = [float(value) for value in values]
        with self.lock:
            self.collision_thresholds.update(updates)
            self._apply_collision_thresholds()
        return {
            "ok": True,
            "message": "Collision thresholds updated",
            "updated": list(updates.keys()),
            "collision_thresholds": self._collision_thresholds_response(),
        }

    def _apply_collision_thresholds(self) -> None:
        lower_torque_acc = self.collision_thresholds["lower_torque_thresholds_acceleration"]
        upper_torque_acc = self.collision_thresholds["upper_torque_thresholds_acceleration"]
        lower_torque_nom = self.collision_thresholds["lower_torque_thresholds_nominal"]
        upper_torque_nom = self.collision_thresholds["upper_torque_thresholds_nominal"]
        lower_force_acc = self.collision_thresholds["lower_force_thresholds_acceleration"]
        upper_force_acc = self.collision_thresholds["upper_force_thresholds_acceleration"]
        lower_force_nom = self.collision_thresholds["lower_force_thresholds_nominal"]
        upper_force_nom = self.collision_thresholds["upper_force_thresholds_nominal"]
        for name in ("set_collision_behavior", "setCollisionBehavior"):
            method = getattr(self.robot, name, None)
            if method is not None:
                method(
                    lower_torque_acc,
                    upper_torque_acc,
                    lower_torque_nom,
                    upper_torque_nom,
                    lower_force_acc,
                    upper_force_acc,
                    lower_force_nom,
                    upper_force_nom,
                )
                return

    def _collision_thresholds_response(self) -> Dict[str, List[float]]:
        return {name: list(values) for name, values in self.collision_thresholds.items()}

    def _get_gripper(self) -> Any:
        if self.gripper is None:
            self.gripper = self.franky.Gripper(self.robot_ip)
        return self.gripper

    def _gripper_state_or_none(self, create: bool = False) -> Optional[Dict[str, Any]]:
        if self.gripper is None and not create:
            return None
        try:
            state = self._get_gripper().state
            width = _values(_attr(state, "width"))
            max_width = _values(_attr(state, "max_width"))
            is_grasped = _attr(state, "is_grasped")
            response: Dict[str, Any] = {
                "gripper_pos": width[0] if width else None,
                "have_gripper": True,
            }
            if max_width:
                response["gripper_max_width"] = max_width[0]
            if is_grasped is not None:
                response["gripper_is_grasped"] = bool(is_grasped)
            return response
        except Exception:  # pylint: disable=broad-except
            return None

    def _jacobian(self, robot_state: Any) -> Optional[List[float]]:
        model = self.robot.model
        frame = getattr(self.franky.Frame, "EndEffector", None)
        for attr_name in ("zero_jacobian", "body_jacobian", "jacobian", "J"):
            values = _values(_attr(robot_state, attr_name))
            if _valid_jacobian(values):
                return values
        for name in ("zero_jacobian", "body_jacobian"):
            method = getattr(model, name, None)
            if method is None:
                continue
            try:
                values = _values(method(frame, robot_state))
                if _valid_jacobian(values):
                    return values
            except Exception:  # pylint: disable=broad-except
                pass
            try:
                q = _values(_attr(robot_state, "q"))
                f_t_ee = _values(_attr(robot_state, "F_T_EE"))
                ee_t_k = _values(_attr(robot_state, "EE_T_K"))
                if len(q) == 7 and len(f_t_ee) == 16 and len(ee_t_k) == 16:
                    f_t_ee_affine = self.franky.Affine(_column_major_matrix(f_t_ee))
                    ee_t_k_affine = self.franky.Affine(_column_major_matrix(ee_t_k))
                    values = _values(method(frame, q, f_t_ee_affine, ee_t_k_affine))
                    if _valid_jacobian(values):
                        return values
            except Exception:  # pylint: disable=broad-except
                continue
        return None

    @staticmethod
    def _force_torque(robot_state: Any) -> List[float]:
        for name in ("k_f_ext_hat_k", "K_F_ext_hat_K", "o_f_ext_hat_k", "O_F_ext_hat_K"):
            values = _values(_attr(robot_state, name))
            if len(values) >= 6:
                return values[:6]
        return [0.0] * 6


def main() -> None:
    parser = argparse.ArgumentParser(description="HTTP server for franky/Ruckig Franka control")
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--relative-dynamics-factor", type=float, default=0.2)
    args, _ = parser.parse_known_args()

    server = FrankyHTTPServer(
        robot_ip=args.robot_ip,
        host=args.host,
        port=args.port,
        relative_dynamics_factor=args.relative_dynamics_factor,
    )
    server.run()


if __name__ == "__main__":
    main()
