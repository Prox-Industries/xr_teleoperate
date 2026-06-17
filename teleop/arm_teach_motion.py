"""Teaching playback ヘルパー: 記録した関節軌道の順/逆再生(重力補償つき)。

teleop_hand_and_arm.py と teach_play.py から共用する(重複実装を避ける)。
- load_traj                : npz をロード。mirror=True で記録腕→反対腕へ左右対称コピー。
- make_grav                : reduced_robot.model から重力補償トルク関数(pin.rnea, v=a=0)。
- load_reduced_model_cache : IK と共有の g1_29_model_cache.pkl から reduced_model を読む。
- nearest_index            : 軌道中で現在姿勢に最も近いサンプル番号。
- play                     : 軌道を q_target でなぞって再生(順/逆、重力補償、速度制御)。
"""
import time
import pickle
import numpy as np
import pinocchio as pin
import logging_mp

logger_mp = logging_mp.getLogger(__name__)

# 1腕7関節 [pitch, roll, yaw, elbow, wristRoll, wristPitch, wristYaw]
# 左右ミラー: roll/yaw/wristRoll/wristYaw を反転、pitch/elbow/wristPitch は同符号
MIRROR_SIGN = np.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0])


def load_traj(path, mirror=True):
    """npz から (traj(N,14), dt) を返す。mirror=True なら記録腕を反対腕へ対称コピー。"""
    d = np.load(path, allow_pickle=True)
    traj = d['traj'].astype(float)
    dt = float(d['dt'])
    rec = str(d['arm'])
    if mirror:
        if rec == 'right':
            traj[:, 0:7] = traj[:, 7:14] * MIRROR_SIGN     # 右→左
        elif rec == 'left':
            traj[:, 7:14] = traj[:, 0:7] * MIRROR_SIGN     # 左→右
    return traj, dt


def make_grav(reduced_model, scale=1.0):
    """reduced_robot.model から重力補償トルク関数を作る (teleop/推論と同じ pin.rnea)。"""
    data = reduced_model.createData()
    z = np.zeros(reduced_model.nv)

    def grav(q):
        return scale * pin.rnea(reduced_model, data, np.asarray(q, dtype=float), z, z)
    return grav


def load_reduced_model_cache(cache_path):
    """robot_arm_ik と共有の g1_29_model_cache.pkl から reduced_model を読む。"""
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    m = data["reduced_model"]
    return m, m.createData()


def nearest_index(traj, q):
    """軌道中で現在姿勢 q(14) に最も近いサンプル番号。"""
    q = np.asarray(q, dtype=float)
    return int(np.argmin(np.linalg.norm(traj - q, axis=1)))


def _tauff(grav, q):
    return grav(q) if grav is not None else np.zeros(14)


def _wait_reached(arm, target, grav=None, tol=0.12, timeout=8.0, stop_check=None):
    target = np.asarray(target, dtype=float)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if stop_check is not None and stop_check():
            return False
        cur = arm.get_current_dual_arm_q()
        arm.ctrl_dual_arm(target, _tauff(grav, target))   # 待機中も指令維持(重力補償)
        if np.all(np.abs(cur - target) < tol):
            return True
        time.sleep(0.03)
    return False


def play(arm, traj, dt, grav=None, reverse=False, speed=1.0, vel=None,
         move_to_start=True, from_current=False, stop_check=None):
    """軌道を q_target でなぞって再生。

    grav         : 重力補償トルク関数 (make_grav) or None
    reverse      : True で逆再生
    from_current : True かつ reverse のとき、現在姿勢に最も近いサンプルから降下を開始
                   (終了時に一旦挙上端へ上げてしまうのを防ぐ)
    vel          : 指定すると arm.arm_velocity_limit をこの値に固定(速度ランプを無効化)
    move_to_start: 再生前に開始サンプルへ移動してから開始
    stop_check   : 呼んで True なら中断
    戻り値       : 最終到達 q(14)
    """
    if vel is not None:
        arm.arm_velocity_limit = vel
        arm._speed_gradual_max = False   # gradual-max ランプに上書きされないように
    N = traj.shape[0]
    if reverse:
        start = nearest_index(traj, arm.get_current_dual_arm_q()) if from_current else N - 1
        seq = list(range(start, -1, -1))
    else:
        seq = list(range(N))
    if not seq:
        return traj[0]
    if move_to_start:
        if not _wait_reached(arm, traj[seq[0]], grav=grav, stop_check=stop_check):
            logger_mp.warning("[arm_teach_motion] 開始姿勢に未到達(timeout)。速度制限で追従しつつ再生継続。")
    step_dt = dt / max(speed, 1e-3)
    last = traj[seq[0]]
    for i in seq:
        if stop_check is not None and stop_check():
            break
        q = traj[i]
        arm.ctrl_dual_arm(q, _tauff(grav, q))
        last = q
        time.sleep(step_dt)
    return last
