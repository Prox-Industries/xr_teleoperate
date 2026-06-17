#!/usr/bin/env python3
"""ティーチング記録: アームを脱力(コンプライアント)させ、手で動かした軌道を記録して保存。

teleop と同じ G1_29_ArmController (debug lowcmd) を使い、対象アームのモータを
kp=0 / kd=小 にして「手で自由に動かせる」状態にし、関節角を一定 Hz で N 秒記録する。
再生(playback)は別スクリプトで、この軌道を q_target になぞらせる。

⚠ 重要(安全):
  - kp=0 にすると対象アームは脱力し、自重で垂れます。**必ず腕を掴んだ状態**で
    「脱力(コンプライアント)化」してください。記録中も支えたまま動かす。
  - 脚・腰は剛性維持なのでロボットは立位のまま。対象でない腕も保持されます。
  - L2+B(ダンピング)を即押せる状態で。ロボットは debug mode(motion control 停止)で。
"""
import sys, os, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from robot_control.robot_arm import G1_29_ArmController

# q/motor 配列順と一致する関節ID (G1_29_JointArmIndex)
LEFT_IDS  = [15, 16, 17, 18, 19, 20, 21]   # L: ShoulderPitch,Roll,Yaw, Elbow, WristRoll,Pitch,Yaw
RIGHT_IDS = [22, 23, 24, 25, 26, 27, 28]   # R: 同順
WRIST_IDS = {19, 20, 21, 26, 27, 28}


def set_compliant(arm, ids, kp, kd):
    with arm.ctrl_lock:
        for jid in ids:
            arm.msg.motor_cmd[jid].kp = kp
            arm.msg.motor_cmd[jid].kd = kd


def restore_stiffness(arm, ids):
    with arm.ctrl_lock:
        for jid in ids:
            if jid in WRIST_IDS:
                arm.msg.motor_cmd[jid].kp = arm.kp_wrist
                arm.msg.motor_cmd[jid].kd = arm.kd_wrist
            else:
                arm.msg.motor_cmd[jid].kp = arm.kp_low
                arm.msg.motor_cmd[jid].kd = arm.kd_low


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', default='enx6c6e07085087', help='DDS network interface')
    ap.add_argument('--arm', choices=['left', 'right', 'both'], default='right',
                    help='脱力して実演・記録する腕 (既定 right。片腕実演→再生時にミラー可)')
    ap.add_argument('--sec', type=float, default=5.0, help='記録時間 [s]')
    ap.add_argument('--hz', type=float, default=50.0, help='記録サンプリング周波数 [Hz]')
    ap.add_argument('--kd', type=float, default=1.0, help='脱力中のダンピング kd (小さいほど軽い)')
    ap.add_argument('--out', default='utils/teach/raise_teach.npz', help='保存先')
    args = ap.parse_args()

    ids = {'left': LEFT_IDS, 'right': RIGHT_IDS, 'both': LEFT_IDS + RIGHT_IDS}[args.arm]

    ChannelFactoryInitialize(0, args.net)
    arm = G1_29_ArmController(motion_mode=False)   # teleop と同じ debug mode
    # ★起動直後、コントローラ既定の q_target=0 へ駆動して腕が上がるのを防ぐ:
    #   即・現在姿勢を q_target に固定 (delta≈0 → 動かない)
    arm.ctrl_dual_arm(arm.get_current_dual_arm_q(), np.zeros(14))
    time.sleep(0.3)

    print("\n================ ティーチング記録 ================")
    print(f"  対象アーム : {args.arm}  /  記録 {args.sec}s @ {args.hz}Hz")
    print("  起動時は現在の姿勢を保持します(腕は勝手に上がりません)。")
    print("  ⚠ これから対象アームを脱力させます。腕は『今のうちに掴んで』ください。")
    print("    脚・腰・反対腕は保持されます。L2+B 即押せる状態で。\n")
    try:
        input("[Enter] 腕を掴んだ状態で → 脱力(コンプライアント)化 ...")
        set_compliant(arm, ids, kp=0.0, kd=args.kd)
        print("  → 脱力しました。腕は支えたまま、開始姿勢(腕を下げた状態)にしてください。")

        input("[Enter] 記録開始(3秒カウント後 {:.0f}秒間記録) ...".format(args.sec))
        for c in (3, 2, 1):
            print(f"   ... {c}")
            time.sleep(1.0)
        print(f"   ▶ 記録中 ({args.sec}s) — 下→横→挙上→前 など、出したい軌道で動かしてください")

        dt = 1.0 / args.hz
        traj = []
        stamps = []
        t0 = time.time()
        next_t = t0
        while time.time() - t0 < args.sec:
            traj.append(arm.get_current_dual_arm_q().copy())
            stamps.append(time.time() - t0)
            next_t += dt
            time.sleep(max(0.0, next_t - time.time()))
        traj = np.array(traj)          # (N, 14)
        print(f"   ■ 記録終了: {traj.shape[0]} サンプル / {stamps[-1]:.2f}s")

        # 落下防止: 現在(最終)姿勢で q_target を固定 → 剛性を戻して保持
        q_end = arm.get_current_dual_arm_q()
        arm.ctrl_dual_arm(q_end, np.zeros(14))
        restore_stiffness(arm, ids)
        print("  → 最終姿勢で剛性を戻して保持しました(腕を離してOK)。")

        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(out, traj=traj, stamps=np.array(stamps), dt=dt, hz=args.hz, arm=args.arm)
        print(f"\n保存: {out}")
        # サマリ(対象腕の開始/終了角)
        idx0 = 0 if args.arm in ('left', 'both') else 7
        print(f"  {args.arm} 開始 q: {np.round(traj[0][idx0:idx0+7], 3)}")
        print(f"  {args.arm} 終了 q: {np.round(traj[-1][idx0:idx0+7], 3)}")

        input("\n[Enter] home(腕を下げる)へ戻す ...")
        arm.ctrl_dual_arm_go_home(); time.sleep(2.0)
        print("done.")
    except KeyboardInterrupt:
        print("\n中断 → 現在姿勢で保持してから home へ戻します ...")
        q_now = arm.get_current_dual_arm_q()
        arm.ctrl_dual_arm(q_now, np.zeros(14))
        restore_stiffness(arm, ids)
        time.sleep(0.5)
        arm.ctrl_dual_arm_go_home(); time.sleep(2.0)


if __name__ == '__main__':
    main()
