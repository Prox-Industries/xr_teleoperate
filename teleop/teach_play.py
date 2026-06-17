#!/usr/bin/env python3
"""ティーチング再生(単体テスト): teach_record.py の記録軌道を再生する。

再生ロジック(軌道ロード/ミラー/重力補償/順逆再生)は arm_teach_motion に集約し、
ここからは呼ぶだけ(teleop 起動時の挙上と同一コードを使う)。

  --mirror : 記録腕を反対腕にもミラーして両腕で再生
  --grav   : 重力補償スケール(垂れるなら>1)

⚠ 安全: 記録動作を自動再生します。前後左右クリアランス確保 + L2+B 即押せる状態で。
"""
import sys, os, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from robot_control.robot_arm import G1_29_ArmController
from arm_teach_motion import load_traj, make_grav, load_reduced_model_cache, play

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', default='enx6c6e07085087')
    ap.add_argument('--file', default='utils/teach/raise_teach.npz')
    ap.add_argument('--vel', type=float, default=10.0, help='arm velocity limit (rad/s)')
    ap.add_argument('--speed', type=float, default=1.0, help='再生速度倍率')
    ap.add_argument('--grav', type=float, default=1.0, help='重力補償スケール(垂れるなら>1)')
    ap.add_argument('--mirror', action='store_true', help='記録腕を反対腕にもミラーして両腕で再生')
    args = ap.parse_args()

    traj, dt = load_traj(os.path.join(HERE, args.file), mirror=args.mirror)
    model, _ = load_reduced_model_cache(os.path.join(HERE, 'g1_29_model_cache.pkl'))
    grav = make_grav(model, scale=args.grav)
    print(f"{traj.shape[0]} samples, dt={dt:.4f}s, mirror={args.mirror}, grav×{args.grav}")

    ChannelFactoryInitialize(0, args.net)
    arm = G1_29_ArmController(motion_mode=False)
    arm.ctrl_dual_arm(arm.get_current_dual_arm_q(), np.zeros(14))   # 起動時保持(勝手に上がらない)
    time.sleep(0.3)

    print(f"\n⚠ 記録動作を再生 (vel={args.vel}, speed×{args.speed})。クリアランス + L2+B。")
    try:
        input("[Enter] 1) 開始姿勢へ → 順再生(挙上) ...")
        play(arm, traj, dt, grav=grav, reverse=False, vel=args.vel, speed=args.speed, move_to_start=True)
        print("   → 順再生完了。最終姿勢を保持中。")

        input("[Enter] 2) 逆再生で戻す ...")
        play(arm, traj, dt, grav=grav, reverse=True, vel=args.vel, speed=args.speed, move_to_start=True)
        print("   → 逆再生完了。")

        input("[Enter] 3) home (腕を下げる) へ ...")
        arm._speed_gradual_max = False
        arm.ctrl_dual_arm_go_home(); time.sleep(2.0)
        print("done.")
    except KeyboardInterrupt:
        print("\n中断 → home へ ...")
        arm.ctrl_dual_arm_go_home(); time.sleep(2.0)


if __name__ == '__main__':
    main()
