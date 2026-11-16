"""Closed-loop CAN simulation of a CUSW Jeep Cherokee (JEEP_CHEROKEE_5TH_GEN) running jvePilot.

The real CarInterface parses CUSW CAN frames (built with the CUSW DBC, counters and checksums) and its CarController
sends CRUISE_BUTTONS. A simulated stock ACC module consumes those frames the way FCA modules treat rolling counters
(a frame whose counter was already seen is dropped), changes its set speed / follow distance, and reports back on
ACC_HUD with configurable latency. Every frame openpilot sends is also checked against CUSW Panda safety.
"""
from collections import deque

import pytest

from opendbc.can import CANPacker
from opendbc.car import gen_empty_fingerprint, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.chrysler.carcontroller import ACC_BUTTON_HOLD_FRAMES, ACC_MAX_UNCONFIRMED
from opendbc.car.chrysler.interface import CarInterface
from opendbc.car.chrysler.values import CAR
from opendbc.safety.tests.libsafety import libsafety_py

CRUISE_BUTTONS = 0x2FA
DT_NANOS = 10_000_000  # 100 Hz
BUTTON_BITS = {"ACC_Cancel": 0, "ACC_Distance_Dec": 1, "ACC_Accel": 2, "ACC_Decel": 3, "ACC_Resume": 4, "ACC_Distance_Inc": 8}


class CuswStockAcc:
  """Simulated CUSW car + stock ACC module."""
  def __init__(self, set_kph=113.0, imperial=True, hud_hz=5, hud_latency_frames=10, min_set_kph=32.0, max_set_kph=145.0,
               distance_setting=1, buttons_work=True, v_ego=31.0, drop_odd_frames=False):
    self.packer = CANPacker("chrysler_cusw")
    self.frame = 0
    self.v_ego = v_ego
    self.step_kph = CV.MPH_TO_KPH if imperial else 1.0
    self.set_kph = set_kph
    self.min_set_kph, self.max_set_kph = min_set_kph, max_set_kph
    self.distance_setting = distance_setting  # 0..3
    self.buttons_work = buttons_work
    self.drop_odd_frames = drop_odd_frames
    self.acc_active = True
    self.hud_period = int(100 / hud_hz)
    self.hud_latency = hud_latency_frames
    self.history = deque()  # (frame, set_kph, distance) snapshots for delayed ACC_HUD reporting
    self.hud_set_kph, self.hud_distance = round(set_kph), distance_setting
    self.sccm_counter = 0
    self.last_counter = None
    self.driver_buttons: dict[str, int] = {}
    self.mailbox_button: str | None = None
    self.mailbox_from_openpilot = False
    self.held_button: str | None = None
    self.accepted_presses: list[str] = []

  def _accept_frame(self, values: dict[str, int], counter: int, from_openpilot: bool):
    if counter == self.last_counter:
      return  # duplicate rolling counter: dropped by the receiving module
    self.last_counter = counter
    pressed = [name for name, v in values.items() if v]
    self.mailbox_button = pressed[0] if pressed else None
    self.mailbox_from_openpilot = from_openpilot and bool(pressed)

  def _sample_mailbox(self):
    btn = self.mailbox_button
    if btn == self.held_button:
      return
    prev = self.held_button
    self.held_button = btn
    if btn is None or prev is not None or not self.buttons_work:
      return
    if self.mailbox_from_openpilot:
      self.accepted_presses.append(btn)
    if btn == "ACC_Cancel":
      self.acc_active = False
    elif btn == "ACC_Accel":
      self.set_kph = min(self.set_kph + self.step_kph, self.max_set_kph)
    elif btn == "ACC_Decel":
      self.set_kph = max(self.set_kph - self.step_kph, self.min_set_kph)
    elif btn == "ACC_Distance_Inc":
      self.distance_setting = min(self.distance_setting + 1, 3)
    elif btn == "ACC_Distance_Dec":
      self.distance_setting = max(self.distance_setting - 1, 0)

  def can_frames(self):
    p, f = self.packer, self.frame
    msgs = [
      p.make_can_msg("BRAKE_1", 0, {"VEHICLE_SPEED": self.v_ego}),
      p.make_can_msg("BRAKE_3", 0, {"DRIVER_BRAKE_SWITCH": 0}),
      p.make_can_msg("ACCEL_GAS", 0, {"GAS_HUMAN": 0}),
      p.make_can_msg("CLUSTER_1", 0, {"SPEEDOMETER": self.v_ego}),
      p.make_can_msg("WHEEL_SPEEDS_FRONT", 0, {"WHEEL_SPEED_FL": self.v_ego, "WHEEL_SPEED_FR": self.v_ego}),
      p.make_can_msg("WHEEL_SPEEDS_REAR", 0, {"WHEEL_SPEED_RL": self.v_ego, "WHEEL_SPEED_RR": self.v_ego}),
      p.make_can_msg("STEERING", 0, {"STEER_ANGLE": 0}),
      p.make_can_msg("EPS_STATUS", 0, {}),
      p.make_can_msg("ACC_CONTROL", 0, {"ACC_MAIN_ON": 1, "ACC_ACTIVE": int(self.acc_active)}),
      p.make_can_msg("DOORS", 0, {}),
      p.make_can_msg("SEATBELT_STATUS", 0, {}),
      p.make_can_msg("STEERING_LEVERS", 0, {}),
      p.make_can_msg("TRACTION_BUTTON", 0, {}),
      p.make_can_msg("GEAR", 0, {"PRNDL": 4}),
      p.make_can_msg("DAS_6", 2, {"CAR_MODEL": 1}),
    ]

    # SCCM steering-wheel buttons at 50 Hz
    if f % 2 == 0:
      self.sccm_counter = (self.sccm_counter + 1) % 16
      values = {name: self.driver_buttons.get(name, 0) for name in BUTTON_BITS}
      self._accept_frame(values, self.sccm_counter, from_openpilot=False)
      msgs.append(p.make_can_msg("CRUISE_BUTTONS", 0, {**values, "COUNTER": self.sccm_counter}))

    # stock ACC reports set speed / distance on ACC_HUD at hud_hz, hud_latency frames behind its internal state
    self.history.append((f, self.set_kph, self.distance_setting))
    while self.history and self.history[0][0] <= f - self.hud_latency:
      _, kph, dist = self.history.popleft()
      self.hud_set_kph, self.hud_distance = round(kph), dist
    if f % self.hud_period == 0:
      msgs.append(p.make_can_msg("ACC_HUD", 0, {"ACC_SET_SPEED_KMH": self.hud_set_kph, "DISTANCE_SETTING": self.hud_distance}))
    return msgs

  def receive_from_openpilot(self, sends):
    for addr, dat, bus in sends:
      if addr == CRUISE_BUTTONS and bus == 0:
        if self.drop_odd_frames and self.frame % 2 == 1:
          continue
        values = {name: (dat[bit // 8] >> (bit % 8)) & 1 for name, bit in BUTTON_BITS.items()}
        self._accept_frame(values, dat[1] >> 4, from_openpilot=True)
    self._sample_mailbox()


class Harness:
  def __init__(self, sim: CuswStockAcc):
    self.sim = sim
    self.CP = CarInterface.get_params(CAR.JEEP_CHEROKEE_5TH_GEN, gen_empty_fingerprint(), [], False, False, False)
    self.CI = CarInterface(self.CP)
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(structs.CarParams.SafetyModel.chryslerCusw, 0)
    self.safety.init_tests()
    self.t = 0
    self.CS = None
    self.sent = 0
    # warm up: CANParser registers messages lazily and learns frequencies
    self.run(200, target_kph=0.0)

  def run(self, frames, target_kph, driver_max_kph=113.0, bars=0, enabled=True):
    for _ in range(frames):
      self.t += DT_NANOS
      self.CS = self.CI.update([(self.t, self.sim.can_frames())])
      CC = structs.CarControl()
      CC.enabled = enabled and self.CS.cruiseState.enabled
      CC.actuators.speed = target_kph * CV.KPH_TO_MS
      CC.hudControl.setSpeed = driver_max_kph * CV.KPH_TO_MS
      CC.hudControl.leadDistanceBars = bars
      self.safety.set_controls_allowed(CC.enabled)
      _, sends = self.CI.apply(CC.as_reader(), self.t)
      for addr, dat, bus in sends:
        if addr == CRUISE_BUTTONS:
          assert self.sim.frame % 2 == 0, "transmitted CRUISE_BUTTONS on Tick B without new SCCM frame"
          assert self.safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, dat)), f"Panda rejected {dat.hex()}"
          self.sent += 1
      self.sim.receive_from_openpilot(sends)
      self.sim.frame += 1
    return self.CS


@pytest.mark.parametrize("imperial", [True, False])
@pytest.mark.parametrize("hud_hz,latency", [(10, 5), (5, 10), (2, 30)])
def test_cusw_set_speed_tracks_target_without_overshoot(imperial, hud_hz, latency):
  sim = CuswStockAcc(set_kph=113.0, imperial=imperial, hud_hz=hud_hz, hud_latency_frames=latency)
  h = Harness(sim)
  assert h.CS.canValid and h.CS.cruiseState.enabled
  assert h.sent == 0  # stock mode during warm-up sends nothing

  # slow for a curve: 113 -> 72 km/h
  h.run(3000, target_kph=72.0)
  # ends at the closest set speed the car can reach (within half a button step, plus ACC_HUD integer rounding)
  assert abs(sim.set_kph - 72.0) <= sim.step_kph / 2 + 0.5, sim.set_kph
  steps_down = len(sim.accepted_presses)
  assert steps_down <= (113.0 - 72.0) / sim.step_kph + 1, f"extra presses: {steps_down}"
  assert all(p == "ACC_Decel" for p in sim.accepted_presses)
  assert h.sent == steps_down * ACC_BUTTON_HOLD_FRAMES, "every held press should have been accepted once by the ACC module"

  # back up to the driver max after the curve
  sim.accepted_presses.clear()
  h.run(3000, target_kph=113.0)
  assert abs(sim.set_kph - 113.0) <= sim.step_kph / 2 + 0.5, sim.set_kph
  assert sim.set_kph <= 113.0 + 0.5, "never above the driver's max"
  assert all(p == "ACC_Accel" for p in sim.accepted_presses)
  assert not h.CI.CC.jve_inhibited["speed"]


def test_cusw_adverse_sccm_phase_still_tracks_target():
  # Even when odd-frame transmissions are overwritten by the 50 Hz SCCM before the ACC module samples, the 4-frame hold still succeeds
  sim = CuswStockAcc(set_kph=113.0, imperial=True, hud_hz=5, hud_latency_frames=10, drop_odd_frames=True)
  h = Harness(sim)
  h.run(3000, target_kph=72.0)
  assert abs(sim.set_kph - 72.0) <= sim.step_kph / 2 + 0.5, sim.set_kph
  assert not h.CI.CC.jve_inhibited["speed"]


@pytest.mark.parametrize("min_set_kph,start_kph", [(40.0, 50.0), (69.0, 74.0)])
def test_cusw_learns_real_min_set_speed_without_disabling(min_set_kph, start_kph):
  # car's real minimum is higher than the 32 km/h assumed (e.g. 69 km/h on 2019 Jeep Cherokee): one unanswered press at the floor, no fault
  sim = CuswStockAcc(set_kph=start_kph, min_set_kph=min_set_kph, imperial=False)
  h = Harness(sim)
  h.run(3000, target_kph=20.0, driver_max_kph=start_kph)
  assert sim.set_kph == min_set_kph
  assert h.CI.CC.jve_speed_floor is not None
  assert h.CI.CC.jve_unconfirmed["speed"] == 0
  assert not h.CI.CC.jve_inhibited["speed"]
  sent_at_floor = h.sent
  h.run(1000, target_kph=20.0, driver_max_kph=start_kph)
  assert h.sent == sent_at_floor, "kept pressing decel at the floor"
  h.run(2000, target_kph=start_kph, driver_max_kph=start_kph)
  assert sim.set_kph == start_kph  # speed control still works upward


def test_cusw_presses_ignored_fail_closed():
  sim = CuswStockAcc(set_kph=113.0, buttons_work=False)
  h = Harness(sim)
  h.run(3000, target_kph=72.0)
  assert h.sent == ACC_MAX_UNCONFIRMED * ACC_BUTTON_HOLD_FRAMES
  assert h.CI.CC.jve_inhibited["speed"]
  assert sim.set_kph == 113.0


def test_cusw_auto_follow_reaches_target_bars():
  sim = CuswStockAcc(set_kph=113.0, distance_setting=0)  # 1 bar
  h = Harness(sim)
  h.run(2000, target_kph=113.0, bars=4)
  assert sim.distance_setting == 3
  assert h.CI.CS.acc_follow_bars == 4
  assert h.sent == 3 * ACC_BUTTON_HOLD_FRAMES


def test_cusw_driver_buttons_take_priority():
  sim = CuswStockAcc(set_kph=113.0)
  h = Harness(sim)
  sim.driver_buttons = {"ACC_Accel": 1}  # driver holding RES+
  h.run(500, target_kph=72.0)
  assert sim.accepted_presses == []
  sim.driver_buttons = {}
  h.run(3000, target_kph=72.0)
  assert sim.accepted_presses and all(p == "ACC_Decel" for p in sim.accepted_presses)


def test_cusw_cancel_and_disengage_stop_presses():
  sim = CuswStockAcc(set_kph=113.0)
  h = Harness(sim)
  h.run(100, target_kph=72.0)
  sim.acc_active = False  # e.g. driver brake cancels stock ACC
  before = h.sent
  h.run(1000, target_kph=72.0)
  assert h.sent == before
