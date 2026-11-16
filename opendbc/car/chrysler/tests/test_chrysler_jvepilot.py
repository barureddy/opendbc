"""jvePilot hybrid ACC tests for Chrysler/CUSW: virtual cruise-button control, feedback/fail-closed logic,
safety guards, CarState parsing, and Panda safety acceptance of every frame we send."""
from types import SimpleNamespace

import pytest

from opendbc.can import CANPacker
from opendbc.car import Bus, gen_empty_fingerprint, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.chrysler.carcontroller import (ACC_BUTTON_HOLD_FRAMES, ACC_BUTTON_STEP, ACC_FEEDBACK_TIMEOUT,
                                                ACC_MAX_UNCONFIRMED, CarController)
from opendbc.car.chrysler import chryslercan
from opendbc.car.chrysler.carstate import CarState
from opendbc.car.chrysler.interface import CarInterface
from opendbc.car.chrysler.values import CAR, DBC, ChryslerFlags
from opendbc.safety.tests.libsafety import libsafety_py

CUSW = CAR.JEEP_CHEROKEE_5TH_GEN
CRUISE_BUTTONS_CUSW = 0x2FA
BSM_LEFT_CUSW = 0x62CC033


def cusw_cp(bsm=False):
  fingerprint = gen_empty_fingerprint()
  if bsm:
    fingerprint[0][BSM_LEFT_CUSW] = 8
  return CarInterface.get_params(CUSW, fingerprint, [], False, False, False)


def make_cs(set_speed_kph=113.0, bars=2, **kwargs):
  out = structs.CarState()
  out.vEgo = 31.0
  out.canValid = True
  out.cruiseState.available = True
  out.cruiseState.enabled = True
  out.cruiseState.speed = set_speed_kph * CV.KPH_TO_MS
  for k, v in kwargs.items():
    setattr(out, k, v)
  return SimpleNamespace(out=out, button_counter=0, lkas_car_model=-1, auto_high_beam=0,
                         cruise_button_pressed=False, acc_follow_bars=bars)


def make_cc(target_kph=72.0, set_speed_kph=113.0, bars=0, enabled=True):
  CC = structs.CarControl()
  CC.enabled = enabled
  CC.actuators.speed = target_kph * CV.KPH_TO_MS
  CC.hudControl.setSpeed = set_speed_kph * CV.KPH_TO_MS
  CC.hudControl.leadDistanceBars = bars
  return CC


class TestJvePilotCarController:
  def setup_method(self):
    self.CP = cusw_cp()
    self.cc = CarController(DBC[CUSW], self.CP)
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(structs.CarParams.SafetyModel.chryslerCusw, 0)
    self.safety.init_tests()
    self.safety.set_controls_allowed(True)

  def _step(self, CC, CS, n=1):
    """Run n frames; returns list of (frame_idx, data) for CRUISE_BUTTONS frames, each checked against Panda safety."""
    sent = []
    for i in range(n):
      CS.button_counter = (CS.button_counter + 1) % 16
      _, sends = self.cc.update(CC.as_reader(), CS, 0)
      for addr, dat, bus in sends:
        if addr == CRUISE_BUTTONS_CUSW:
          assert self.safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, dat)), dat.hex()
          assert (dat[1] >> 4) == (CS.button_counter + 1) % 16, "counter must be SCCM counter + 1"
          sent.append((i, bytes(dat)))
    return sent

  def test_stock_mode_sends_nothing(self):
    assert self._step(make_cc(target_kph=0.0), make_cs(), 200) == []

  def test_decel_waits_for_feedback_then_confirms(self):
    CC, CS = make_cc(target_kph=72.0), make_cs(set_speed_kph=113.0)
    first = self._step(CC, CS, ACC_BUTTON_STEP + ACC_BUTTON_HOLD_FRAMES)
    assert len(first) == ACC_BUTTON_HOLD_FRAMES and all(f[0] & 0x08 for _, f in first)  # ACC_Decel held

    # no stock ACC response yet: must not press again before the feedback timeout
    assert self._step(CC, CS, ACC_FEEDBACK_TIMEOUT - 10) == []

    # stock ACC lowers its set speed -> press confirmed, next press follows after ACC_BUTTON_STEP
    CS.out.cruiseState.speed = 112.0 * CV.KPH_TO_MS
    nxt = self._step(CC, CS, ACC_BUTTON_STEP + ACC_BUTTON_HOLD_FRAMES)
    assert len(nxt) == ACC_BUTTON_HOLD_FRAMES and all(f[0] & 0x08 for _, f in nxt)
    assert self.cc.jve_unconfirmed["speed"] == 0

  def test_accel_press(self):
    sent = self._step(make_cc(target_kph=113.0), make_cs(set_speed_kph=72.0), ACC_BUTTON_STEP + ACC_BUTTON_HOLD_FRAMES)
    assert len(sent) == ACC_BUTTON_HOLD_FRAMES and all(f[0] & 0x04 for _, f in sent)  # ACC_Accel

  def test_unconfirmed_presses_inhibit_speed_control(self):
    CC, CS = make_cc(target_kph=72.0), make_cs(set_speed_kph=113.0)
    sent = self._step(CC, CS, (ACC_FEEDBACK_TIMEOUT + 1) * (ACC_MAX_UNCONFIRMED + 2))
    assert len(sent) == ACC_MAX_UNCONFIRMED * ACC_BUTTON_HOLD_FRAMES
    assert self.cc.jve_inhibited["speed"]
    # re-engagement clears the inhibit
    self._step(make_cc(enabled=False), CS, 1)
    assert not self.cc.jve_inhibited["speed"]

  def test_unconfirmed_inhibit_recovers_when_demand_clears_or_target_drops(self):
    CC, CS = make_cc(target_kph=87.0, set_speed_kph=89.0), make_cs(set_speed_kph=89.0)
    sent = self._step(CC, CS, (ACC_FEEDBACK_TIMEOUT + 1) * (ACC_MAX_UNCONFIRMED + 1))
    assert len(sent) == ACC_MAX_UNCONFIRMED * ACC_BUTTON_HOLD_FRAMES
    assert self.cc.jve_inhibited["speed"] and self.cc.jve_inhibited_dir["speed"] == -1

    # When target returns to current set speed (speed_dir == 0), transient unconfirmed inhibit clears
    CC.actuators.speed = 89.0 * CV.KPH_TO_MS
    self._step(CC, CS, 1)
    assert not self.cc.jve_inhibited["speed"]

    # Inhibit again during minor trim demand, then drop target for slower traffic (15:11:33 scenario)
    CC.actuators.speed = 87.0 * CV.KPH_TO_MS
    self._step(CC, CS, (ACC_FEEDBACK_TIMEOUT + 1) * (ACC_MAX_UNCONFIRMED + 1))
    assert self.cc.jve_inhibited["speed"]
    CC.actuators.speed = 70.0 * CV.KPH_TO_MS
    recovered = self._step(CC, CS, ACC_BUTTON_STEP + ACC_BUTTON_HOLD_FRAMES)
    assert not self.cc.jve_inhibited["speed"]
    assert len(recovered) == ACC_BUTTON_HOLD_FRAMES and all(f[0] & 0x08 for _, f in recovered)

  def test_tick_b_suppressed_when_cruise_buttons_not_received(self):
    CC, CS = make_cc(target_kph=72.0), make_cs(set_speed_kph=113.0)
    sent_even, sent_odd = [], []
    for i in range(ACC_BUTTON_STEP + ACC_BUTTON_HOLD_FRAMES * 2 + 2):
      CS.cruise_buttons_rx = (i % 2 == 0)
      if CS.cruise_buttons_rx:
        CS.button_counter = (CS.button_counter + 1) % 16
      _, sends = self.cc.update(CC.as_reader(), CS, 0)
      for addr, dat, bus in sends:
        if addr == CRUISE_BUTTONS_CUSW:
          assert self.safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, dat)), dat.hex()
          (sent_even if CS.cruise_buttons_rx else sent_odd).append(bytes(dat))
    assert sent_odd == []
    assert len(sent_even) == ACC_BUTTON_HOLD_FRAMES

    # Standstill resume also suppresses Tick-B frames when cruise_buttons_rx is False
    CC_res, CS_res = make_cc(target_kph=0.0), make_cs(set_speed_kph=50.0)
    CC_res.cruiseControl.resume = True
    CS_res.out.cruiseState.standstill = True
    CS_res.cruise_buttons_rx = False
    for _ in range(6):
      assert [a for a, _, _ in self.cc.update(CC_res.as_reader(), CS_res, 0)[1] if a == CRUISE_BUTTONS_CUSW] == []
    CS_res.cruise_buttons_rx = True
    CS_res.button_counter = (CS_res.button_counter + 1) % 16
    res_frames = [d for a, d, _ in self.cc.update(CC_res.as_reader(), CS_res, 0)[1] if a == CRUISE_BUTTONS_CUSW]
    assert len(res_frames) == 1 and res_frames[0][0] == 0x10

  def test_opposite_response_inhibits(self):
    CC, CS = make_cc(target_kph=72.0), make_cs(set_speed_kph=113.0)
    self._step(CC, CS, ACC_BUTTON_STEP + ACC_BUTTON_HOLD_FRAMES)
    CS.out.cruiseState.speed = 114.0 * CV.KPH_TO_MS  # went up after a decel press
    self._step(CC, CS, 1)
    assert self.cc.jve_inhibited["speed"]

  def test_auto_follow_uses_can_feedback(self):
    CC, CS = make_cc(target_kph=113.0, bars=3), make_cs(set_speed_kph=113.0, bars=1)
    sent = self._step(CC, CS, ACC_BUTTON_STEP + ACC_BUTTON_HOLD_FRAMES)
    assert len(sent) == ACC_BUTTON_HOLD_FRAMES and all(f[1] & 0x01 for _, f in sent)  # ACC_Distance_Inc (byte 1 bit 0)
    CS.acc_follow_bars = 2
    sent = self._step(CC, CS, ACC_BUTTON_STEP + ACC_BUTTON_HOLD_FRAMES)
    assert len(sent) == ACC_BUTTON_HOLD_FRAMES and all(f[1] & 0x01 for _, f in sent)
    CS.acc_follow_bars = 3
    assert self._step(CC, CS, 100) == []  # target reached

  def test_auto_follow_wrong_direction_inhibits_distance_only(self):
    CC, CS = make_cc(target_kph=113.0, bars=3), make_cs(set_speed_kph=113.0, bars=2)
    self._step(CC, CS, ACC_BUTTON_STEP + ACC_BUTTON_HOLD_FRAMES)
    CS.acc_follow_bars = 1  # Distance_Inc reduced bars: mapping assumption wrong -> stop Auto Follow
    self._step(CC, CS, 1)
    assert self.cc.jve_inhibited["dist"] and not self.cc.jve_inhibited["speed"]
    assert self._step(CC, CS, 100) == []
    # speed control still works
    CC.actuators.speed = 72.0 * CV.KPH_TO_MS
    sent = self._step(CC, CS, ACC_BUTTON_STEP + ACC_BUTTON_HOLD_FRAMES)
    assert len(sent) == ACC_BUTTON_HOLD_FRAMES and all(f[0] & 0x08 for _, f in sent)

  def test_auto_follow_requires_known_can_bars(self):
    assert self._step(make_cc(target_kph=113.0, bars=3), make_cs(set_speed_kph=113.0, bars=0), 100) == []

  @pytest.mark.parametrize("guard", ["brakePressed", "gasPressed", "accFaulted", "canInvalid", "driverButton", "standstill"])
  def test_guards_block_presses(self, guard):
    CC, CS = make_cc(target_kph=72.0), make_cs(set_speed_kph=113.0)
    if guard == "canInvalid":
      CS.out.canValid = False
    elif guard == "driverButton":
      CS.cruise_button_pressed = True
    elif guard == "standstill":
      CS.out.cruiseState.standstill = True
    else:
      setattr(CS.out, guard, True)
    assert self._step(CC, CS, 200) == []

  def test_never_above_driver_set_speed_or_below_minimum(self):
    # target above driver max -> clamp, already at driver max: no accel
    assert self._step(make_cc(target_kph=140.0, set_speed_kph=113.0), make_cs(set_speed_kph=113.0), 100) == []
    # target below FCA minimum while at minimum: no decel
    assert self._step(make_cc(target_kph=10.0), make_cs(set_speed_kph=32.0), 100) == []

  def test_radar_lead_distance_caps_target_speed(self):
    CC, CS = make_cc(target_kph=113.0, set_speed_kph=113.0), make_cs(set_speed_kph=113.0, bars=2)
    CS.acc_lead_detected = True
    CS.acc_lead_distance = 18.0
    sent = self._step(CC, CS, ACC_BUTTON_STEP + ACC_BUTTON_HOLD_FRAMES)
    assert len(sent) == ACC_BUTTON_HOLD_FRAMES
    assert all((d[0] & 0x08) and not (d[0] & 0x04) for _, d in sent)

  def test_single_missed_decel_at_43mph_does_not_latch_false_floor(self):
    CC, CS = make_cc(target_kph=50.0, set_speed_kph=72.0), make_cs(set_speed_kph=69.0)  # 43 mph
    first = self._step(CC, CS, ACC_BUTTON_STEP + ACC_BUTTON_HOLD_FRAMES)
    assert len(first) == ACC_BUTTON_HOLD_FRAMES and all(f[0] & 0x08 for _, f in first)
    # Time out the first press at 69 km/h (43 mph): must NOT latch jve_speed_floor on a single missed press, and must retry
    nxt = self._step(CC, CS, ACC_FEEDBACK_TIMEOUT + 1)
    assert self.cc.jve_speed_floor is None
    assert len(nxt) == ACC_BUTTON_HOLD_FRAMES and all(f[0] & 0x08 for _, f in nxt)
    # Second unconfirmed decel press at 69 km/h (43 mph) learns the hardware floor without inhibiting speed control
    self._step(CC, CS, ACC_FEEDBACK_TIMEOUT + 1)
    assert self.cc.jve_speed_floor == pytest.approx(69.0 * CV.KPH_TO_MS)
    assert not self.cc.jve_inhibited["speed"]
    # When target rises back to 72 km/h, SET+ is sent immediately to restore dashboard set speed
    CC.actuators.speed = 72.0 * CV.KPH_TO_MS
    up = self._step(CC, CS, ACC_BUTTON_STEP + ACC_BUTTON_HOLD_FRAMES)
    assert len(up) == ACC_BUTTON_HOLD_FRAMES and all(f[0] & 0x04 for _, f in up)

  def test_acc_hud_mph_signal_preferred_when_present(self):
    CP = cusw_cp()
    cs = CarState(CP)
    parsers = cs.get_can_parsers(CP)
    cs.update(parsers)
    packer = CANPacker(DBC[CUSW][Bus.pt])
    # Real SEG5 frame: ACC_SET_SPEED_MPH=55, ACC_SET_SPEED_KMH=89 (89 km/h = 55.30 mph)
    msgs = [
      packer.make_can_msg("ACC_CONTROL", 0, {"ACC_MAIN_ON": 1, "ACC_ACTIVE": 1}),
      packer.make_can_msg("ACC_HUD", 0, {"ACC_SET_SPEED_KMH": 89, "ACC_SET_SPEED_MPH": 55, "DISTANCE_SETTING": 0}),
    ]
    parsers[Bus.pt].update([(10_000_000, msgs)])
    out = cs.update(parsers)
    assert out.cruiseState.speed == pytest.approx(55.0 * CV.MPH_TO_MS)
    assert out.cruiseState.speedCluster == pytest.approx(55.0 * CV.MPH_TO_MS)

  def test_resume_guarded_by_brake(self):
    CC, CS = make_cc(target_kph=0.0), make_cs(set_speed_kph=50.0)
    CC.cruiseControl.resume = True
    CS.out.cruiseState.standstill = True
    CS.out.brakePressed = True
    assert self._step(CC, CS, 50) == []
    CS.out.brakePressed = False
    sent = self._step(CC, CS, 50)
    assert len(sent) > 0 and all(d[0] == 0x10 for _, d in sent)  # ACC_Resume only

  def test_cancel_always_sent(self):
    CC, CS = make_cc(target_kph=72.0), make_cs(set_speed_kph=113.0, brakePressed=True)
    CC.cruiseControl.cancel = True
    CS.cruise_button_pressed = True
    sent = self._step(CC, CS, 10)
    assert len(sent) > 0 and all(d[0] == 0x01 for _, d in sent)

  def test_driver_button_post_release_cooldown(self):
    CC, CS = make_cc(target_kph=72.0), make_cs(set_speed_kph=113.0)
    CS.cruise_button_pressed = True
    assert self._step(CC, CS, 30) == []
    CS.cruise_button_pressed = False
    # must wait ACC_BUTTON_STEP frames after driver releases button before sending virtual press
    assert self._step(CC, CS, ACC_BUTTON_STEP - 1) == []
    sent = self._step(CC, CS, ACC_BUTTON_HOLD_FRAMES)
    assert len(sent) == ACC_BUTTON_HOLD_FRAMES

  def test_cusw_disengage_ramp_down_on_cruise_disable_before_lat_active_drops(self):
    CC, CS = make_cc(target_kph=0.0), make_cs(set_speed_kph=113.0)
    CC.latActive = True
    CC.actuators.torque = 0.5
    # pass the 200-frame initial LKAS cooldown and build up torque
    for _ in range(250):
      self.safety.set_torque_meas(self.cc.apply_torque_last, self.cc.apply_torque_last)
      CS.out.steeringTorqueEps = float(self.cc.apply_torque_last)
      _, sends = self.cc.update(CC.as_reader(), CS, 0)
      for addr, dat, bus in sends:
        if addr == 0x1F6:
          assert self.safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, dat))
    assert self.cc.apply_torque_last > 100

    # ACC_ACTIVE drops on CAN (so Panda safety immediately clears controls_allowed), while controlsd still has CC.latActive=True for 1 frame
    CS.out.cruiseState.enabled = False
    self.safety.set_controls_allowed(False)
    while self.cc.apply_torque_last > 0:
      _, sends = self.cc.update(CC.as_reader(), CS, 0)
      for addr, dat, bus in sends:
        if addr == 0x1F6:
          assert self.safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, dat)), "Panda rejected ramp-down frame"
      CC.latActive = False

  def test_cusw_das6_hud_matches_stock_camera(self):
    packer = CANPacker(DBC[CUSW][Bus.pt])
    _, dat_standby, _ = chryslercan.create_lkas_hud(packer, self.CP, False, structs.CarControl.HUDControl.VisualAlert.none, 10, 0x68, 0)
    _, dat_active, _ = chryslercan.create_lkas_hud(packer, self.CP, True, structs.CarControl.HUDControl.VisualAlert.none, 10, 0x68, 0)
    _, dat_alert, _ = chryslercan.create_lkas_hud(packer, self.CP, True, structs.CarControl.HUDControl.VisualAlert.steerRequired, 10, 0x68, 0)
    _, dat_off, _ = chryslercan.create_lkas_hud(packer, self.CP, False, structs.CarControl.HUDControl.VisualAlert.none, 10, 0x00, 0)
    assert bytes(dat_standby) == bytes.fromhex("01680100")
    assert bytes(dat_active) == bytes.fromhex("02680600")
    assert bytes(dat_alert) == bytes.fromhex("04680400")
    assert bytes(dat_off) == bytes.fromhex("00000000")


class TestJvePilotCarStateCusw:
  def test_parse(self):
    CP = cusw_cp(bsm=True)
    assert CP.flags & ChryslerFlags.HAS_BSM
    cs = CarState(CP)
    parsers = cs.get_can_parsers(CP)
    cs.update(parsers)  # CANParser registers messages lazily on first access
    packer = CANPacker(DBC[CUSW][Bus.pt])
    msgs = [
      packer.make_can_msg("CLUSTER_1", 0, {"SPEEDOMETER": 31.5}),
      packer.make_can_msg("ACC_CONTROL", 0, {"ACC_MAIN_ON": 1, "ACC_ACTIVE": 1}),
      packer.make_can_msg("ACC_HUD", 0, {"ACC_SET_SPEED_KMH": 113, "DISTANCE_SETTING": 2, "LEAD_DETECTED": 1, "DISTANCE_TO_LEAD": 25}),
      packer.make_can_msg("CRUISE_BUTTONS", 0, {"ACC_Accel": 1, "COUNTER": 7}),
      packer.make_can_msg("BSM_LEFT", 0, {"LEFT_DETECTED": 1}),
      packer.make_can_msg("BSM_RIGHT", 0, {"RIGHT_DETECTED": 0}),
    ]
    parsers[Bus.pt].update([(10_000_000, msgs)])
    out = cs.update(parsers)
    assert cs.button_counter == 7
    assert cs.cruise_buttons_rx
    assert cs.acc_follow_bars == 3
    assert cs.acc_lead_detected
    assert cs.acc_lead_distance == 25.0
    assert cs.cruise_button_pressed
    assert not out.accFaulted
    assert abs(out.vEgoCluster - 31.5) < 0.05
    assert out.cruiseState.enabled and abs(out.cruiseState.speed - 113 * CV.KPH_TO_MS) < 1e-3
    assert out.leftBlindspot and not out.rightBlindspot
    assert len(out.buttonEvents) == 1 and out.buttonEvents[0].type == structs.CarState.ButtonEvent.Type.accelCruise

    # ACC_HUD byte 2 bits [5:0] == 0x3A signals ACC/FCW fault; repeated button counter marks cruise_buttons_rx False
    parsers[Bus.pt].update([(20_000_000, [
      packer.make_can_msg("ACC_HUD", 0, {"ACC_SET_SPEED_KMH": 113, "ACC_ICON": 0x3A, "DISTANCE_SETTING": 2}),
    ])])
    out_fault = cs.update(parsers)
    assert out_fault.accFaulted
    assert not cs.cruise_buttons_rx

    # verify CUSW lowSpeedAlert hysteresis (18.5 m/s up, 16.5 m/s down)
    for speed, expected_alert in [(17.5, True), (19.0, False), (17.5, False), (16.0, True), (17.5, True), (19.0, False)]:
      for _ in range(50):
        parsers[Bus.pt].update([(10_000_000, [packer.make_can_msg("BRAKE_1", 0, {"VEHICLE_SPEED": speed})])])
        out = cs.update(parsers)
      assert out.lowSpeedAlert == expected_alert, f"speed={speed} lowSpeedAlert={out.lowSpeedAlert}"

  def test_cusw_dbc_new_messages(self):
    from opendbc.can import CANParser
    packer = CANPacker(DBC[CUSW][Bus.pt])
    parser = CANParser(DBC[CUSW][Bus.pt], [
      ("EPS_STATUS_2", 100),
      ("ENGINE_1", 100),
      ("ECM_TORQUE", 100),
      ("ECM_2", 100),
      ("ACC_2", 100),
      ("WHEEL_PULSES", 50),
    ], 0)
    msgs = [
      packer.make_can_msg("EPS_STATUS_2", 0, {"LONG_ACCEL": 0.5, "LATERAL_ACCEL": -0.3, "YAW_RATE": 0.1}),
      packer.make_can_msg("ENGINE_1", 0, {"ENGINE_RPM": 1800, "ACTUAL_ENGINE_TORQUE": 120}),
      packer.make_can_msg("ECM_TORQUE", 0, {"REQUESTED_ENGINE_TORQUE": 150, "EXPECTED_ENGINE_TORQUE": 148}),
      packer.make_can_msg("ECM_2", 0, {"PEDAL_DEMAND": 24.0, "ENGINE_DRAG_TORQUE": -20}),
      packer.make_can_msg("ACC_2", 0, {"ACC_STATUS_1": 1, "ACC_DECEL_CMD": 8000}),
      packer.make_can_msg("WHEEL_PULSES", 0, {"WHEEL_PULSE_FL": 42, "WHEEL_PULSE_FR": 43, "WHEEL_PULSE_RL": 44, "WHEEL_PULSE_RR": 45}),
    ]
    parser.update([(10_000_000, msgs)])
    assert abs(parser.vl["EPS_STATUS_2"]["LONG_ACCEL"] - 0.5) < 0.02
    assert abs(parser.vl["EPS_STATUS_2"]["LATERAL_ACCEL"] - (-0.3)) < 0.02
    assert abs(parser.vl["EPS_STATUS_2"]["YAW_RATE"] - 0.1) < 0.01
    assert parser.vl["ENGINE_1"]["ENGINE_RPM"] == 1800
    assert parser.vl["ENGINE_1"]["ACTUAL_ENGINE_TORQUE"] == 120
    assert parser.vl["ECM_TORQUE"]["REQUESTED_ENGINE_TORQUE"] == 150
    assert parser.vl["ECM_TORQUE"]["EXPECTED_ENGINE_TORQUE"] == 148
    assert parser.vl["WHEEL_PULSES"]["WHEEL_PULSE_FL"] == 42
    assert parser.vl["WHEEL_PULSES"]["WHEEL_PULSE_RR"] == 45
