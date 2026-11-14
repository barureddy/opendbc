"""jvePilot hybrid ACC tests for Chrysler/CUSW: virtual cruise-button control, feedback/fail-closed logic,
safety guards, CarState parsing, and Panda safety acceptance of every frame we send."""
from types import SimpleNamespace

import pytest

from opendbc.can import CANPacker
from opendbc.car import Bus, gen_empty_fingerprint, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.chrysler.carcontroller import (ACC_BUTTON_STEP, ACC_FEEDBACK_TIMEOUT, ACC_MAX_UNCONFIRMED,
                                                CarController)
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
    first = self._step(CC, CS, ACC_BUTTON_STEP + 1)
    assert len(first) == 1 and first[0][1][0] & 0x08  # ACC_Decel

    # no stock ACC response yet: must not press again before the feedback timeout
    assert self._step(CC, CS, ACC_FEEDBACK_TIMEOUT - 5) == []

    # stock ACC lowers its set speed -> press confirmed, next press follows after ACC_BUTTON_STEP
    CS.out.cruiseState.speed = 112.0 * CV.KPH_TO_MS
    nxt = self._step(CC, CS, ACC_BUTTON_STEP + 1)
    assert len(nxt) == 1 and nxt[0][1][0] & 0x08
    assert self.cc.jve_unconfirmed["speed"] == 0

  def test_accel_press(self):
    sent = self._step(make_cc(target_kph=113.0), make_cs(set_speed_kph=72.0), ACC_BUTTON_STEP + 1)
    assert len(sent) == 1 and sent[0][1][0] & 0x04  # ACC_Accel

  def test_unconfirmed_presses_inhibit_speed_control(self):
    CC, CS = make_cc(target_kph=72.0), make_cs(set_speed_kph=113.0)
    sent = self._step(CC, CS, (ACC_FEEDBACK_TIMEOUT + 1) * (ACC_MAX_UNCONFIRMED + 2))
    assert len(sent) == ACC_MAX_UNCONFIRMED
    assert self.cc.jve_inhibited["speed"]
    # re-engagement clears the inhibit
    self._step(make_cc(enabled=False), CS, 1)
    assert not self.cc.jve_inhibited["speed"]

  def test_opposite_response_inhibits(self):
    CC, CS = make_cc(target_kph=72.0), make_cs(set_speed_kph=113.0)
    self._step(CC, CS, ACC_BUTTON_STEP + 1)
    CS.out.cruiseState.speed = 114.0 * CV.KPH_TO_MS  # went up after a decel press
    self._step(CC, CS, 1)
    assert self.cc.jve_inhibited["speed"]

  def test_auto_follow_uses_can_feedback(self):
    CC, CS = make_cc(target_kph=113.0, bars=3), make_cs(set_speed_kph=113.0, bars=1)
    sent = self._step(CC, CS, ACC_BUTTON_STEP + 1)
    assert len(sent) == 1 and sent[0][1][1] & 0x01  # ACC_Distance_Inc (byte 1 bit 0)
    CS.acc_follow_bars = 2
    sent = self._step(CC, CS, ACC_BUTTON_STEP + 1)
    assert len(sent) == 1 and sent[0][1][1] & 0x01
    CS.acc_follow_bars = 3
    assert self._step(CC, CS, 100) == []  # target reached

  def test_auto_follow_wrong_direction_inhibits_distance_only(self):
    CC, CS = make_cc(target_kph=113.0, bars=3), make_cs(set_speed_kph=113.0, bars=2)
    self._step(CC, CS, ACC_BUTTON_STEP + 1)
    CS.acc_follow_bars = 1  # Distance_Inc reduced bars: mapping assumption wrong -> stop Auto Follow
    self._step(CC, CS, 1)
    assert self.cc.jve_inhibited["dist"] and not self.cc.jve_inhibited["speed"]
    assert self._step(CC, CS, 100) == []
    # speed control still works
    CC.actuators.speed = 72.0 * CV.KPH_TO_MS
    sent = self._step(CC, CS, ACC_BUTTON_STEP + 1)
    assert len(sent) == 1 and sent[0][1][0] & 0x08

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
      packer.make_can_msg("ACC_HUD", 0, {"ACC_SET_SPEED_KMH": 113, "DISTANCE_SETTING": 2}),
      packer.make_can_msg("CRUISE_BUTTONS", 0, {"ACC_Accel": 1, "COUNTER": 7}),
      packer.make_can_msg("BSM_LEFT", 0, {"LEFT_DETECTED": 1}),
      packer.make_can_msg("BSM_RIGHT", 0, {"RIGHT_DETECTED": 0}),
    ]
    parsers[Bus.pt].update([(10_000_000, msgs)])
    out = cs.update(parsers)
    assert cs.button_counter == 7
    assert cs.acc_follow_bars == 3
    assert cs.cruise_button_pressed
    assert abs(out.vEgoCluster - 31.5) < 0.05
    assert out.cruiseState.enabled and abs(out.cruiseState.speed - 113 * CV.KPH_TO_MS) < 1e-3
    assert out.leftBlindspot and not out.rightBlindspot
    assert len(out.buttonEvents) == 1 and out.buttonEvents[0].type == structs.CarState.ButtonEvent.Type.accelCruise
