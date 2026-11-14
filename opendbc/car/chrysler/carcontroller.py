from opendbc.can import CANPacker
from opendbc.car import Bus, DT_CTRL
from opendbc.car.carlog import carlog
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.lateral import apply_meas_steer_torque_limits
from opendbc.car.chrysler import chryslercan
from opendbc.car.chrysler.values import CUSW_CARS, RAM_CARS, CarControllerParams, ChryslerFlags
from opendbc.car.interfaces import CarControllerBase

# jvePilot hybrid ACC button control parameters
ACC_BUTTON_STEP = 8           # at least 80 ms between virtual presses (single frame per press, like upstream cancel/resume)
ACC_FEEDBACK_TIMEOUT = 50     # 500 ms for a press to show up in the stock ACC set speed / distance before it counts as unconfirmed
ACC_MAX_UNCONFIRMED = 3       # consecutive unconfirmed presses before that button type is disabled until the next engagement
# Set-speed change per press. Starts at 1 mph and is learned from CAN feedback (1 km/h on metric cars; 1 mph shows up
# as 1-2 km/h on the integer km/h ACC_HUD signal). Pressing only when the error exceeds half a step plus a margin
# moves to the closest reachable set speed and can never oscillate.
DEFAULT_SET_SPEED_STEP = CV.MPH_TO_MS
SET_SPEED_MARGIN = 0.05  # m/s
MIN_ACC_SET_SPEED = 32.0 * CV.KPH_TO_MS  # ~20 mph / 32 km/h FCA minimum selectable ACC set speed
# An unanswered press near these set speeds is treated as the car's real min/max ACC set speed (learned per
# engagement) instead of a fault, since the exact CUSW limits are not verified yet
LEARN_FLOOR_BELOW = 50.0 * CV.KPH_TO_MS
LEARN_CEILING_ABOVE = 120.0 * CV.KPH_TO_MS


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.apply_torque_last = 0

    self.hud_count = 0
    self.last_lkas_falling_edge = 0
    self.lkas_control_bit_prev = False
    self.last_button_frame = 0
    self.jve_set_speed_step = None  # learned stock ACC set-speed change per press (m/s), kept across engagements
    self._jve_reset()

    self.packer = CANPacker(dbc_names[Bus.pt])
    self.params = CarControllerParams(CP)

  def _jve_reset(self):
    # pending press awaiting CAN feedback: (kind "speed"|"dist", value at press, direction +1/-1, frame)
    self.jve_pending = None
    self.jve_unconfirmed = {"speed": 0, "dist": 0}
    self.jve_inhibited = {"speed": False, "dist": False}
    self.jve_speed_floor = None    # learned stock ACC min set speed (m/s) for this engagement
    self.jve_speed_ceiling = None  # learned stock ACC max set speed (m/s) for this engagement

  def _jve_inhibit(self, kind, reason):
    if not self.jve_inhibited[kind]:
      carlog.warning(f"jvePilot: disabling {kind} button control until next engagement ({reason})")
    self.jve_inhibited[kind] = True

  def _jve_check_feedback(self, cs_out, CS, driver_button_active):
    """Confirm the last virtual press moved stock ACC the expected way; fail closed if it didn't."""
    if self.jve_pending is None:
      return
    kind, ref, direction, frame = self.jve_pending
    if driver_button_active:
      # driver is using the buttons, so any change can't be attributed to our press
      self.jve_pending = None
      return

    if kind == "speed":
      delta = cs_out.cruiseState.speed - ref
      changed = abs(delta) > 0.1
    else:
      delta = int(getattr(CS, "acc_follow_bars", 0)) - ref
      changed = delta != 0

    if changed:
      self.jve_pending = None
      if delta * direction > 0:
        self.jve_unconfirmed[kind] = 0
        if kind == "speed":
          self.jve_set_speed_step = max(self.jve_set_speed_step or 0.0, abs(delta))
      else:
        self._jve_inhibit(kind, f"stock ACC moved opposite to the press (delta={delta})")
    elif self.frame - frame >= ACC_FEEDBACK_TIMEOUT:
      self.jve_pending = None
      if kind == "speed":
        if direction < 0 and ref < LEARN_FLOOR_BELOW:
          self.jve_speed_floor = ref
        elif direction > 0 and ref > LEARN_CEILING_ABOVE:
          self.jve_speed_ceiling = ref
      self.jve_unconfirmed[kind] += 1
      if self.jve_unconfirmed[kind] >= ACC_MAX_UNCONFIRMED:
        self._jve_inhibit(kind, f"{ACC_MAX_UNCONFIRMED} presses with no stock ACC response")

  def _jve_select_button(self, CC, cs_out, CS):
    """Returns (kind, direction, button_kwargs) for the next virtual press, or None."""
    v_acc = cs_out.cruiseState.speed
    v_target = max(CC.actuators.speed, MIN_ACC_SET_SPEED)
    if CC.hudControl.setSpeed > 0:
      v_target = min(v_target, CC.hudControl.setSpeed)

    if not self.jve_inhibited["speed"]:
      at_ceiling = self.jve_speed_ceiling is not None and v_acc >= self.jve_speed_ceiling - 0.1
      at_floor = v_acc <= MIN_ACC_SET_SPEED + 0.1 or (self.jve_speed_floor is not None and v_acc <= self.jve_speed_floor + 0.1)
      threshold = (self.jve_set_speed_step or DEFAULT_SET_SPEED_STEP) / 2 + SET_SPEED_MARGIN
      if v_target > v_acc + threshold and not at_ceiling:
        return "speed", 1, {"accel": True}
      if v_target < v_acc - threshold and not at_floor:
        return "speed", -1, {"decel": True}

    # Auto Follow: only act on a known stock ACC bar setting reported on CAN
    target_bars = int(getattr(CC.hudControl, "leadDistanceBars", 0))
    can_bars = int(getattr(CS, "acc_follow_bars", 0))
    if (not self.jve_inhibited["dist"] and target_bars in (1, 2, 3, 4) and can_bars in (1, 2, 3, 4) and
        target_bars != can_bars):
      if target_bars > can_bars:
        return "dist", 1, {"distance_inc": True}
      return "dist", -1, {"distance_dec": True}
    return None

  def update(self, CC, CS, now_nanos):
    can_sends = []
    cs_out = CS.out if hasattr(CS, "out") else CS
    driver_button_active = getattr(CS, "cruise_button_pressed", False) or len(cs_out.buttonEvents) > 0

    lkas_active = CC.latActive and self.lkas_control_bit_prev

    # cruise buttons (cancel, standstill auto-resume, and jvePilot hybrid ACC speed control)
    das_bus = 2 if self.CP.carFingerprint in RAM_CARS else 0
    if (self.frame - self.last_button_frame) * DT_CTRL > 0.05:
      # ACC cancellation
      if CC.cruiseControl.cancel:
        self.last_button_frame = self.frame
        can_sends.append(chryslercan.create_cruise_buttons(self.packer, CS.button_counter + 1, das_bus, cancel=True))

      # ACC resume from standstill (0 mph stop-and-go), guarded against pedal/fault/button conflicts and stale CAN
      elif (CC.cruiseControl.resume and not cs_out.brakePressed and not cs_out.gasPressed and
            not cs_out.accFaulted and cs_out.canValid and not driver_button_active):
        self.last_button_frame = self.frame
        can_sends.append(chryslercan.create_cruise_buttons(self.packer, CS.button_counter + 1, das_bus, resume=True))

    # jvePilot hybrid ACC set-speed and follow-distance modulation via virtual cruise button pulses
    if not CC.enabled or not cs_out.cruiseState.enabled:
      self._jve_reset()
    elif CC.actuators.speed > 0:
      self._jve_check_feedback(cs_out, CS, driver_button_active)
      can_press = (not CC.cruiseControl.cancel and not CC.cruiseControl.resume and
                   not cs_out.cruiseState.standstill and not cs_out.gasPressed and not cs_out.brakePressed and
                   not cs_out.accFaulted and cs_out.canValid and not driver_button_active and
                   self.jve_pending is None and (self.frame - self.last_button_frame) >= ACC_BUTTON_STEP)
      if can_press:
        selection = self._jve_select_button(CC, cs_out, CS)
        if selection is not None:
          kind, direction, button = selection
          ref = cs_out.cruiseState.speed if kind == "speed" else int(getattr(CS, "acc_follow_bars", 0))
          self.jve_pending = (kind, ref, direction, self.frame)
          self.last_button_frame = self.frame
          can_sends.append(chryslercan.create_cruise_buttons(self.packer, CS.button_counter + 1, das_bus, **button))

    # HUD alerts
    if self.frame % 25 == 0:
      if CS.lkas_car_model != -1:
        can_sends.append(chryslercan.create_lkas_hud(self.packer, self.CP, lkas_active, CC.hudControl.visualAlert,
                                                     self.hud_count, CS.lkas_car_model, CS.auto_high_beam))
        self.hud_count += 1

    # steering
    if self.frame % self.params.STEER_STEP == 0:

      # TODO: can we make this more sane? why is it different for all the cars?
      lkas_control_bit = self.lkas_control_bit_prev
      if CS.out.vEgo > self.CP.minSteerSpeed:
        lkas_control_bit = True
      elif self.CP.flags & ChryslerFlags.HIGHER_MIN_STEERING_SPEED:
        if CS.out.vEgo < (self.CP.minSteerSpeed - 3.0):
          lkas_control_bit = False
      elif self.CP.carFingerprint in RAM_CARS:
        if CS.out.vEgo < (self.CP.minSteerSpeed - 0.5):
          lkas_control_bit = False
      elif self.CP.carFingerprint in CUSW_CARS:
        if CS.out.vEgo < (self.CP.minSteerSpeed - 2.0):
          lkas_control_bit = False

      # EPS faults if LKAS re-enables too quickly
      lkas_control_bit = lkas_control_bit and (self.frame - self.last_lkas_falling_edge > 200)

      if not lkas_control_bit and self.lkas_control_bit_prev:
        self.last_lkas_falling_edge = self.frame
      self.lkas_control_bit_prev = lkas_control_bit

      # steer torque
      new_torque = int(round(CC.actuators.torque * self.params.STEER_MAX))
      apply_torque = apply_meas_steer_torque_limits(new_torque, self.apply_torque_last, CS.out.steeringTorqueEps, self.params)

      lkas_control_bit_cmd = lkas_control_bit
      if not lkas_active or not lkas_control_bit:
        if self.CP.carFingerprint in CUSW_CARS and self.apply_torque_last != 0:
          # CUSW faults if torque is slammed to zero, ramp down at MAX_RATE_DOWN
          if self.apply_torque_last > 0:
            apply_torque = max(self.apply_torque_last - self.params.STEER_DELTA_DOWN, 0)
          else:
            apply_torque = min(self.apply_torque_last + self.params.STEER_DELTA_DOWN, 0)
          # keep LKAS active on CAN until torque reaches zero
          if apply_torque != 0:
            lkas_control_bit_cmd = True
        else:
          apply_torque = 0
      self.apply_torque_last = apply_torque

      can_sends.append(chryslercan.create_lkas_command(self.packer, self.CP, int(apply_torque), lkas_control_bit_cmd))

    self.frame += 1

    new_actuators = CC.actuators.as_builder()
    new_actuators.torque = self.apply_torque_last / self.params.STEER_MAX
    new_actuators.torqueOutputCan = self.apply_torque_last

    return new_actuators, can_sends
