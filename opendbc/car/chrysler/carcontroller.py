from opendbc.can import CANPacker
from opendbc.car import Bus, DT_CTRL
from opendbc.car.carlog import carlog
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.lateral import apply_meas_steer_torque_limits
from opendbc.car.chrysler import chryslercan
from opendbc.car.chrysler.values import CUSW_CARS, RAM_CARS, CarControllerParams, ChryslerFlags
from opendbc.car.interfaces import CarControllerBase

# jvePilot hybrid ACC button control parameters
ACC_BUTTON_HOLD_FRAMES = 7    # 7 SCCM-synced Tick-A pulses (~120-140 ms across 7 distinct 50 Hz SCCM counters)
ACC_RESUME_HOLD_FRAMES = ACC_BUTTON_HOLD_FRAMES  # multi-frame SCCM-synced hold burst for standstill auto-resume
ACC_BUTTON_STEP = 18          # 180 ms release cooldown after the last pulse before starting the next virtual press
ACC_FEEDBACK_TIMEOUT = 51     # 510 ms (odd) for a press to show up in the stock ACC set speed / distance before it counts as unconfirmed
ACC_MAX_UNCONFIRMED = 3       # consecutive unconfirmed presses before that button type is inhibited
ACC_FLOOR_UNCONFIRMED = 3     # consecutive unconfirmed decel/accel presses before learning a trim's min/max set speed
ACC_INHIBIT_BACKOFF = 250     # 2.5 s cooldown before retrying a timeout-inhibited speed direction once buttons are confirmed working
ACC_MAX_BACKOFF_RETRIES = 3   # max consecutive backoff retries without any confirmation before staying fail-closed until next engagement
# Set-speed change per press. Starts at 1 mph and is learned from CAN feedback (1 km/h on metric cars; 1 mph shows up
# as 1-2 km/h on the integer km/h ACC_HUD signal). Pressing only when the error exceeds half a step plus a margin
# moves to the closest reachable set speed and can never oscillate.
DEFAULT_SET_SPEED_STEP = CV.MPH_TO_MS
SET_SPEED_MARGIN = 0.05  # m/s
MIN_ACC_SET_SPEED = 32.0 * CV.KPH_TO_MS  # ~20 mph / 32 km/h FCA minimum selectable ACC set speed
# An unanswered press near these set speeds is treated as the car's real min/max ACC set speed (learned per
# engagement) instead of a fault, since some CUSW trims have a higher minimum set speed (e.g. ~43 mph / 69 km/h)
LEARN_FLOOR_BELOW = 75.0 * CV.KPH_TO_MS
LEARN_CEILING_ABOVE = 120.0 * CV.KPH_TO_MS


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.apply_torque_last = 0

    self.hud_count = 0
    self.last_lkas_falling_edge = 0
    self.lkas_control_bit_prev = False
    self.lkas_control_bit_cmd_prev = False
    self.last_button_frame = 0
    self.last_resume_frame = -ACC_BUTTON_STEP
    self.jve_set_speed_step = None  # learned stock ACC set-speed change per press (m/s), kept across engagements
    self.jve_speed_confirmed_ever = False
    self.jve_min_observed_set_speed = None
    self._jve_reset()

    self.packer = CANPacker(dbc_names[Bus.pt])
    self.params = CarControllerParams(CP)

  def _jve_reset(self):
    # pending press awaiting CAN feedback: (kind "speed"|"dist", value at press, direction +1/-1, frame)
    self.jve_pending = None
    self.jve_hold_button = None
    self.jve_hold_frames = 0
    self.jve_resume_hold_frames = 0
    self.jve_unconfirmed = {"speed": 0, "dist": 0}
    self.jve_inhibited = {"speed": False, "dist": False}
    self.jve_inhibited_dir = {"speed": 0, "dist": 0}
    self.jve_inhibited_frame = {"speed": 0, "dist": 0}
    self.jve_backoff_retries = {"speed": 0, "dist": 0}
    self.jve_inhibited_target = None
    self.jve_last_v_target = None
    self.jve_speed_floor = None    # learned stock ACC min set speed (m/s) for this engagement
    self.jve_speed_ceiling = None  # learned stock ACC max set speed (m/s) for this engagement
    self.jve_floor_target = None
    self.jve_floor_vego = None
    self.jve_last_speed_dir = 0
    self.jve_last_speed_frame = -1000

  def _jve_inhibit(self, kind, reason, direction=0):
    if not self.jve_inhibited[kind]:
      carlog.warning(f"jvePilot: disabling {kind} button control until next engagement ({reason})")
    self.jve_inhibited[kind] = True
    self.jve_inhibited_dir[kind] = direction
    self.jve_inhibited_frame[kind] = self.frame
    if kind == "speed" and direction != 0:
      self.jve_inhibited_target = self.jve_last_v_target

  def _jve_check_feedback(self, cs_out, CS, driver_button_active):
    """Confirm the last virtual press moved stock ACC the expected way; fail closed if it didn't."""
    if self.jve_pending is None:
      return
    kind, ref, direction, frame = self.jve_pending
    if driver_button_active:
      # driver is using the buttons, so any change can't be attributed to our press
      self.jve_pending = None
      self.jve_hold_button = None
      self.jve_hold_frames = 0
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
        self.jve_backoff_retries[kind] = 0
        if kind == "speed":
          self.jve_speed_confirmed_ever = True
          self.jve_set_speed_step = min(max(self.jve_set_speed_step or 0.0, abs(delta)), 2.0 * CV.KPH_TO_MS)
      else:
        self.jve_hold_button = None
        self.jve_hold_frames = 0
        self._jve_inhibit(kind, f"stock ACC moved opposite to the press (delta={delta})", direction=0)
    elif self.frame - frame >= ACC_FEEDBACK_TIMEOUT:
      self.jve_pending = None
      self.jve_hold_button = None
      self.jve_hold_frames = 0
      if kind == "speed":
        at_lowest_seen = (self.jve_min_observed_set_speed is None or ref <= self.jve_min_observed_set_speed + 0.1)
        if direction < 0 and ref < LEARN_FLOOR_BELOW and at_lowest_seen and self.jve_unconfirmed[kind] + 1 >= ACC_FLOOR_UNCONFIRMED:
          self.jve_speed_floor = ref
          self.jve_floor_target = self.jve_last_v_target
          self.jve_floor_vego = cs_out.vEgo
          self.jve_unconfirmed[kind] = 0
          return
        elif direction > 0 and ref > LEARN_CEILING_ABOVE and self.jve_unconfirmed[kind] + 1 >= ACC_FLOOR_UNCONFIRMED:
          self.jve_speed_ceiling = ref
          self.jve_unconfirmed[kind] = 0
          return
      self.jve_unconfirmed[kind] += 1
      if self.jve_unconfirmed[kind] >= ACC_MAX_UNCONFIRMED:
        self._jve_inhibit(kind, f"{ACC_MAX_UNCONFIRMED} presses with no stock ACC response", direction=direction)

  def _jve_select_button(self, CC, cs_out, CS):
    """Returns (kind, direction, button_kwargs) for the next virtual press, or None."""
    v_acc = cs_out.cruiseState.speed
    v_target = max(CC.actuators.speed, MIN_ACC_SET_SPEED)
    if CC.hudControl.setSpeed > 0:
      v_target = min(v_target, CC.hudControl.setSpeed)
    self.jve_last_v_target = v_target

    if self.jve_speed_floor is not None and (
        abs(v_acc - self.jve_speed_floor) > 0.1 or
        v_target >= self.jve_speed_floor - 0.1 or
        (self.jve_floor_target is not None and abs(v_target - self.jve_floor_target) >= 2.0 * DEFAULT_SET_SPEED_STEP) or
        (self.jve_floor_vego is not None and cs_out.vEgo < self.jve_floor_vego - 1.0)):
      self.jve_speed_floor = None
      self.jve_floor_target = None
      self.jve_floor_vego = None

    at_ceiling = self.jve_speed_ceiling is not None and v_acc >= self.jve_speed_ceiling - 0.1
    at_floor = v_acc <= MIN_ACC_SET_SPEED + 0.1 or (self.jve_speed_floor is not None and v_acc <= self.jve_speed_floor + 0.1)
    threshold = (self.jve_set_speed_step or DEFAULT_SET_SPEED_STEP) / 2 + SET_SPEED_MARGIN
    rev_margin = 0.20 if (self.frame - self.jve_last_speed_frame) < 150 else 0.0
    speed_dir = 0
    if v_target > v_acc + threshold + (rev_margin if self.jve_last_speed_dir < 0 else 0.0) and not at_ceiling:
      speed_dir = 1
    elif v_target < v_acc - threshold - (rev_margin if self.jve_last_speed_dir > 0 else 0.0) and not at_floor:
      speed_dir = -1
    if self.jve_inhibited["speed"] and self.jve_inhibited_dir["speed"] != 0:
      demand_reset = (speed_dir == 0 or
                      speed_dir == -self.jve_inhibited_dir["speed"] or
                      (self.jve_inhibited_target is not None and abs(v_target - self.jve_inhibited_target) >= 2.0 * DEFAULT_SET_SPEED_STEP))
      can_backoff_retry = (self.jve_speed_confirmed_ever and
                           self.jve_backoff_retries["speed"] < ACC_MAX_BACKOFF_RETRIES and
                           (self.frame - self.jve_inhibited_frame["speed"]) >= ACC_INHIBIT_BACKOFF)
      if demand_reset or can_backoff_retry:
        if not demand_reset and can_backoff_retry:
          self.jve_backoff_retries["speed"] += 1
        self.jve_inhibited["speed"] = False
        self.jve_inhibited_dir["speed"] = 0
        self.jve_unconfirmed["speed"] = 0
        self.jve_inhibited_target = None
    if not self.jve_inhibited["speed"] and speed_dir != 0:
      self.jve_last_speed_dir = speed_dir
      self.jve_last_speed_frame = self.frame
      return "speed", speed_dir, ({"accel": True} if speed_dir > 0 else {"decel": True})

    # Auto Follow: only act on a known stock ACC bar setting reported on CAN
    target_bars = int(getattr(CC.hudControl, "leadDistanceBars", 0))
    can_bars = int(getattr(CS, "acc_follow_bars", 0))
    if target_bars in (1, 2, 3, 4) and can_bars in (1, 2, 3, 4):
      dist_dir = 0 if target_bars == can_bars else (1 if target_bars > can_bars else -1)
      if self.jve_inhibited["dist"] and self.jve_inhibited_dir["dist"] != 0 and (
          dist_dir == 0 or dist_dir == -self.jve_inhibited_dir["dist"]):
        self.jve_inhibited["dist"] = False
        self.jve_inhibited_dir["dist"] = 0
        self.jve_unconfirmed["dist"] = 0
      if not self.jve_inhibited["dist"] and dist_dir != 0:
        return "dist", dist_dir, ({"distance_inc": True} if dist_dir > 0 else {"distance_dec": True})
    return None

  def update(self, CC, CS, now_nanos):
    can_sends = []
    cs_out = CS.out if hasattr(CS, "out") else CS
    driver_button_active = getattr(CS, "cruise_button_pressed", False) or len(cs_out.buttonEvents) > 0
    cruise_buttons_rx = getattr(CS, "cruise_buttons_rx", True)

    lkas_active = CC.latActive and self.lkas_control_bit_prev and cs_out.cruiseState.enabled and not cs_out.brakePressed

    # cruise buttons (cancel, standstill auto-resume, and jvePilot hybrid ACC speed control)
    das_bus = 2 if self.CP.carFingerprint in RAM_CARS else 0
    resume_guard_ok = (CC.cruiseControl.resume and not CC.cruiseControl.cancel and
                       not cs_out.brakePressed and not cs_out.gasPressed and
                       not cs_out.accFaulted and cs_out.canValid and not driver_button_active)
    if not resume_guard_ok:
      self.jve_resume_hold_frames = 0

    # ACC cancellation
    if CC.cruiseControl.cancel:
      if (self.frame - self.last_button_frame) * DT_CTRL >= 0.05:
        self.last_button_frame = self.frame
        can_sends.append(chryslercan.create_cruise_buttons(self.packer, CS.button_counter + 1, das_bus, cancel=True))

    # ACC resume from standstill (0 mph stop-and-go), guarded against pedal/fault/button conflicts, stale CAN, and Tick-B collisions
    elif resume_guard_ok:
      if self.jve_resume_hold_frames > 0:
        if cruise_buttons_rx:
          self.jve_resume_hold_frames -= 1
          self.last_button_frame = self.frame
          self.last_resume_frame = self.frame
          can_sends.append(chryslercan.create_cruise_buttons(self.packer, CS.button_counter + 1, das_bus, resume=True))
      elif (cruise_buttons_rx and (self.frame - self.last_button_frame) * DT_CTRL >= 0.05 and
            (self.frame - self.last_resume_frame) >= ACC_BUTTON_STEP):
        self.jve_resume_hold_frames = ACC_RESUME_HOLD_FRAMES - 1
        self.last_button_frame = self.frame
        self.last_resume_frame = self.frame
        can_sends.append(chryslercan.create_cruise_buttons(self.packer, CS.button_counter + 1, das_bus, resume=True))

    if driver_button_active and not CC.cruiseControl.cancel:
      self.last_button_frame = self.frame

    # jvePilot hybrid ACC set-speed and follow-distance modulation via virtual cruise button pulses
    if not CC.enabled or not cs_out.cruiseState.enabled:
      self._jve_reset()
    elif CC.actuators.speed > 0:
      if cs_out.cruiseState.speed >= MIN_ACC_SET_SPEED - 0.5:
        if self.jve_min_observed_set_speed is None or cs_out.cruiseState.speed < self.jve_min_observed_set_speed:
          self.jve_min_observed_set_speed = cs_out.cruiseState.speed
      self._jve_check_feedback(cs_out, CS, driver_button_active)
      guard_ok = (not CC.cruiseControl.cancel and not CC.cruiseControl.resume and
                  not cs_out.cruiseState.standstill and not cs_out.gasPressed and not cs_out.brakePressed and
                  not cs_out.accFaulted and cs_out.canValid and not driver_button_active)
      if not guard_ok:
        self.jve_hold_button = None
        self.jve_hold_frames = 0
      elif self.jve_hold_frames > 0 and self.jve_hold_button is not None:
        if cruise_buttons_rx:
          self.jve_hold_frames -= 1
          self.last_button_frame = self.frame
          can_sends.append(chryslercan.create_cruise_buttons(self.packer, CS.button_counter + 1, das_bus, **self.jve_hold_button))
      elif cruise_buttons_rx and self.jve_pending is None and (self.frame - self.last_button_frame) >= ACC_BUTTON_STEP:
        selection = self._jve_select_button(CC, cs_out, CS)
        if selection is not None:
          kind, direction, button = selection
          ref = cs_out.cruiseState.speed if kind == "speed" else int(getattr(CS, "acc_follow_bars", 0))
          self.jve_pending = (kind, ref, direction, self.frame)
          self.jve_hold_button = button
          self.jve_hold_frames = ACC_BUTTON_HOLD_FRAMES - 1
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
      if not lkas_control_bit_cmd and self.lkas_control_bit_cmd_prev:
        self.last_lkas_falling_edge = self.frame
      self.lkas_control_bit_cmd_prev = lkas_control_bit_cmd
      self.apply_torque_last = apply_torque

      can_sends.append(chryslercan.create_lkas_command(self.packer, self.CP, int(apply_torque), lkas_control_bit_cmd))

    self.frame += 1

    new_actuators = CC.actuators.as_builder()
    new_actuators.torque = self.apply_torque_last / self.params.STEER_MAX
    new_actuators.torqueOutputCan = self.apply_torque_last

    return new_actuators, can_sends
