from opendbc.can import CANPacker
from opendbc.car import Bus, DT_CTRL
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.lateral import apply_meas_steer_torque_limits
from opendbc.car.chrysler import chryslercan
from opendbc.car.chrysler.values import CUSW_CARS, RAM_CARS, CarControllerParams, ChryslerFlags
from opendbc.car.interfaces import CarControllerBase

# jvePilot hybrid ACC button control parameters
ACC_BUTTON_STEP = 8          # 80 ms cycle (12.5 Hz max button tap rate)
ACC_BUTTON_HOLD_FRAMES = 2   # 20 ms button hold followed by 60 ms release so SCCM sends 0-frames
ACC_SPEED_DEADBAND = 0.35    # m/s (~0.78 mph) deadband to prevent integer km/h/mph rounding oscillation
MIN_ACC_SET_SPEED = 32.0 * CV.KPH_TO_MS  # ~20 mph / 32 km/h FCA minimum selectable ACC set speed


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.apply_torque_last = 0

    self.hud_count = 0
    self.last_lkas_falling_edge = 0
    self.lkas_control_bit_prev = False
    self.last_button_frame = 0
    self.last_acc_button = None
    self.auto_follow_bars = 0

    self.packer = CANPacker(dbc_names[Bus.pt])
    self.params = CarControllerParams(CP)

  def update(self, CC, CS, now_nanos):
    can_sends = []

    lkas_active = CC.latActive and self.lkas_control_bit_prev

    # cruise buttons (cancel, standstill auto-resume, and jvePilot hybrid ACC speed control)
    das_bus = 2 if self.CP.carFingerprint in RAM_CARS else 0
    if (self.frame - self.last_button_frame) * DT_CTRL > 0.05:
      # ACC cancellation
      if CC.cruiseControl.cancel:
        self.last_button_frame = self.frame
        self.last_acc_button = None
        can_sends.append(chryslercan.create_cruise_buttons(self.packer, CS.button_counter + 1, das_bus, cancel=True))

      # ACC resume from standstill (0 mph stop-and-go)
      elif CC.cruiseControl.resume:
        self.last_button_frame = self.frame
        self.last_acc_button = None
        can_sends.append(chryslercan.create_cruise_buttons(self.packer, CS.button_counter + 1, das_bus, resume=True))

    # jvePilot hybrid ACC set-speed modulation via virtual ACC_Accel / ACC_Decel button presses
    if (not CC.cruiseControl.cancel and not CC.cruiseControl.resume and
        CC.enabled and CS.cruiseState.enabled and not CS.cruiseState.standstill and
        not CS.gasPressed and len(CS.buttonEvents) == 0 and CC.actuators.speed > 0):
      v_acc = CS.cruiseState.speed
      v_target = max(CC.actuators.speed, MIN_ACC_SET_SPEED)
      if CC.hudControl.setSpeed > 0:
        v_target = min(v_target, CC.hudControl.setSpeed)

      frames_since_btn = self.frame - self.last_button_frame
      if 0 < frames_since_btn < ACC_BUTTON_HOLD_FRAMES and self.last_acc_button in ("accel", "decel", "dist_inc", "dist_dec"):
        can_sends.append(chryslercan.create_cruise_buttons(
          self.packer, CS.button_counter + 1, das_bus,
          accel=(self.last_acc_button == "accel"),
          decel=(self.last_acc_button == "decel"),
          distance_inc=(self.last_acc_button == "dist_inc"),
          distance_dec=(self.last_acc_button == "dist_dec"),
        ))
      elif frames_since_btn >= ACC_BUTTON_STEP:
        target_bars = int(getattr(CC.hudControl, "leadDistanceBars", 0))
        if target_bars in (1, 2, 3, 4) and self.auto_follow_bars == 0:
          self.auto_follow_bars = target_bars

        if v_target > v_acc + ACC_SPEED_DEADBAND:
          self.last_button_frame = self.frame
          self.last_acc_button = "accel"
          can_sends.append(chryslercan.create_cruise_buttons(self.packer, CS.button_counter + 1, das_bus, accel=True))
        elif v_target < v_acc - ACC_SPEED_DEADBAND and v_acc > MIN_ACC_SET_SPEED + 0.1:
          self.last_button_frame = self.frame
          self.last_acc_button = "decel"
          can_sends.append(chryslercan.create_cruise_buttons(self.packer, CS.button_counter + 1, das_bus, decel=True))
        elif target_bars in (1, 2, 3, 4) and target_bars > self.auto_follow_bars:
          self.last_button_frame = self.frame
          self.last_acc_button = "dist_inc"
          self.auto_follow_bars += 1
          can_sends.append(chryslercan.create_cruise_buttons(self.packer, CS.button_counter + 1, das_bus, distance_inc=True))
        elif target_bars in (1, 2, 3, 4) and target_bars < self.auto_follow_bars:
          self.last_button_frame = self.frame
          self.last_acc_button = "dist_dec"
          self.auto_follow_bars -= 1
          can_sends.append(chryslercan.create_cruise_buttons(self.packer, CS.button_counter + 1, das_bus, distance_dec=True))
        else:
          self.last_acc_button = None
    else:
      self.last_acc_button = None
      if not CC.enabled or not CS.cruiseState.enabled:
        self.auto_follow_bars = 0

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
