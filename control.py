import simulation

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0
    
    def compute(self, position, setpoint, dt):
        error = setpoint - position
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output
    
    def reset(self):
        self.integral = 0
        self.prev_error = 0

class TrajectoryFollower:
    def __init__(self, arm_sim, Kp, Ki, Kd):
        self.controller1 = PIDController(Kp, Ki, Kd)
        self.controller2 = PIDController(Kp, Ki, Kd)
        self.arm_sim = arm_sim

    def follow_trajectory(self, trajectory, time, dt = 0.02):
        """Given a list of (t1_setpoint, t2_setpoint) pairs and a time step dt,
        compute and apply motor voltages to follow the trajectory."""
        for t1_setpoint, t2_setpoint in trajectory:
            t1 = self.arm_sim.upperArm.position
            t2 = self.arm_sim.forearm.position
            voltage1 = self.controller1.compute(t1, t1_setpoint, dt)
            voltage2 = self.controller2.compute(t2, t2_setpoint, dt)
            self.arm_sim.upperArm.setVoltage(voltage1)
            self.arm_sim.forearm.setVoltage(voltage2)
            self.arm_sim.update()