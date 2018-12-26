import pymunk
from pymunk import Vec2d
import random
import math
class Bot(object):
    """description of class"""
    def __init__(self, initX, initY, scale = 1, mass = 120, width = 2.5, length = 2.5):
        self.width = width * scale
        self.length = length * scale
        self.initX = initX * scale
        self.initY = initY * scale
        points = [(-self.width/2,-self.length/2),(-self.width/2, self.length/2), (self.width/2, self.length/2), (self.width/2, -self.length/2) ]
        self.mass = mass
        self.body = pymunk.Body(self.mass,pymunk.moment_for_poly(self.mass, points))
        self.body.position = Vec2d(self.initX,self.initY)
        self.shape = pymunk.Poly(self.body, points)
        self.force = Vec2d(0,0)
        self.torque = 0
        self.inputs = []
        self.inputList = []
        
        self.scale = scale
        self.maxForce = 396
        self.maxTorque = 150
        self.friction = self.maxForce * 0.2
        self.angFriction = self.maxTorque * 0.1

    def KillLateralMvmt(self):
        self.body.velocity -= self.body.velocity.projection(Vec2d(math.cos(self.body.angle), math.sin(self.body.angle)).perpendicular())
    
    def TurnLeft(self, proportion):
        self.torque += proportion * self.maxTorque

    def TurnRight(self, proportion):
        self.torque -= proportion * self.maxTorque

    def Forward(self, proportion):
        direction = Vec2d(1,1)
        direction.angle = self.body.angle
        direction = direction.normalized()
        self.force += proportion * self.maxForce * direction 

    def Backward(self, proportion):
        direction = Vec2d(1,1)
        direction.angle = self.body.angle
        direction = -direction.normalized()
        self.force += proportion * self.maxForce * direction 

    def ControlVelUpdate(self, dt, time):
        self.body.velocity/=self.scale
        self.body.angular_velocity/=self.scale
        self.force += self.body.force
        accel = self.force / self.body.mass
        self.body.velocity += accel * dt
        fricDelta = dt * self.friction * self.body.velocity.normalized() / self.body.mass
        if (self.body.velocity.length > fricDelta.length ):
            self.body.velocity -= fricDelta
        else:    
            self.body.velocity = (0,0)
        angAccel = self.torque / self.body.moment
        self.body.angular_velocity += angAccel * dt
        angFricDelta = dt * self.angFriction /(self.body.moment)
        angFricDelta *= 2 * int(self.body.angular_velocity >= 0) - 1 # get sign
        if (self.body.angular_velocity - angFricDelta >= 0) == (self.body.angular_velocity >= 0):
            self.body.angular_velocity -= angFricDelta
        else: 
            self.body.angular_velocity = 0
        self.KillLateralMvmt()
        self.body.velocity *= self.scale
        self.body.angular_velocity *= self.scale
        self.force = Vec2d(0,0)
        self.torque = 0
        self.inputs = [random.normalvariate(accel.x, 1.6),random.normalvariate(accel.y, 1.6), self.body.angle, self.initX, self.initY, time]#inputs passed to neural network
        self.inputList.append(self.inputs)




