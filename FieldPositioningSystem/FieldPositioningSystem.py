import pymunk
import pygame
from pymunk import pygame_util
from pygame.color import *
import tensorflow as tf
from pygame.key import *
from pygame.locals import *
from Bot import Bot
import random
import numpy as np


mode = "SIM"#change to DRAW to draw, SIM to train
restore = False
batchSize = 32
numTimeSteps = 15 #how many previous timesteps are considered in rnn
numEpisodes = 100000
ckptEps = 1000 # save session every x episodes


scale = 20
bot = Bot(30,20, scale)
pygame.init()
screen = pygame.display.set_mode((1200, 600))
draw_options = pymunk.pygame_util.DrawOptions(screen)
clock = pygame.time.Clock()
running = True
NUM_STEPS = 100 # num steps per second
space = pymunk.Space()
space.add(bot.body, bot.shape)
gameTime = 0
forwardFlag = False
backFlag = False
rightFlag = False
leftFlag = False
inputList = []
truthList = []


steps = 0
actionSteps = 1 #take action every 6 steps
staticBody = pymunk.Body(0,0,pymunk.Body.STATIC)
field = [pymunk.Segment(staticBody,(0,0),(1200,0),1),
         pymunk.Segment(staticBody, (1200,0), (1200, 600),1),
         pymunk.Segment(staticBody, (1200,600), (0,600),1),
         pymunk.Segment(staticBody, (0,600), (0,0), 1)]
for line in field:
    space.add(line)
endTime = 7200 # num secs to run for
while(running and gameTime < endTime):
    for event in pygame.event.get():
        # debug input
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN and event.key == K_ESCAPE:
            running = False
        elif event.type == KEYDOWN and event.key == K_w:
            forwardFlag = True
        elif event.type == KEYUP and event.key == K_w:
            forwardFlag = False
        elif event.type == KEYDOWN and event.key == K_s:
            backFlag = True
        elif event.type == KEYUP and event.key == K_s:
            backFlag = False
        elif event.type == KEYDOWN and event.key == K_d:
            rightFlag = True
        elif event.type == KEYUP and event.key == K_d:
            rightFlag = False
        elif event.type == KEYDOWN and event.key == K_a:
            leftFlag = True
        elif event.type == KEYUP and event.key == K_a:
            leftFlag = False
        elif event.type == KEYDOWN and event.key == K_g:
            bot.Brake()
    if forwardFlag:
        bot.Forward(1)
    if rightFlag:
        bot.TurnRight(1)
    if leftFlag:
        bot.TurnLeft(1)
    if backFlag:
        bot.Backward(1)
    
    ## Update physics, move forward 1/NUM_STEPS second
    dt = 1.0/NUM_STEPS
    for x in range(1):
        bot.ControlVelUpdate(dt,gameTime)
        space.step(dt)
        gameTime += dt
        steps += 1
    if steps%actionSteps==0:# every actionstep
        inputList.append(bot.inputList[-numTimeSteps:])
        inputList[-1].extend([([0]*6)]*(max(numTimeSteps-len(bot.inputList),0)))
        pos = bot.body.position
        truthList.append([pos.x/scale, pos.y/scale])
    if int(gameTime - dt) < int(gameTime):# every second
        c = random.choice(range(0,4))
        if c == 0:
            forwardFlag = not forwardFlag
        elif c==1:
            backFlag = not backFlag
        elif c==2:
            rightFlag = not rightFlag
        elif c==3:
            leftFlag = not leftFlag
    if mode =="DRAW":
        ### Clear screen
        screen.fill(THECOLORS["white"])
        ### Draw stuff
        space.debug_draw(draw_options)
        pygame.display.flip()
        clock.tick()
        pygame.display.set_caption("fps: " + str(clock.get_fps()) + str(bot.body.position))


def model_fn(features):# featuredims (batch_size, num_timesteps, data_size)
    splitFeatures = tf.unstack(features, axis = 1)
    cell = tf.nn.rnn_cell.BasicRNNCell(100, tf.nn.relu)
    hidden = tf.nn.static_rnn(cell, splitFeatures,dtype= tf.double)
    hidden2 = tf.layers.dense(hidden[-1], 200)
    output = tf.layers.dense(hidden2, 2)#regression for coordinates
    return output

labels = tf.placeholder(tf.double)
inputs = tf.placeholder(tf.double, [None, numTimeSteps, 6])
dataSet = tf.data.Dataset.from_tensor_slices({"inputs" : inputs,
                                             "labels" : labels})
dataSet = dataSet.shuffle(100000)
dataSet = dataSet.batch(32)
iterator = tf.data.Iterator.from_structure(dataSet.output_types,dataSet.output_shapes)
next_element = iterator.get_next()
training_initializer = iterator.make_initializer(dataSet)
outputs = model_fn(next_element["inputs"])
loss = tf.losses.absolute_difference(next_element["labels"], outputs)
optimizer = tf.train.AdamOptimizer().minimize(loss)
sess = tf.Session()

init_op = tf.global_variables_initializer()
sess.run(init_op)
SAVE_PATH = "/FPSCheckpoints/model.ckpt"
saver = tf.train.Saver()
if restore:
    saver.restore(sess,SAVE_PATH)

sess.run(training_initializer,feed_dict = {inputs: inputList,
                                           labels: truthList})
for i in range(numEpisodes):
    try:
        l = sess.run(loss)
        sess.run(optimizer)
        if i % 25 == 0:
            print("Epoch: {}, loss: {}".format(i,str(np.average(l))))
        if i%ckptEps == 0:
            saver.save(sess, SAVE_PATH)
    except tf.errors.OutOfRangeError as o:
        break
sess.close()
            