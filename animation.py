#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      kol
#
# Created:     06.02.2020
# Copyright:   (c) kol 2020
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import sys
import cv2
from skimage import draw, segmentation
import numpy as np
import matplotlib as plt
from pathlib import Path
import random
import copy
import itertools
import time
import simpleaudio as sa

from img_utils import rgba_to_rgb, rgb_to_rgba, apply_patch, color_to_cv_color
from img_utils import increase_brightness

sys.path.append('..\\gbr')
from gr.utils import resize3

def reversable_iter(src):
    class __iter:
        def __init__(self, src, fun=itertools.cycle):
            self.__src = src
            self.__fun=fun
            self.__iter = self.__fun(self.__src)
            self.direct = True

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.__iter)

        def reverse(self):
            cur_val = next(self.__iter)
            self.__src = [x for x in reversed(self.__src)]
            self.__iter = self.__fun(self.__src)
            self.direct = not self.direct
            for v in self.__src:
                next(self.__iter)
                if v == cur_val:
                    next(self.__iter)
                    break

        def reverse_if(self, direct):
            if self.direct != direct:
                self.reverse()

    return __iter(src)

class Sprite:
    def __init__(self, scene):
        self.scene = scene
        self.body, self.mask = None, None
        self.x, self.y = 0, 0
        self.loc = None
        self.load()

    def load(self):
        pass

    def set_loc(self, loc):
        self.loc = loc

    def one_frame(self, frame, n_frame, elapsed_time):
        return frame

    def move(self, n_frame, elapsed_time):
        pass

    def flip(self):
        self.body = cv2.flip(self.body, 1)
        self.mask = cv2.flip(self.mask, 1)

    @property
    def nested_sprites(self):
        return []

    @property
    def is_outside(self):
        return False

class Car(Sprite):
    def __init__(self, scene):
        super(Car, self).__init__(scene)
        self.delay_start = 3
        self.__moving = None
        n = [-1, 0, 0, 0]
        p = [1, 0, 0, 0]
        self.shift_iter = itertools.cycle(n + p + p + n)

    def load(self):
        img = cv2.imread('animate\\body.png', -1)
        if img is None:
            raise Exception("Image not found: animate\\body.png")
        self.scale = 1 - max(img.shape) / self.scene.frame_size
        self.body, self.mask = \
            rgba_to_rgb(cv2.resize(img, None,
                        fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC))

        self.wheels = [Wheel(self.scene, self, n, f) \
            for n, f in enumerate(Path().cwd().joinpath('animate').glob('wheel*.png'))]

        self.glass = Glass(self.scene, self)

        self.sounds = [sa.WaveObject.from_wave_file(str(f)) \
            for f in Path().cwd().joinpath('animate').glob('sound*.wav')]
        self.cur_sound = None

    def set_loc(self, loc):
        self.loc = loc
        if self.scene.dir < 0:
            self.x = loc.road[1][0] - self.body.shape[1] - 20
            self.y = loc.road[1][1] - self.body.shape[0] - 20
        else:
            self.x = loc.road[0][0] + 20
            self.y = loc.road[1][1] - self.body.shape[0] - 20

    def one_frame(self, frame, n_frame, elapsed_time):
        if self.cur_sound is None or not self.cur_sound.is_playing():
            n = 1 if self.moving else 0
            self.cur_sound = self.sounds[n].play()
        frame = apply_patch(frame, self.x, self.y, self.body, self.mask)
        return frame

    def move(self, n_frame, elapsed_time):
        self.y += next(self.shift_iter)
        if elapsed_time > self.delay_start and self.moving is None:
            self.moving = True
        if self.moving:
            self.x = self.x + 5 * self.scene.dir

    @property
    def nested_sprites(self):
        a = self.wheels
        a.extend([self.glass])
        return a

    @property
    def is_outside(self):
        if scene.dir < 0:
            return self.x < 0 or self.y < 0
        else:
            return self.x + self.body.shape[1] >= self.loc.body.shape[1] or \
                   self.y + self.body.shape[0] >= self.loc.body.shape[0]

    @property
    def moving(self):
        return self.__moving

    @moving.setter
    def moving(self, m):
        self.__moving = m
        if self.cur_sound is not None:
            self.cur_sound.stop()
            self.cur_sound = None

class Wheel(Sprite):
    def __init__(self, scene, car, n_wheel, fname):
        self.fname = str(fname)
        self.car = car
        self.n_wheel = n_wheel
        self.angle = 0
        super(Wheel, self).__init__(scene)

    def load(self):
        img = cv2.imread(self.fname, -1)
        if img is None:
            raise Exception("Image not found: ", self.fname)

        self.body, self.mask = \
            rgba_to_rgb(cv2.resize(img, None, fx=self.car.scale, fy=self.car.scale,
                                   interpolation=cv2.INTER_CUBIC))

    def one_frame(self, frame, n_frame, elapsed_time):
        if self.n_wheel == 0:
            self.x = self.car.x + 20
            self.y = self.car.y + int(self.car.body.shape[0] * 0.7)
        else:
            self.x = self.car.x + int(self.car.body.shape[1] * 0.7)
            self.y = self.car.y + int(self.car.body.shape[0] * 0.65)

        wy, wx = self.body.shape[0] // 2, self.body.shape[1] // 2
        M = cv2.getRotationMatrix2D((wx, wy), self.angle, 1)
        img = cv2.warpAffine(self.body, M, self.body.shape[:2] )
        mask = cv2.warpAffine(self.mask, M, self.mask.shape[:2] )

        frame = apply_patch(frame, self.x, self.y, img, mask)
        return frame

    def move(self, n_frame, elapsed_time):
        if self.car.moving:
            self.angle = self.angle - 10 * self.scene.dir
            if self.angle >= 360:
                self.angle = 0

class Glass(Sprite):
    def __init__(self, scene, car):
        self.car = car
        super(Glass, self).__init__(scene)

    def load(self):
        img = cv2.imread("animate\\glass.png", -1)
        if img is None:
            raise Exception("Image not found: ", self.fname)

        self.body, self.mask = \
            rgba_to_rgb(cv2.resize(img, None, fx=self.car.scale, fy=self.car.scale,
                                   interpolation=cv2.INTER_CUBIC))

    def one_frame(self, frame, n_frame, elapsed_time):
        self.x = self.car.x + int(self.car.body.shape[1] * 0.2) - 3
        self.y = self.car.y + int(self.car.body.shape[0] * 0.1)
        if self.scene.dir > 0:
            self.x -= 19

        frame = apply_patch(frame, self.x, self.y, self.body, self.mask, alpha=0.4)
        return frame


class Sun(Sprite):
    def load(self):
        self.is_sun = True
        self.y, self.x, self.r = 0, 0, 0
        self.cy, self.cx, self.cr = 0, 0, 0
        self.angle = -200
        self.delay_start = 3
        self.moving = None

    def set_loc(self, loc):
        self.loc = loc
        self.cy = self.loc.horizon[0][1]
        self.cx = loc.body.shape[1] // 2
        self.cr = int(self.cx / 1.5)

    def one_frame(self, frame, n_frame, elapsed_time):
        if n_frame == 0:
            self.cur_cx, self.cur_cy, self.cur_cr = self.cx, self.cy, self.cr

        # Draw circle
        sky_shape = [self.cy, frame.shape[1]]
        yellow = color_to_cv_color('yellow')

        self.x = self.cur_cx + int(self.cur_cr * np.cos(self.angle * np.pi / 180))
        self.y = self.cur_cy + int(self.cur_cr * np.sin(self.angle * np.pi / 180))
        self.r = 60

        rr, cc = draw.circle(self.y, self.x, self.r, shape=sky_shape)
        rr_n = rr[rr < self.cy]
        if len(rr_n) > 0:
            frame[rr_n, cc] = yellow
        else:
            self.cur_cx, self.cur_cy, self.cur_cr = self.cx, self.cy, self.cr

        # Draw spikes
        for a in range(10, 361, 10):
            x1 = self.x + int(self.r * np.cos((a-5) * np.pi / 180))
            y1 = self.y + int(self.r * np.sin((a-5) * np.pi / 180))
            x2 = self.x + int((self.r + 30) * np.cos(a * np.pi / 180))
            y2 = self.y + int((self.r + 30) * np.sin(a * np.pi / 180))
            x3 = self.x + int(self.r * np.cos((a+5) * np.pi / 180))
            y3 = self.y + int(self.r * np.sin((a+5) * np.pi / 180))
            xc = self.x + int((self.r + 5) * np.cos(a * np.pi / 180))
            yc = self.y + int((self.r + 5) * np.sin(a * np.pi / 180))

            rr, cc = draw.bezier_curve(y1, x1, y2, x2, y3, x3, 8, shape=sky_shape)
            frame[rr, cc] = yellow

        # Brigtness
        if self.y > self.loc.horizon[0][1]:
            value = -30
        else:
            show_pct = float(self.loc.horizon[0][1] - self.y) / self.loc.horizon[0][1]
            value = 60 * show_pct - 30

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:,:,2] = cv2.add(hsv[:,:,2], int(value))
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return frame

    def move(self, n_frame, elapsed_time):
        if elapsed_time > self.delay_start and self.moving is None:
            self.moving = True
        if self.moving and n_frame % 2 == 0:
            self.angle += 0.5

class Location(Sprite):
    def __init__(self, scene, n_loc, fname):
        self.n_loc = n_loc
        self.fname = str(fname)
        self.elements = []
        super(Location, self).__init__(scene)

    def load(self):
        img = cv2.imread(self.fname)
        if img is None:
            raise Exception("Image not found: ", self.fname)

        self.body, self.scale, _ = resize3(img, self.scene.frame_size)
        self.road = [[0, self.body.shape[0]-150], [self.body.shape[1], self.body.shape[0]-50]]
        self.horizon = [[0, 310], [self.body.shape[1], 310]]

        deltas = [[0, 0], [80, 20], [60, -20], [38, -40], [70, 30]]
        for h in self.horizon:
            h[1] += deltas[self.n_loc][0]
        for r in self.road:
            r[1] += deltas[self.n_loc][1]

        self.elements = [LocElement(self.scene, self, n, str(f)) \
            for n, f in enumerate(Path().cwd().joinpath('animate').glob( \
            'elem' + str(self.n_loc+1) + '_*.png'))]

    def flip(self):
        pass

    @property
    def nested_sprites(self):
        return self.elements


class LocElement(Sprite):
    def __init__(self, scene, loc, n_elem, fname):
        self.fname = str(fname)
        self.parent_loc = loc
        self.n_elem = n_elem
        super(LocElement, self).__init__(scene)

    def load(self):
        img = cv2.imread(self.fname, -1)
        if img is None:
            raise Exception("Image not found: ", self.fname)

        self.body, self.mask = \
            rgba_to_rgb(cv2.resize(img, None, fx=self.parent_loc.scale[0],
                                   fy=self.parent_loc.scale[1],
                                   interpolation=cv2.INTER_CUBIC))

    def one_frame(self, frame, n_frame, elapsed_time):
        if self.loc == self.parent_loc:
            if self.parent_loc.n_loc == 0:
                self.x = 0
                self.y = self.loc.horizon[0][1] - self.body.shape[0] + 8
            elif self.parent_loc.n_loc == 1:
                self.x = frame.shape[1] - self.body.shape[1] - 0
                self.y = frame.shape[0] - self.body.shape[0] - 182
            elif self.parent_loc.n_loc == 2:
                self.x = 143
                self.y = 95
            elif self.parent_loc.n_loc == 3:
                self.x = 0
                self.y = self.parent_loc.horizon[0][1] - self.body.shape[0] + 10
            elif self.parent_loc.n_loc == 4:
                self.x = 0
                self.y = self.parent_loc.horizon[0][1] - self.body.shape[0] + 35
            frame = apply_patch(frame, self.x, self.y, self.body, self.mask)
        return frame

    def flip(self):
        pass


class Scene:
    def __init__(self):
        self.frame_size = 1024
        self.locations = [Location(self, n, f) \
            for n, f in enumerate(Path().cwd().joinpath('animate').glob('back*.jpg'))]
        self.sprites = [Sun(self)]
        self.sprites.extend(self.locations)
        self.sprites.extend([Car(self)])
        self.car = self.sprites[-1]
        self.loc_iter = reversable_iter(self.locations)
        self.recorder = None
        self.dir = -1

    @property
    def all_sprites(self):
        sprites = []
        for sprite in self.sprites:
            sprites.extend([sprite])
            sprites.extend(sprite.nested_sprites)
        return sprites

    def run(self, interactive=True, record=False):
        start_time = time.time()
        n_frame = 0
        delay_iter = reversable_iter(list(np.linspace(5, stop=30, num=5).astype(int)))
        delay = next(delay_iter)
        recording = False

        sprites = self.all_sprites

        loc = next(self.loc_iter)
        for sprite in sprites:
            sprite.set_loc(loc)

        if record:
            recording = True

        while(True):
            frame = loc.body.copy()
            tm = time.time() - start_time
            for sprite in sprites:
                frame = sprite.one_frame(frame, n_frame, tm)

            cv2.imshow('Frame', frame)
            if recording:
                self.record(frame)

            key = cv2.waitKey(delay)
            if key > 0 and not interactive:
                break
            elif key & 0xFF == ord('-'):
                delay_iter.reverse_if(True)
                delay = next(delay_iter)
                print('Delay set to', delay)
            elif key & 0xFF == ord('+'):
                delay_iter.reverse_if(False)
                delay = next(delay_iter)
                print('Delay set to', delay)
            elif key & 0xFF == ord(' '):
                self.car.moving = \
                    True if self.car.moving is None \
                    else not self.car.moving
            elif key & 0xFF == ord('r'):
                if not recording:
                    self.record(frame)
                    recording = True
                else:
                    self.stop_recording()
                    recording = False
            elif key & 0xFF == ord('q'):
                print('Stopping...')
                break
            elif key & 0xFF == ord('/'):
                self.dir = -1 if self.dir == 1 else 1
                self.loc_iter.reverse_if(True)
                for sprite in sprites:
                    sprite.flip()

            for sprite in sprites:
                sprite.move(n_frame, tm)
            if self.car.is_outside:
                loc = next(self.loc_iter)
                if loc.n_loc == 0 and record:
                    break;
                for sprite in sprites:
                    sprite.set_loc(loc)

            sprites = [s for s in sprites if s == self.car or not sprite.is_outside]
            n_frame += 1

        self.stop_recording()
        sa.stop_all()
        cv2.waitKey()

    def show_location(self, n_loc=0, highlight=False):
        img = self.locations[n_loc].body.copy()
        if highlight:
            r = self.locations[n_loc].road
            poly = [ [r[0][0], r[0][1]], [r[0][0], r[1][1]], [r[1][0], r[1][1]], [r[1][0], r[0][1]] ]
            cv2.fillPoly(img, [np.array(poly)], [127, 127, 127])

            h = self.locations[n_loc].horizon
            cv2.line(img, tuple(h[0]), tuple(h[1]), (0,0,255), 2)

        cv2.imshow('Location', img)
        cv2.waitKey()

    def show_frame(self, n_loc=0):
        loc = self.locations[n_loc]
        frame = loc.body.copy()
        for sprite in self.all_sprites:
            sprite.set_loc(loc)
            frame = sprite.one_frame(frame, 0, 0)

        cv2.imshow('Frame', frame)
        sa.stop_all()
        cv2.waitKey()

    def record(self, frame):
        if self.recorder is None:
            file_name = 'animate\\car.avi'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.recorder = cv2.VideoWriter(file_name, fourcc, 40.0, (frame.shape[1], frame.shape[0]))
            print('Recoding started')

        self.recorder.write(frame)

    def stop_recording(self):
        if self.recorder is not None:
            self.recorder.release()
            self.recorder = None
            print('Recoding stopped')


scene = Scene()
##for n in range(len(scene.locations)):
##    scene.show_location(n, highlight=True)
#scene.sprites[0].angle = -45
#scene.show_frame(0)
scene.run(record=False)

cv2.destroyAllWindows()
