import taichi as ti
import numpy as np


@ti.data_oriented
class SdfModel:
    def __init__(self, fixed):
        self.vel = ti.Vector.field(3, float, shape=())
        self.fixed = ti.field(int, shape=())
        self.fixed[None] = fixed

    @ti.func
    def check(self, pos, vel):
        phi = self.dist(pos)
        inside = False
        dotnv = 0.0
        diff_vel = ti.Vector.zero(ti.f32, 3)
        n = ti.Vector.zero(ti.f32, 3)
        if phi < 0.0:
            n = self.normal(pos)
            diff_vel = self.vel[None] - vel
            dotnv = n.dot(diff_vel)
            if dotnv > 0.0 or self.fixed[None]:
                inside = True
        
        return self.fixed[None], inside, dotnv, diff_vel, n

@ti.data_oriented
class HangSdfModel(SdfModel):
    def __init__(self, pos):
        super().__init__(fixed=True)
        self.sphere_pos = ti.Vector.field(3, float, shape=pos.shape[0])
        self.sphere_pos.from_numpy(pos.astype(np.float32))
        self.sphere_radius = 0.013

    @ti.func
    def dist(self, pos): # Function computing the signed distance field
        dist = 1e5
        for i in range(self.sphere_pos.shape[0]):
            dist = min((pos - self.sphere_pos[i]).norm(1e-9) - self.sphere_radius, dist)
        return dist

    @ti.func
    def normal(self, pos): # Function computing the gradient of signed distance field
        dist = 1e5
        normal = ti.Vector.zero(ti.f32, 3)
        for i in range(self.sphere_pos.shape[0]):
            dist0 = (pos - self.sphere_pos[i]).norm(1e-9) - self.sphere_radius
            if dist0 < dist:
                dist = dist0
                normal = (pos - self.sphere_pos[0]).normalized(1e-9)
        return normal
    
    def render(self, scene):
        scene.particles(self.sphere_pos, self.sphere_radius, color = (1, 0, 0))


@ti.data_oriented
class PlaneSdf(SdfModel):
    def __init__(self, point, dir, fixed=False):
        super().__init__(fixed=fixed)
        self.point = ti.Vector.field(3, float, shape=())
        self.dir = ti.Vector.field(3, float, shape=())
        self.point[None] = ti.Vector(point)
        self.dir[None] = ti.Vector(dir).normalized()

    @ti.func
    def dist(self, pos):
        return (pos - self.point[None]).dot(self.dir[None])

    @ti.func
    def normal(self, pos):
        return self.dir[None]


@ti.data_oriented
class SphereSdf(SdfModel):
    def __init__(self, center, radius, fixed=False):
        super().__init__(fixed=fixed)
        self.center = ti.Vector.field(3, float, shape=())
        self.radius = ti.field(float, shape=())
        self.center[None] = ti.Vector(center)
        self.radius[None] = radius

    @ti.func
    def dist(self, pos):
        return (pos - self.center[None]).norm() - self.radius[None]

    @ti.func
    def normal(self, pos):
        return (pos - self.center[None]).normalized(1e-9)

    def render(self, scene):
        scene.particles(self.center, self.radius[None], color=(0, 1, 0))


@ti.data_oriented
class CubeSdf(SdfModel):
    def __init__(self, center, size, fixed=False):
        super().__init__(fixed=fixed)
        self.center = ti.Vector.field(3, float, shape=())
        self.half_size = ti.Vector.field(3, float, shape=())
        self.center[None] = ti.Vector(center)
        self.half_size[None] = ti.Vector(size) * 0.5

    @ti.func
    def dist(self, pos):
        d = ti.abs(pos - self.center[None]) - self.half_size[None]
        return ti.max(d, 0.0).norm() + ti.min(ti.max(d.x, ti.max(d.y, d.z)), 0.0)

    @ti.func
    def normal(self, pos):
        d = pos - self.center[None]
        abs_d = ti.abs(d)
        max_axis = 0
        if abs_d.y > abs_d.x:
            max_axis = 1
        if abs_d.z > abs_d[max_axis]:
            max_axis = 2
        
        n = ti.Vector.zero(ti.f32, 3)
        n[max_axis] = 1.0 if d[max_axis] > 0 else -1.0
        return n