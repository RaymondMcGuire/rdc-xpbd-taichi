import time
import taichi as ti
from xpbd_gpu.xpbd_base import XPBDSolverBase, calc_tet_volume


def calc_first_lame(E, nu):
    return 2.0 * nu / (1.0 - 2.0 * nu) * calc_second_lame(E, nu)


def calc_second_lame(E, nu):
    return E / (2.0 * (1.0 + nu))


@ti.data_oriented
class XPBDNeoHookeanSolver(XPBDSolverBase):
    """
    XPBD solver using neohookean constraints (deviatoric and hydrostatic preservation)
    Chebyshev-accelerated Jacobi outer loop (CSI).
    """

    def __init__(self,
                 rest_pose,
                 sdf,
                 young_modulus=200000.0,
                 poisson_ratio=0.495,
                 scale=1.0,
                 offset=(0.0, 0.0, 0.0),
                 frame_dt=1e-2,
                 dt=1e-2,
                 substeps_num=20,
                 constraints_iter=5,
                 reorder_all=False,
                 block_size=256,
                 hydrostatic_relaxation=1.2,
                 deviatoric_relaxation=1.2,
                 cheb_enable=True,
                 cheb_warmup=5,
                 cheb_gamma=0.6666,

                 block_neohookean_enable=False,
                 monitor_convergence=False):
        self.constraints_iter = constraints_iter
        self.hydrostatic_relaxation = hydrostatic_relaxation
        self.deviatoric_relaxation = deviatoric_relaxation

        self.lame1 = calc_first_lame(young_modulus, poisson_ratio)
        self.lame2 = calc_second_lame(young_modulus, poisson_ratio)

        self.cheb_enable = cheb_enable
        self.cheb_gamma = cheb_gamma
        self.cheb_warmup = cheb_warmup

        self.block_neohookean_enable = block_neohookean_enable

        self.monitor_convergence = monitor_convergence
        self.convergence_history = []
        self.current_substep_convergence = []
        self.current_frame = 0
        self.current_substep = 0
        self.convergence_callback = None

        self.hydro_error_sum = ti.field(dtype=ti.f32, shape=())
        self.dev_error_sum = ti.field(dtype=ti.f32, shape=())
        self.constraint_count = ti.field(dtype=ti.i32, shape=())

        self.residual_sum = ti.field(dtype=ti.f32, shape=())
        self.residual_count = ti.field(dtype=ti.i32, shape=())
        self.rho_hat = 0.999
        self.rho_min = 0.00001
        self.rho_max = 0.99999
        self.rho_beta = 0.2
        self.prev_r_hat = None

        self.solve_compute_time_total = 0.0
        self.solve_compute_calls = 0
        self.solve_compute_iterations = 0

        self.block_size = block_size
        super().__init__(rest_pose, sdf, scale, offset, frame_dt, dt,
                         substeps_num, reorder_all, block_size)

    def initialize_specific(self, scale, offset):
        self.mesh.verts.place({
            'delta_x': ti.math.vec3,
            'cons_num': ti.u32,
            'pred_x_prev': ti.math.vec3,
            'pred_x_curr': ti.math.vec3,
        }, reorder=False)

        self.mesh.cells.place({
            'V0': ti.f32,
            'inv_Dm': ti.math.mat3,
            'lambda_H': ti.f32,
            'lambda_D': ti.f32,
        }, reorder=False)

        self.initialize(scale, offset)
        self.compute_lumped_mass()

    @ti.kernel
    def initialize(self, scale: ti.f32, offset: ti.template()):
        for v in self.mesh.verts:
            v.x = v.x * scale + ti.Vector(offset)

        for c in self.mesh.cells:
            c.V0 = calc_tet_volume(c.verts[0].x, c.verts[1].x,
                                   c.verts[2].x, c.verts[3].x)
            c.inv_Dm = ti.Matrix.cols([c.verts[1].x - c.verts[0].x,
                                       c.verts[2].x - c.verts[0].x,
                                       c.verts[3].x - c.verts[0].x]).inverse()

    @ti.kernel
    def pre_solve(self):
        for v in self.mesh.verts:
            v.delta_x.fill(0.0)
            v.cons_num = 0

    @ti.kernel
    def post_solve(self):
        for v in self.mesh.verts:
            if v.cons_num > 0:
                v.pred_x += v.delta_x / v.cons_num

    @ti.kernel
    def _save_curr_to_prev(self):
        for v in self.mesh.verts:
            v.pred_x_prev = v.pred_x

    @ti.kernel
    def _save_curr_to_savebuf(self):
        for v in self.mesh.verts:
            v.pred_x_curr = v.pred_x

    @ti.kernel
    def _chebyshev_update(self, omega: ti.f32):
        for v in self.mesh.verts:
            x_new = self.cheb_gamma * \
                (v.pred_x - v.pred_x_curr) + v.pred_x_curr
            v.pred_x = v.pred_x_prev + omega * (x_new - v.pred_x_prev)

    @ti.kernel
    def _rotate_prev_with_save(self):
        for v in self.mesh.verts:
            v.pred_x_prev = v.pred_x_curr

    @ti.kernel
    def solve_deviatoric(self, dt: ti.f32):
        ti.loop_config(block_dim=self.block_size)
        ti.mesh_local(self.mesh.verts.delta_x, self.mesh.verts.inv_m,
                      self.mesh.verts.pred_x)

        for c in self.mesh.cells:
            p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
            w0, w1, w2, w3 = p0.inv_m, p1.inv_m, p2.inv_m, p3.inv_m

            pos0, pos1, pos2, pos3 = p0.pred_x, p1.pred_x, p2.pred_x, p3.pred_x

            v1, v2, v3 = pos1 - pos0, pos2 - pos0, pos3 - pos0
            Ds = ti.Matrix.cols([v1, v2, v3])
            F = Ds @ c.inv_Dm

            trFTF = F.norm_sqr()
            C = trFTF - 3.0

            grad_F = 2.0 * F
            grad_123 = grad_F @ c.inv_Dm.transpose()

            grad_1 = ti.Vector([grad_123[j, 0] for j in ti.static(range(3))])
            grad_2 = ti.Vector([grad_123[j, 1] for j in ti.static(range(3))])
            grad_3 = ti.Vector([grad_123[j, 2] for j in ti.static(range(3))])
            grad_0 = -(grad_1 + grad_2 + grad_3)

            w_sum = w0 * grad_0.dot(grad_0)
            w_sum += w1 * grad_1.dot(grad_1)
            w_sum += w2 * grad_2.dot(grad_2)
            w_sum += w3 * grad_3.dot(grad_3)

            if w_sum > 1e-6:
                alpha_D = 1.0 / (self.lame2 * c.V0)
                alpha_tilde = alpha_D / (dt ** 2)

                d_lambda = -(C + alpha_tilde * c.lambda_D) / \
                    (w_sum + alpha_tilde)
                d_lambda *= self.deviatoric_relaxation
                c.lambda_D += d_lambda

                ti.atomic_add(p0.delta_x, d_lambda * w0 * grad_0)
                ti.atomic_add(p1.delta_x, d_lambda * w1 * grad_1)
                ti.atomic_add(p2.delta_x, d_lambda * w2 * grad_2)
                ti.atomic_add(p3.delta_x, d_lambda * w3 * grad_3)

                ti.atomic_add(p0.cons_num, 1)
                ti.atomic_add(p1.cons_num, 1)
                ti.atomic_add(p2.cons_num, 1)
                ti.atomic_add(p3.cons_num, 1)

    @ti.kernel
    def solve_hydrostatic(self, dt: ti.f32):
        ti.loop_config(block_dim=self.block_size)
        ti.mesh_local(self.mesh.verts.delta_x, self.mesh.verts.inv_m,
                      self.mesh.verts.pred_x)

        for c in self.mesh.cells:
            p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
            w0, w1, w2, w3 = p0.inv_m, p1.inv_m, p2.inv_m, p3.inv_m

            pos0, pos1, pos2, pos3 = p0.pred_x, p1.pred_x, p2.pred_x, p3.pred_x

            v1, v2, v3 = pos1 - pos0, pos2 - pos0, pos3 - pos0
            Ds = ti.Matrix.cols([v1, v2, v3])
            F = Ds @ c.inv_Dm

            detF = F.determinant()
            C = detF - 1.0

            F_col0 = ti.Vector([F[j, 0] for j in ti.static(range(3))])
            F_col1 = ti.Vector([F[j, 1] for j in ti.static(range(3))])
            F_col2 = ti.Vector([F[j, 2] for j in ti.static(range(3))])

            cross0 = F_col1.cross(F_col2)
            cross1 = F_col2.cross(F_col0)
            cross2 = F_col0.cross(F_col1)
            grad_F = ti.Matrix.cols([cross0, cross1, cross2])

            grad_123 = grad_F @ c.inv_Dm.transpose()

            grad_1 = ti.Vector([grad_123[j, 0] for j in ti.static(range(3))])
            grad_2 = ti.Vector([grad_123[j, 1] for j in ti.static(range(3))])
            grad_3 = ti.Vector([grad_123[j, 2] for j in ti.static(range(3))])
            grad_0 = -(grad_1 + grad_2 + grad_3)

            w_sum = w0 * grad_0.dot(grad_0)
            w_sum += w1 * grad_1.dot(grad_1)
            w_sum += w2 * grad_2.dot(grad_2)
            w_sum += w3 * grad_3.dot(grad_3)

            if w_sum > 1e-6:
                alpha_H = 1.0 / (self.lame1 * c.V0)
                alpha_tilde = alpha_H / (dt ** 2)

                d_lambda = -(C + alpha_tilde * c.lambda_H) / \
                    (w_sum + alpha_tilde)
                d_lambda *= self.hydrostatic_relaxation
                c.lambda_H += d_lambda

                ti.atomic_add(p0.delta_x, d_lambda * w0 * grad_0)
                ti.atomic_add(p1.delta_x, d_lambda * w1 * grad_1)
                ti.atomic_add(p2.delta_x, d_lambda * w2 * grad_2)
                ti.atomic_add(p3.delta_x, d_lambda * w3 * grad_3)

                ti.atomic_add(p0.cons_num, 1)
                ti.atomic_add(p1.cons_num, 1)
                ti.atomic_add(p2.cons_num, 1)
                ti.atomic_add(p3.cons_num, 1)

    @ti.kernel
    def compute_hydrostatic_error(self):
        self.hydro_error_sum[None] = 0.0
        self.constraint_count[None] = 0

        ti.loop_config(block_dim=self.block_size)
        ti.mesh_local(self.mesh.verts.pred_x)

        for c in self.mesh.cells:
            p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
            pos0, pos1, pos2, pos3 = p0.pred_x, p1.pred_x, p2.pred_x, p3.pred_x

            v1, v2, v3 = pos1 - pos0, pos2 - pos0, pos3 - pos0
            Ds = ti.Matrix.cols([v1, v2, v3])
            F = Ds @ c.inv_Dm

            detF = F.determinant()
            C_H = detF - 1.0

            ti.atomic_add(self.hydro_error_sum[None], ti.abs(C_H))
            ti.atomic_add(self.constraint_count[None], 1)

    @ti.kernel
    def compute_deviatoric_error(self):
        self.dev_error_sum[None] = 0.0

        ti.loop_config(block_dim=self.block_size)
        ti.mesh_local(self.mesh.verts.pred_x)

        for c in self.mesh.cells:
            p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
            pos0, pos1, pos2, pos3 = p0.pred_x, p1.pred_x, p2.pred_x, p3.pred_x

            v1, v2, v3 = pos1 - pos0, pos2 - pos0, pos3 - pos0
            Ds = ti.Matrix.cols([v1, v2, v3])
            F = Ds @ c.inv_Dm

            trFTF = F.norm_sqr()
            C_D = trFTF - 3.0

            ti.atomic_add(self.dev_error_sum[None], ti.abs(C_D))

    @ti.kernel
    def compute_xpbd_residual(self, dt: ti.f32):
        self.residual_sum[None] = 0.0
        self.residual_count[None] = 0

        ti.loop_config(block_dim=self.block_size)
        ti.mesh_local(self.mesh.verts.pred_x)

        for c in self.mesh.cells:
            p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
            pos0, pos1, pos2, pos3 = p0.pred_x, p1.pred_x, p2.pred_x, p3.pred_x

            v1, v2, v3 = pos1 - pos0, pos2 - pos0, pos3 - pos0
            Ds = ti.Matrix.cols([v1, v2, v3])
            F = Ds @ c.inv_Dm

            C_D = F.norm_sqr() - 3.0
            C_H = F.determinant() - 1.0

            alpha_D = 1.0 / (self.lame2 * c.V0)
            alpha_H = 1.0 / (self.lame1 * c.V0)
            aD = alpha_D / (dt * dt)
            aH = alpha_H / (dt * dt)

            rD = C_D + aD * c.lambda_D
            rH = C_H + aH * c.lambda_H

            r = ti.sqrt(rD * rD + rH * rH)

            ti.atomic_add(self.residual_sum[None], r)
            ti.atomic_add(self.residual_count[None], 1)

    @ti.func
    def solve2x2_pivot_lu(self, A00, A01, A10, A11, b0, b1, eps):
        ok = 1

        if ti.abs(A00) < ti.abs(A10):
            A00, A10 = A10, A00
            A01, A11 = A11, A01
            b0,  b1 = b1,  b0

        if ti.abs(A00) < eps:
            ok = 0

        x0 = 0.0
        x1 = 0.0
        if ok == 1:
            m = A10 / A00
            A11_ = A11 - m * A01
            b1_ = b1 - m * b0

            if ti.abs(A11_) < eps:
                ok = 0
            else:
                x1 = b1_ / A11_
                x0 = (b0 - A01 * x1) / A00

        return ok, x0, x1

    @ti.kernel
    def solve_neohookean_block(self, dt: ti.f32):
        ti.loop_config(block_dim=self.block_size)
        ti.mesh_local(self.mesh.verts.delta_x, self.mesh.verts.inv_m,
                      self.mesh.verts.pred_x)

        eps_A = 1e-12
        for c in self.mesh.cells:
            p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
            w0, w1, w2, w3 = p0.inv_m, p1.inv_m, p2.inv_m, p3.inv_m

            pos0, pos1, pos2, pos3 = p0.pred_x, p1.pred_x, p2.pred_x, p3.pred_x

            v1, v2, v3 = pos1 - pos0, pos2 - pos0, pos3 - pos0
            Ds = ti.Matrix.cols([v1, v2, v3])
            F = Ds @ c.inv_Dm

            trFTF = F.norm_sqr()
            C_D = trFTF - 3.0

            detF = F.determinant()
            C_H = detF - 1.0

            invDmT = c.inv_Dm.transpose()

            gradF_D = F * 2.0
            grad123_D = gradF_D @ invDmT
            gD1 = ti.Vector([grad123_D[j, 0] for j in ti.static(range(3))])
            gD2 = ti.Vector([grad123_D[j, 1] for j in ti.static(range(3))])
            gD3 = ti.Vector([grad123_D[j, 2] for j in ti.static(range(3))])
            gD0 = -(gD1 + gD2 + gD3)

            F0 = ti.Vector([F[j, 0] for j in ti.static(range(3))])
            F1 = ti.Vector([F[j, 1] for j in ti.static(range(3))])
            F2 = ti.Vector([F[j, 2] for j in ti.static(range(3))])

            cross0 = F1.cross(F2)
            cross1 = F2.cross(F0)
            cross2 = F0.cross(F1)
            gradF_H = ti.Matrix.cols([cross0, cross1, cross2])

            grad123_H = gradF_H @ invDmT
            gH1 = ti.Vector([grad123_H[j, 0] for j in ti.static(range(3))])
            gH2 = ti.Vector([grad123_H[j, 1] for j in ti.static(range(3))])
            gH3 = ti.Vector([grad123_H[j, 2] for j in ti.static(range(3))])
            gH0 = -(gH1 + gH2 + gH3)

            A11 = w0 * gD0.dot(gD0) + w1 * gD1.dot(gD1) + \
                w2 * gD2.dot(gD2) + w3 * gD3.dot(gD3)
            A22 = w0 * gH0.dot(gH0) + w1 * gH1.dot(gH1) + \
                w2 * gH2.dot(gH2) + w3 * gH3.dot(gH3)
            A12 = w0 * gD0.dot(gH0) + w1 * gD1.dot(gH1) + \
                w2 * gD2.dot(gH2) + w3 * gD3.dot(gH3)

            alpha_D = 1.0 / (self.lame2 * c.V0)
            alpha_H = 1.0 / (self.lame1 * c.V0)
            aD = alpha_D / (dt * dt)
            aH = alpha_H / (dt * dt)

            A11 += aD
            A22 += aH

            b1 = -(C_D + aD * c.lambda_D)
            b2 = -(C_H + aH * c.lambda_H)
            ok, dlamD, dlamH = self.solve2x2_pivot_lu(
                A11, A12, A12, A22, b1, b2, eps_A)
            if ok == 0:
                dlamD = 0.0
                dlamH = 0.0
                if ti.abs(A11) > eps_A:
                    dlamD = b1 / A11
                if ti.abs(A22) > eps_A:
                    dlamH = b2 / A22

            c.lambda_D += dlamD
            c.lambda_H += dlamH

            dp0 = w0 * (gD0 * dlamD + gH0 * dlamH)
            dp1 = w1 * (gD1 * dlamD + gH1 * dlamH)
            dp2 = w2 * (gD2 * dlamD + gH2 * dlamH)
            dp3 = w3 * (gD3 * dlamD + gH3 * dlamH)

            ti.atomic_add(p0.delta_x, dp0)
            ti.atomic_add(p1.delta_x, dp1)
            ti.atomic_add(p2.delta_x, dp2)
            ti.atomic_add(p3.delta_x, dp3)

            ti.atomic_add(p0.cons_num, 1)
            ti.atomic_add(p1.cons_num, 1)
            ti.atomic_add(p2.cons_num, 1)
            ti.atomic_add(p3.cons_num, 1)

    def solve_step(self, dt):
        compute_time_acc = 0.0
        setup_time_start = time.perf_counter()

        self.mesh.cells.lambda_H.fill(0.0)
        self.mesh.cells.lambda_D.fill(0.0)

        self.prev_r_hat = None

        self.solve_sdf_collision(dt)

        use_cheb = self.cheb_enable
        omega = 1.0

        if use_cheb:
            self._save_curr_to_prev()

        compute_time_acc += time.perf_counter() - setup_time_start

        for it in range(self.constraints_iter):
            iter_compute_time_sec = 0.0
            iter_compute_start = time.perf_counter()

            if use_cheb:
                self._save_curr_to_savebuf()

            self.pre_solve()

            if self.block_neohookean_enable:
                self.solve_neohookean_block(dt)
            else:
                self.solve_hydrostatic(dt)
                self.solve_deviatoric(dt)

            self.post_solve()

            if use_cheb:
                self.compute_xpbd_residual(dt)
                cnt = self.residual_count[None]
                r_hat = self.residual_sum[None] / cnt if cnt > 0 else 0.0

                if self.prev_r_hat is not None and self.prev_r_hat > 1e-12:
                    rho_inst = r_hat / (self.prev_r_hat + 1e-12)
                    rho_inst = max(self.rho_min, min(
                        self.rho_max, float(rho_inst)))
                    self.rho_hat = (1.0 - self.rho_beta) * \
                        self.rho_hat + self.rho_beta * rho_inst
                self.prev_r_hat = r_hat

            iter_compute_time_sec += time.perf_counter() - iter_compute_start

            convergence_record = None
            if self.monitor_convergence:
                self.compute_hydrostatic_error()
                self.compute_deviatoric_error()

                num_constraints = self.constraint_count[None]
                if num_constraints > 0:
                    hydro_error = self.hydro_error_sum[None] / num_constraints
                    dev_error = self.dev_error_sum[None] / num_constraints
                else:
                    hydro_error = 0.0
                    dev_error = 0.0

                record = {
                    'frame': self.current_frame,
                    'substep': self.current_substep,
                    'iter': it,
                    'iter_compute_time_sec': iter_compute_time_sec,
                    'hydro_error': hydro_error,
                    'dev_error': dev_error
                }

                self.convergence_history.append(record)
                self.current_substep_convergence.append(record)
                convergence_record = record

            if use_cheb:
                iter_compute_start = time.perf_counter()

                rho = float(self.rho_hat)
                rho2 = rho * rho

                if it < self.cheb_warmup:
                    omega = 1.0
                elif it == self.cheb_warmup:
                    omega = 2.0 / (2.0 - rho2)
                else:
                    omega = 4.0 / (4.0 - rho2 * omega)

                self._chebyshev_update(omega)
                self._rotate_prev_with_save()

                iter_compute_time_sec += time.perf_counter() - iter_compute_start

                if convergence_record is not None:
                    convergence_record['iter_compute_time_sec'] = iter_compute_time_sec

            compute_time_acc += iter_compute_time_sec

        self.solve_compute_time_total += compute_time_acc
        self.solve_compute_calls += 1
        self.solve_compute_iterations += self.constraints_iter

    def solve(self):
        frame_time_left = self.frame_dt
        self.current_substep = 0

        while frame_time_left > 0.0:
            dt0 = min(self.dt, frame_time_left)
            frame_time_left -= dt0

            sub_dt = dt0 / self.substeps_num
            for substep in range(self.substeps_num):
                self.current_substep = substep

                if self.monitor_convergence:
                    self.current_substep_convergence = []

                self.apply_external_forces(sub_dt)
                self.solve_step(sub_dt)
                self.advance(sub_dt, self.damping)

                if self.monitor_convergence and hasattr(self, 'convergence_callback') and self.convergence_callback:
                    self.convergence_callback(self.current_frame, self.current_substep,
                                              self.current_substep_convergence)

            self.time += dt0

        self.current_frame += 1

