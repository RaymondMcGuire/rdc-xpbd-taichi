import time
import taichi as ti
from xpbd_gpu.xpbd_base import XPBDSolverBase, calc_tet_volume


# --------- material helpers ---------
def calc_first_lame(E, nu):
    # lambda = 2 nu / (1 - 2 nu) * mu
    return 2.0 * nu / (1.0 - 2.0 * nu) * calc_second_lame(E, nu)


def calc_second_lame(E, nu):
    # mu = E / (2 (1 + nu))
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
                 # ---- Chebyshev outer-loop parameters ----
                 cheb_enable=True,     # enable Chebyshev semi-iterative acceleration
                 # estimated spectral radius (be conservative)
                 cheb_rho=0.999,
                 dynamic_cheb_rho=True,  # use adaptive rho_hat for omega if True
                 cheb_warmup=5,        # number of warmup iterations with omega=1
                 cheb_gamma=0.6666,            # relaxation parameter for Chebyshev update

                 neohookean_block_enable=False,  # enable Neo-Hookean block solver
                 # ---- Convergence monitoring  ----
                 monitor_convergence=False,  # enable constraint error monitoring
                 monitor_rho=False):  # enable rho tracking
        # Store constraint parameters
        self.constraints_iter = constraints_iter
        self.hydrostatic_relaxation = hydrostatic_relaxation
        self.deviatoric_relaxation = deviatoric_relaxation

        # Compliance / material parameters
        self.lame1 = calc_first_lame(young_modulus, poisson_ratio)
        self.lame2 = calc_second_lame(young_modulus, poisson_ratio)

        # Chebyshev config (outer acceleration)
        self.cheb_enable = cheb_enable
        self.cheb_rho = cheb_rho
        self.dynamic_cheb_rho = dynamic_cheb_rho
        self.cheb_gamma = cheb_gamma
        self.cheb_warmup = cheb_warmup

        self.neohookean_block_enable = neohookean_block_enable

        # Convergence monitoring
        self.monitor_convergence = monitor_convergence
        # List of dicts with {frame, substep, iter, hydro_error, dev_error}
        self.convergence_history = []
        self.current_substep_convergence = []  # Current substep's convergence data
        self.current_frame = 0
        self.current_substep = 0
        self.convergence_callback = None  # Callback function for per-substep plotting

        # Rho tracking
        self.monitor_rho = monitor_rho
        # List of dicts with {frame, substep, iter, rho_hat, r_hat, rho_inst}
        self.rho_history = []
        self.current_substep_rho = []  # Current substep's rho data
        self.rho_callback = None  # Callback for per-substep rho plotting

        # Taichi fields for computing constraint errors
        self.hydro_error_sum = ti.field(dtype=ti.f32, shape=())
        self.dev_error_sum = ti.field(dtype=ti.f32, shape=())
        self.constraint_count = ti.field(dtype=ti.i32, shape=())

        self.residual_sum = ti.field(dtype=ti.f32, shape=())
        self.residual_count = ti.field(dtype=ti.i32, shape=())
        self.rho_hat = float(self.cheb_rho)
        self.rho_min = 0.00001
        self.rho_max = 0.99999
        self.rho_beta = 0.2     # EMA
        self.prev_r_hat = None

        self.cheb_safeguard = False
        self.cheb_tol = 0.05
        self.solve_compute_time_total = 0.0
        self.solve_compute_calls = 0
        self.solve_compute_iterations = 0

        print("dynamic_cheb_rho:", self.dynamic_cheb_rho)

        # Base init
        self.block_size = block_size
        super().__init__(rest_pose, sdf, scale, offset, frame_dt, dt,
                         substeps_num, reorder_all, block_size)

    # ----------------------------------------------------------------------
    # Setup fields specific to XPBD + Chebyshev caching
    # ----------------------------------------------------------------------
    def initialize_specific(self, scale, offset):
        """Initialize XPBD-specific and Chebyshev-caching fields"""

        # Per-vertex fields
        self.mesh.verts.place({
            # accumulated position correction (Jacobi)
            'delta_x': ti.math.vec3,
            'cons_num': ti.u32,        # number of constraints affecting the vertex
            # --- Chebyshev caches  ---
            'pred_x_prev': ti.math.vec3,  # y_{k-1} positions
            'pred_x_curr': ti.math.vec3,  # saved y_k (before current sweep)
            'pred_x_hat': ti.math.vec3,
        }, reorder=False)

        # Per-cell (tet) fields
        self.mesh.cells.place({
            'V0': ti.f32,              # rest volume
            'inv_Dm': ti.math.mat3,    # inverse rest pose matrix (3x3)
            'lambda_H': ti.f32,        # XPBD multipliers (hydrostatic)
            'lambda_D': ti.f32,        # XPBD multipliers (deviatoric)
            # --- Chebyshev caches  ---
            'lambda_H_prev': ti.f32,   # y_{k-1} multipliers
            'lambda_D_prev': ti.f32,
            'lambda_H_save': ti.f32,   # saved y_k multipliers
            'lambda_D_save': ti.f32,
            'lambda_H_hat': ti.f32,
            'lambda_D_hat': ti.f32,
        }, reorder=False)

        # Initialize geometry
        self.initialize(scale, offset)

        # Lumped masses (based on topology/material)
        self.compute_lumped_mass()

    @ti.kernel
    def initialize(self, scale: ti.f32, offset: ti.template()):
        """Scale/offset vertices and precompute rest pose per cell"""
        # Scale and offset vertex positions
        for v in self.mesh.verts:
            v.x = v.x * scale + ti.Vector(offset)

        # Precompute rest volume and inverse rest pose matrix for each tet
        for c in self.mesh.cells:
            c.V0 = calc_tet_volume(c.verts[0].x, c.verts[1].x,
                                   c.verts[2].x, c.verts[3].x)
            c.inv_Dm = ti.Matrix.cols([c.verts[1].x - c.verts[0].x,
                                       c.verts[2].x - c.verts[0].x,
                                       c.verts[3].x - c.verts[0].x]).inverse()

    @ti.kernel
    def _save_hat(self):
        for v in self.mesh.verts:
            v.pred_x_hat = v.pred_x
        for c in self.mesh.cells:
            c.lambda_H_hat = c.lambda_H
            c.lambda_D_hat = c.lambda_D

    @ti.kernel
    def _rollback_to_hat(self):
        for v in self.mesh.verts:
            v.pred_x = v.pred_x_hat
        for c in self.mesh.cells:
            c.lambda_H = c.lambda_H_hat
            c.lambda_D = c.lambda_D_hat

    # ----------------------------------------------------------------------
    # Jacobi helper kernels
    # ----------------------------------------------------------------------
    @ti.kernel
    def pre_solve(self):
        """Reset per-vertex accumulators at the beginning of each iteration"""
        for v in self.mesh.verts:
            v.delta_x.fill(0.0)
            v.cons_num = 0

    @ti.kernel
    def post_solve(self):
        """Average and apply Jacobi position corrections (delta_x / cons_num)"""
        for v in self.mesh.verts:
            if v.cons_num > 0:
                v.pred_x += v.delta_x / v.cons_num

    # ----------------------------------------------------------------------
    # Chebyshev cache helpers
    # ----------------------------------------------------------------------
    @ti.kernel
    def _save_curr_to_prev(self):
        """y_{k-1} <- y_k (used at the loop start and for warmup alignment)"""
        for v in self.mesh.verts:
            v.pred_x_prev = v.pred_x
        for c in self.mesh.cells:
            c.lambda_H_prev = c.lambda_H
            c.lambda_D_prev = c.lambda_D

    @ti.kernel
    def _save_curr_to_savebuf(self):
        """savebuf <- y_k (store the state before the current sweep)"""
        for v in self.mesh.verts:
            v.pred_x_curr = v.pred_x
        for c in self.mesh.cells:
            c.lambda_H_save = c.lambda_H
            c.lambda_D_save = c.lambda_D

    @ti.kernel
    def _chebyshev_update(self, omega: ti.f32):
        """
        Chebyshev outer update:
            y^{k+1} = y^{k-1} + omega * (hat{y}^{k+1} - y^{k-1})
        Apply to both positions (x) and multipliers (lambda).
        """
        for v in self.mesh.verts:
            x_new = self.cheb_gamma * \
                (v.pred_x - v.pred_x_curr) + v.pred_x_curr
            v.pred_x = v.pred_x_prev + omega * (x_new - v.pred_x_prev)
        # for c in self.mesh.cells:
        #     c.lambda_H = c.lambda_H_prev + omega * \
        #         (c.lambda_H - c.lambda_H_prev)
        #     c.lambda_D = c.lambda_D_prev + omega * \
        #         (c.lambda_D - c.lambda_D_prev)

    @ti.kernel
    def _rotate_prev_with_save(self):
        """
        y_{k-1} <- saved y_k
        (finish the rotation so the next iteration uses the previous y_k as y_{k-1})
        """
        for v in self.mesh.verts:
            v.pred_x_prev = v.pred_x_curr
        for c in self.mesh.cells:
            c.lambda_H_prev = c.lambda_H_save
            c.lambda_D_prev = c.lambda_D_save

    # ----------------------------------------------------------------------
    # Deviatoric (isochoric) constraint: ||F||_F^2 - 3 = 0  with XPBD compliance
    # ----------------------------------------------------------------------
    @ti.kernel
    def solve_deviatoric(self, dt: ti.f32):
        """Solve deviatoric constraints for all tetrahedra (Jacobi accumulation)"""
        ti.loop_config(block_dim=self.block_size)
        ti.mesh_local(self.mesh.verts.delta_x, self.mesh.verts.inv_m,
                      self.mesh.verts.pred_x)

        for c in self.mesh.cells:
            p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
            w0, w1, w2, w3 = p0.inv_m, p1.inv_m, p2.inv_m, p3.inv_m

            # Predicted positions
            pos0, pos1, pos2, pos3 = p0.pred_x, p1.pred_x, p2.pred_x, p3.pred_x

            # Deformation gradient F = Ds * inv_Dm
            v1, v2, v3 = pos1 - pos0, pos2 - pos0, pos3 - pos0
            Ds = ti.Matrix.cols([v1, v2, v3])
            F = Ds @ c.inv_Dm

            # C = tr(F^T F) - 3 = ||F||_F^2 - 3
            trFTF = F.norm_sqr()
            C = trFTF - 3.0

            # Gradient wrt Ds is 2F; map back to vertex grads via inv_Dm^T
            grad_F = 2.0 * F
            grad_123 = grad_F @ c.inv_Dm.transpose()

            grad_1 = ti.Vector([grad_123[j, 0] for j in ti.static(range(3))])
            grad_2 = ti.Vector([grad_123[j, 1] for j in ti.static(range(3))])
            grad_3 = ti.Vector([grad_123[j, 2] for j in ti.static(range(3))])
            grad_0 = -(grad_1 + grad_2 + grad_3)

            # Weighted gradient norm sum w_sum = Σ w_i ||∂C/∂x_i||^2
            w_sum = w0 * grad_0.dot(grad_0)
            w_sum += w1 * grad_1.dot(grad_1)
            w_sum += w2 * grad_2.dot(grad_2)
            w_sum += w3 * grad_3.dot(grad_3)

            if w_sum > 1e-6:
                # XPBD compliance (alpha_D = 1 /(mu * V0)), scaled by dt^2
                alpha_D = 1.0 / (self.lame2 * c.V0)
                alpha_tilde = alpha_D / (dt ** 2)

                # XPBD local multiplier increment (Jacobi-style)
                d_lambda = -(C + alpha_tilde * c.lambda_D) / \
                    (w_sum + alpha_tilde)
                d_lambda *= self.deviatoric_relaxation
                c.lambda_D += d_lambda

                # Apply PD-style position corrections (scaled by inverse mass)
                ti.atomic_add(p0.delta_x, d_lambda * w0 * grad_0)
                ti.atomic_add(p1.delta_x, d_lambda * w1 * grad_1)
                ti.atomic_add(p2.delta_x, d_lambda * w2 * grad_2)
                ti.atomic_add(p3.delta_x, d_lambda * w3 * grad_3)

                # Count constraints per vertex (for averaging in post_solve)
                ti.atomic_add(p0.cons_num, 1)
                ti.atomic_add(p1.cons_num, 1)
                ti.atomic_add(p2.cons_num, 1)
                ti.atomic_add(p3.cons_num, 1)

    # ----------------------------------------------------------------------
    # Hydrostatic (volume) constraint: det(F) - 1 = 0  with XPBD compliance
    # ----------------------------------------------------------------------
    @ti.kernel
    def solve_hydrostatic(self, dt: ti.f32):
        """Solve hydrostatic constraints for all tetrahedra (Jacobi accumulation)"""
        ti.loop_config(block_dim=self.block_size)
        ti.mesh_local(self.mesh.verts.delta_x, self.mesh.verts.inv_m,
                      self.mesh.verts.pred_x)

        for c in self.mesh.cells:
            p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
            w0, w1, w2, w3 = p0.inv_m, p1.inv_m, p2.inv_m, p3.inv_m

            # Predicted positions
            pos0, pos1, pos2, pos3 = p0.pred_x, p1.pred_x, p2.pred_x, p3.pred_x

            # Deformation gradient F = Ds * inv_Dm
            v1, v2, v3 = pos1 - pos0, pos2 - pos0, pos3 - pos0
            Ds = ti.Matrix.cols([v1, v2, v3])
            F = Ds @ c.inv_Dm

            detF = F.determinant()
            C = detF - 1.0

            # Gradient of det(F) w.r.t F is adj(F)^T; build column-wise
            F_col0 = ti.Vector([F[j, 0] for j in ti.static(range(3))])
            F_col1 = ti.Vector([F[j, 1] for j in ti.static(range(3))])
            F_col2 = ti.Vector([F[j, 2] for j in ti.static(range(3))])

            cross0 = F_col1.cross(F_col2)
            cross1 = F_col2.cross(F_col0)
            cross2 = F_col0.cross(F_col1)
            grad_F = ti.Matrix.cols([cross0, cross1, cross2])

            # Map gradient to vertices via inv_Dm^T
            grad_123 = grad_F @ c.inv_Dm.transpose()

            grad_1 = ti.Vector([grad_123[j, 0] for j in ti.static(range(3))])
            grad_2 = ti.Vector([grad_123[j, 1] for j in ti.static(range(3))])
            grad_3 = ti.Vector([grad_123[j, 2] for j in ti.static(range(3))])
            grad_0 = -(grad_1 + grad_2 + grad_3)

            # Weighted gradient norm sum
            w_sum = w0 * grad_0.dot(grad_0)
            w_sum += w1 * grad_1.dot(grad_1)
            w_sum += w2 * grad_2.dot(grad_2)
            w_sum += w3 * grad_3.dot(grad_3)

            if w_sum > 1e-6:
                # XPBD compliance (alpha_H = 1 /(lambda * V0)), scaled by dt^2
                alpha_H = 1.0 / (self.lame1 * c.V0)
                alpha_tilde = alpha_H / (dt ** 2)

                # XPBD local multiplier increment (Jacobi-style)
                d_lambda = -(C + alpha_tilde * c.lambda_H) / \
                    (w_sum + alpha_tilde)
                d_lambda *= self.hydrostatic_relaxation
                c.lambda_H += d_lambda

                # Apply PD-style position corrections
                ti.atomic_add(p0.delta_x, d_lambda * w0 * grad_0)
                ti.atomic_add(p1.delta_x, d_lambda * w1 * grad_1)
                ti.atomic_add(p2.delta_x, d_lambda * w2 * grad_2)
                ti.atomic_add(p3.delta_x, d_lambda * w3 * grad_3)

                # Count constraints per vertex
                ti.atomic_add(p0.cons_num, 1)
                ti.atomic_add(p1.cons_num, 1)
                ti.atomic_add(p2.cons_num, 1)
                ti.atomic_add(p3.cons_num, 1)

    # ----------------------------------------------------------------------
    # Convergence monitoring: compute constraint errors
    # ----------------------------------------------------------------------
    @ti.kernel
    def compute_hydrostatic_error(self):
        """Compute total hydrostatic constraint error |C_H| = |det(F) - 1|"""
        self.hydro_error_sum[None] = 0.0
        self.constraint_count[None] = 0

        ti.loop_config(block_dim=self.block_size)
        ti.mesh_local(self.mesh.verts.pred_x)

        for c in self.mesh.cells:
            p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
            pos0, pos1, pos2, pos3 = p0.pred_x, p1.pred_x, p2.pred_x, p3.pred_x

            # Deformation gradient F = Ds * inv_Dm
            v1, v2, v3 = pos1 - pos0, pos2 - pos0, pos3 - pos0
            Ds = ti.Matrix.cols([v1, v2, v3])
            F = Ds @ c.inv_Dm

            detF = F.determinant()
            C_H = detF - 1.0

            # Accumulate absolute error
            ti.atomic_add(self.hydro_error_sum[None], ti.abs(C_H))
            ti.atomic_add(self.constraint_count[None], 1)

    @ti.kernel
    def compute_deviatoric_error(self):
        """Compute total deviatoric constraint error |C_D| = ||F||_F^2 - 3|"""
        self.dev_error_sum[None] = 0.0

        ti.loop_config(block_dim=self.block_size)
        ti.mesh_local(self.mesh.verts.pred_x)

        for c in self.mesh.cells:
            p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
            pos0, pos1, pos2, pos3 = p0.pred_x, p1.pred_x, p2.pred_x, p3.pred_x

            # Deformation gradient F = Ds * inv_Dm
            v1, v2, v3 = pos1 - pos0, pos2 - pos0, pos3 - pos0
            Ds = ti.Matrix.cols([v1, v2, v3])
            F = Ds @ c.inv_Dm

            # C = tr(F^T F) - 3 = ||F||_F^2 - 3
            trFTF = F.norm_sqr()
            C_D = trFTF - 3.0

            # Accumulate absolute error
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

            # L2 residual (per tet)
            r = ti.sqrt(rD * rD + rH * rH)

            ti.atomic_add(self.residual_sum[None], r)
            ti.atomic_add(self.residual_count[None], 1)

    @ti.func
    def solve2x2_closed_form_pbat(self, A11, A12, A22, b1, b2):
        # Matches PBAT mini::Inverse<2x2> path: dlam = Inverse(A) * b
        inv_det = 1.0 / (A11 * A22 - A12 * A12)
        dlamD = inv_det * (A22 * b1 - A12 * b2)
        dlamH = inv_det * (-A12 * b1 + A11 * b2)
        return dlamD, dlamH

    @ti.kernel
    def solve_neohookean_block_pbat(self, dt: ti.f32):
        ti.loop_config(block_dim=self.block_size)
        ti.mesh_local(self.mesh.verts.delta_x,
                      self.mesh.verts.inv_m, self.mesh.verts.pred_x)

        eps_C = 1e-8
        eps_norm = 1e-12

        for c in self.mesh.cells:
            p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
            w0, w1, w2, w3 = p0.inv_m, p1.inv_m, p2.inv_m, p3.inv_m

            x0, x1, x2, x3 = p0.pred_x, p1.pred_x, p2.pred_x, p3.pred_x
            Ds = ti.Matrix.cols([x1 - x0, x2 - x0, x3 - x0])
            F = Ds @ c.inv_Dm

            # PBAT style constraints: CD = ||F||_F, CH = det(F) - gamma_rest
            CD = ti.sqrt(F.norm_sqr() + eps_norm)
            detF = F.determinant()
            gamma_rest = 1.0 + self.lame2 / self.lame1
            CH = detF - gamma_rest

            need_solve = (ti.abs(CD) >= eps_C) or (ti.abs(CH) >= eps_C)
            if need_solve:
                invDmT = c.inv_Dm.transpose()

                gradF_D = F / CD
                grad123_D = gradF_D @ invDmT
                gD1 = ti.Vector(
                    [grad123_D[0, 0], grad123_D[1, 0], grad123_D[2, 0]])
                gD2 = ti.Vector(
                    [grad123_D[0, 1], grad123_D[1, 1], grad123_D[2, 1]])
                gD3 = ti.Vector(
                    [grad123_D[0, 2], grad123_D[1, 2], grad123_D[2, 2]])
                gD0 = -(gD1 + gD2 + gD3)

                F0 = ti.Vector([F[0, 0], F[1, 0], F[2, 0]])
                F1 = ti.Vector([F[0, 1], F[1, 1], F[2, 1]])
                F2 = ti.Vector([F[0, 2], F[1, 2], F[2, 2]])
                gradF_H = ti.Matrix.cols(
                    [F1.cross(F2), F2.cross(F0), F0.cross(F1)])
                grad123_H = gradF_H @ invDmT
                gH1 = ti.Vector(
                    [grad123_H[0, 0], grad123_H[1, 0], grad123_H[2, 0]])
                gH2 = ti.Vector(
                    [grad123_H[0, 1], grad123_H[1, 1], grad123_H[2, 1]])
                gH3 = ti.Vector(
                    [grad123_H[0, 2], grad123_H[1, 2], grad123_H[2, 2]])
                gH0 = -(gH1 + gH2 + gH3)

                alpha_D = 1.0 / (self.lame2 * c.V0)  # 1/(mu * V0)
                alpha_H = 1.0 / (self.lame1 * c.V0)  # 1/(lambda * V0)
                aD = alpha_D / (dt * dt)
                aH = alpha_H / (dt * dt)

                A11 = w0 * gD0.dot(gD0) + w1 * gD1.dot(gD1) + \
                    w2 * gD2.dot(gD2) + w3 * gD3.dot(gD3) + aD
                A22 = w0 * gH0.dot(gH0) + w1 * gH1.dot(gH1) + \
                    w2 * gH2.dot(gH2) + w3 * gH3.dot(gH3) + aH
                A12 = w0 * gD0.dot(gH0) + w1 * gD1.dot(gH1) + \
                    w2 * gD2.dot(gH2) + w3 * gD3.dot(gH3)

                b1 = -(CD + aD * c.lambda_D)
                b2 = -(CH + aH * c.lambda_H)

                dlamD, dlamH = self.solve2x2_closed_form_pbat(
                    A11, A12, A22, b1, b2)

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

    @ti.func
    def solve2x2_pivot_lu(self, A00, A01, A10, A11, b0, b1, eps):
        # returns: ok (i32), x0, x1
        ok = 1

        # partial pivoting on first column
        if ti.abs(A00) < ti.abs(A10):
            # swap rows
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
        sqrt3 = 1.7320508075688772

        for c in self.mesh.cells:
            p0, p1, p2, p3 = c.verts[0], c.verts[1], c.verts[2], c.verts[3]
            w0, w1, w2, w3 = p0.inv_m, p1.inv_m, p2.inv_m, p3.inv_m

            pos0, pos1, pos2, pos3 = p0.pred_x, p1.pred_x, p2.pred_x, p3.pred_x

            v1, v2, v3 = pos1 - pos0, pos2 - pos0, pos3 - pos0
            Ds = ti.Matrix.cols([v1, v2, v3])
            F = Ds @ c.inv_Dm

            trFTF = F.norm_sqr()
            # normF = ti.sqrt(trFTF + eps_A)
            # normF_inv = 1.0 / normF
            # C_D = normF - sqrt3

            C_D = trFTF - 3.0

            detF = F.determinant()
            C_H = detF - 1.0

            invDmT = c.inv_Dm.transpose()

            # --- deviatoric gradients ---
            # gradF_D = F * normF_inv
            gradF_D = F * 2.0
            grad123_D = gradF_D @ invDmT
            gD1 = ti.Vector([grad123_D[j, 0] for j in ti.static(range(3))])
            gD2 = ti.Vector([grad123_D[j, 1] for j in ti.static(range(3))])
            gD3 = ti.Vector([grad123_D[j, 2] for j in ti.static(range(3))])
            gD0 = -(gD1 + gD2 + gD3)

            # --- hydro gradients (adj(F)^T) ---
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

            # --- build A ---
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

            # detA = A11 * A22 - A12 * A12

            # if ti.abs(detA) >= eps_A:
            #     b1 = -(C_D + aD * c.lambda_D)
            #     b2 = -(C_H + aH * c.lambda_H)

            #     dlamD = (b1 * A22 - b2 * A12) / detA
            #     dlamH = (-b1 * A12 + b2 * A11) / detA

            #     dlamD *= self.deviatoric_relaxation
            #     dlamH *= self.hydrostatic_relaxation

            #     c.lambda_D += dlamD
            #     c.lambda_H += dlamH

            #     dp0 = w0 * (gD0 * dlamD + gH0 * dlamH)
            #     dp1 = w1 * (gD1 * dlamD + gH1 * dlamH)
            #     dp2 = w2 * (gD2 * dlamD + gH2 * dlamH)
            #     dp3 = w3 * (gD3 * dlamD + gH3 * dlamH)

            #     ti.atomic_add(p0.delta_x, dp0)
            #     ti.atomic_add(p1.delta_x, dp1)
            #     ti.atomic_add(p2.delta_x, dp2)
            #     ti.atomic_add(p3.delta_x, dp3)

            #     ti.atomic_add(p0.cons_num, 1)
            #     ti.atomic_add(p1.cons_num, 1)
            #     ti.atomic_add(p2.cons_num, 1)
            #     ti.atomic_add(p3.cons_num, 1)

    # ----------------------------------------------------------------------
    # One XPBD substep with Chebyshev-accelerated Jacobi outer loop
    # ----------------------------------------------------------------------

    def solve_step(self, dt):
        """
        Perform one XPBD substep.
        We treat each (pre_solve -> solve_hydrostatic -> solve_deviatoric -> post_solve)
        as one "unit iteration" hat{y}^{k+1} and wrap a Chebyshev semi-iterative
        outer update: y^{k+1} = y^{k-1} + omega (hat{y}^{k+1} - y^{k-1}).
        """
        compute_time_acc = 0.0
        setup_time_start = time.perf_counter()

        # Reset XPBD Lagrange multipliers every substep
        self.mesh.cells.lambda_H.fill(0.0)
        self.mesh.cells.lambda_D.fill(0.0)

        self.prev_r_hat = None

        # Collisions (project predicted positions)
        self.solve_sdf_collision(dt)

        # ---- Chebyshev coefficients init ----
        use_cheb = self.cheb_enable
        omega = 1.0  # omega_1
        need_hat_snapshot = use_cheb and self.cheb_safeguard
        need_rho_inst = self.dynamic_cheb_rho or self.monitor_rho
        need_hat_residual = need_rho_inst or self.cheb_safeguard

        # Align y_{-1} with y_0 before starting the loop (important!)
        if use_cheb:
            self._save_curr_to_prev()

        compute_time_acc += time.perf_counter() - setup_time_start

        # Run 'constraints_iter' Jacobi sweeps
        for it in range(self.constraints_iter):
            iter_compute_time_sec = 0.0
            iter_compute_start = time.perf_counter()

            # Save current y_k before the unit sweep (for rotating prev later)
            if use_cheb:
                self._save_curr_to_savebuf()

            # ---- Unit iteration (your original Jacobi pipeline) ----
            self.pre_solve()

            if self.neohookean_block_enable:
                self.solve_neohookean_block(dt)
                # self.solve_neohookean_block_pbat(dt)
            else:
                self.solve_hydrostatic(dt)
                self.solve_deviatoric(dt)

            self.post_solve()  # now fields hold hat{y}^{k+1}

            if need_hat_snapshot:
                # save hat state for possible rollback
                self._save_hat()

            r_hat = None
            rho_inst = None
            if need_hat_residual:
                # compute residual on hat
                self.compute_xpbd_residual(dt)
                cnt = self.residual_count[None]
                r_hat = self.residual_sum[None] / cnt if cnt > 0 else 0.0

                if need_rho_inst:
                    if self.prev_r_hat is not None and self.prev_r_hat > 1e-12:
                        rho_inst = r_hat / (self.prev_r_hat + 1e-12)
                        rho_inst = max(self.rho_min, min(
                            self.rho_max, float(rho_inst)))
                        if self.dynamic_cheb_rho:
                            self.rho_hat = (1.0 - self.rho_beta) * \
                                self.rho_hat + self.rho_beta * rho_inst
                    self.prev_r_hat = r_hat

            iter_compute_time_sec += time.perf_counter() - iter_compute_start

            # ---- Convergence monitoring (after each iteration) ----
            convergence_record = None
            if self.monitor_convergence:
                # Compute constraint errors
                self.compute_hydrostatic_error()
                self.compute_deviatoric_error()

                # Get errors from device
                num_constraints = self.constraint_count[None]
                if num_constraints > 0:
                    hydro_error = self.hydro_error_sum[None] / num_constraints
                    dev_error = self.dev_error_sum[None] / num_constraints
                else:
                    hydro_error = 0.0
                    dev_error = 0.0

                # Record data
                record = {
                    'frame': self.current_frame,
                    'substep': self.current_substep,
                    'iter': it,
                    'iter_compute_time_sec': iter_compute_time_sec,
                    'hydro_error': hydro_error,
                    'dev_error': dev_error
                }

                # Add to full history (for statistics)
                self.convergence_history.append(record)

                # Add to current substep data (for plotting after substep completes)
                self.current_substep_convergence.append(record)
                convergence_record = record

            # ---- Chebyshev outer update ----
            if use_cheb:
                iter_compute_start = time.perf_counter()

                rho = float(self.rho_hat) if self.dynamic_cheb_rho else float(
                    self.cheb_rho)
                rho2 = rho * rho

                # Warmup: keep omega=1 for the first S iterations
                if it < self.cheb_warmup:
                    omega = 1.0
                else:
                    # First accelerated step uses omega = 2/(2 - rho^2)
                    if it == self.cheb_warmup:
                        omega = 2.0 / (2.0 - rho2)
                    else:
                        # Recurrence: omega_{k+1} = 4 / (4 - rho^2 * omega_k)
                        omega = 4.0 / (4.0 - rho2 * omega)

                # y^{k+1} = y^{k-1} + omega * (hat{y}^{k+1} - y^{k-1})
                self._chebyshev_update(omega)

                if self.cheb_safeguard:
                    # compute residual after chebyshev extrapolation
                    self.compute_xpbd_residual(dt)
                    cnt = self.residual_count[None]
                    r_y = self.residual_sum[None] / cnt if cnt > 0 else 0.0
                    if r_y > (1.0 + self.cheb_tol) * r_hat:
                        # rollback to hat (reject extrapolation)
                        self._rollback_to_hat()

                        # also slow down future steps automatically
                        if self.dynamic_cheb_rho and hasattr(self, "rho_hat"):
                            self.rho_hat = max(
                                self.rho_min, 0.5 * self.rho_hat)

                # Rotate prev: y_{k-1} <- saved y_k
                self._rotate_prev_with_save()

                iter_compute_time_sec += time.perf_counter() - iter_compute_start

                if convergence_record is not None:
                    convergence_record['iter_compute_time_sec'] = iter_compute_time_sec

            # ---- Rho tracking (after each iteration) ----
            if self.monitor_rho:
                omega_used = float(omega) if use_cheb else None
                record = {
                    'frame': self.current_frame,
                    'substep': self.current_substep,
                    'iter': it,
                    'iter_compute_time_sec': iter_compute_time_sec,
                    'rho_hat': float(self.rho_hat),
                    'r_hat': float(r_hat) if r_hat is not None else None,
                    'rho_inst': float(rho_inst) if rho_inst is not None else None,
                    'omega': omega_used
                }
                self.rho_history.append(record)
                self.current_substep_rho.append(record)
            compute_time_acc += iter_compute_time_sec

        self.solve_compute_time_total += compute_time_acc
        self.solve_compute_calls += 1
        self.solve_compute_iterations += self.constraints_iter

    # ----------------------------------------------------------------------
    # Override solve to track frame and substep for convergence monitoring
    # ----------------------------------------------------------------------
    def solve(self):
        """Main solve loop - advances simulation by one frame"""
        frame_time_left = self.frame_dt
        self.current_substep = 0

        while frame_time_left > 0.0:
            dt0 = min(self.dt, frame_time_left)
            frame_time_left -= dt0

            sub_dt = dt0 / self.substeps_num
            for substep in range(self.substeps_num):
                self.current_substep = substep

                # Clear current substep convergence data
                if self.monitor_convergence:
                    self.current_substep_convergence = []
                if self.monitor_rho:
                    self.current_substep_rho = []

                self.apply_external_forces(sub_dt)
                self.solve_step(sub_dt)
                self.advance(sub_dt, self.damping)

                # After substep completes, call callback to plot this substep
                if self.monitor_convergence and hasattr(self, 'convergence_callback') and self.convergence_callback:
                    self.convergence_callback(self.current_frame, self.current_substep,
                                              self.current_substep_convergence)
                if self.monitor_rho and hasattr(self, 'rho_callback') and self.rho_callback:
                    self.rho_callback(self.current_frame, self.current_substep,
                                      self.current_substep_rho)

            self.time += dt0

        # Increment frame counter after solving
        self.current_frame += 1

    # ----------------------------------------------------------------------
    # Convergence visualization
    # ----------------------------------------------------------------------
    def plot_substep_convergence(self, frame, substep, convergence_data, save_path, log_scale=False, export_data=True):
        """
        Plot convergence for one substep.

        Args:
            frame: Frame index
            substep: Substep index
            convergence_data: List of dicts with convergence records for this substep
            save_path: Path to save the plot
            log_scale: Use logarithmic scale for y-axis
            export_data: Whether to export convergence data to JSON file
        """
        if not convergence_data:
            return

        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
        import json
        from pathlib import Path

        # Extract data for this substep
        iterations = [d['iter'] for d in convergence_data]
        iter_compute_times_sec = [d['iter_compute_time_sec']
                                  for d in convergence_data]
        hydro_errors = [d['hydro_error'] for d in convergence_data]
        dev_errors = [d['dev_error'] for d in convergence_data]

        # Export data to JSON if requested
        if export_data:
            data_path = Path(save_path).with_suffix('.json')
            export_dict = {
                'frame': frame,
                'substep': substep,
                'iterations': iterations,
                'iter_compute_times_sec': iter_compute_times_sec,
                'hydro_errors': hydro_errors,
                'dev_errors': dev_errors,
                'records': convergence_data,
                'solver_type': self.__class__.__name__,
                'compute_timing': self.get_compute_timing_stats(),
                'config': {
                    'constraints_iter': self.constraints_iter,
                    'hydrostatic_relaxation': self.hydrostatic_relaxation,
                    'deviatoric_relaxation': self.deviatoric_relaxation,
                    'cheb_enable': self.cheb_enable,
                    'cheb_rho': self.cheb_rho,
                    'dynamic_cheb_rho': self.dynamic_cheb_rho,
                    'cheb_gamma': self.cheb_gamma,
                    'cheb_warmup': self.cheb_warmup,
                }
            }
            with open(data_path, 'w') as f:
                json.dump(export_dict, f, indent=2)

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot hydrostatic error
        ax1.plot(iterations, hydro_errors, 'o-',
                 color='#2E86AB', linewidth=2, markersize=5, alpha=0.8)
        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('Hydrostatic Error |det(F) - 1|', fontsize=11)
        ax1.set_title(
            f'Frame {frame}, Substep {substep} - Hydrostatic', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')

        if log_scale and len(hydro_errors) > 0 and max(hydro_errors) > 0:
            ax1.set_yscale('log')
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            ax1.yaxis.set_major_formatter(formatter)
        else:
            # Linear scale with more ticks
            ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
            formatter = ticker.ScalarFormatter(
                useMathText=True, useOffset=False)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-3, 3))
            ax1.yaxis.set_major_formatter(formatter)

        # Add info text
        if len(hydro_errors) > 1:
            initial = hydro_errors[0]
            final = hydro_errors[-1]
            reduction = initial / final if final > 0 else float('inf')
            ax1.text(0.95, 0.95,
                     f'Initial: {initial:.2e}\n'
                     f'Final: {final:.2e}\n'
                     f'Reduction: {reduction:.1f}x',
                     transform=ax1.transAxes, fontsize=9,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # Plot deviatoric error
        ax2.plot(iterations, dev_errors, 's-',
                 color='#A23B72', linewidth=2, markersize=5, alpha=0.8)
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('Deviatoric Error ||F||² - 3|', fontsize=11)
        ax2.set_title(
            f'Frame {frame}, Substep {substep} - Deviatoric', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')

        if log_scale and len(dev_errors) > 0 and max(dev_errors) > 0:
            ax2.set_yscale('log')
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            ax2.yaxis.set_major_formatter(formatter)
        else:
            # Linear scale with more ticks
            ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
            formatter = ticker.ScalarFormatter(
                useMathText=True, useOffset=False)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-3, 3))
            ax2.yaxis.set_major_formatter(formatter)

        # Add info text
        if len(dev_errors) > 1:
            initial = dev_errors[0]
            final = dev_errors[-1]
            reduction = initial / final if final > 0 else float('inf')
            ax2.text(0.95, 0.95,
                     f'Initial: {initial:.2e}\n'
                     f'Final: {final:.2e}\n'
                     f'Reduction: {reduction:.1f}x',
                     transform=ax2.transAxes, fontsize=9,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_convergence_summary(self, save_path='convergence_summary.png'):
        """
        Plot summary of convergence across all substeps (final errors).

        Args:
            save_path: Path to save the plot
        """
        if not self.monitor_convergence or len(self.convergence_history) == 0:
            print("No convergence data to plot.")
            return

        import matplotlib.pyplot as plt
        import numpy as np

        # Extract final errors for each substep
        from collections import defaultdict
        substep_final_errors = defaultdict(lambda: {'hydro': [], 'dev': []})

        current_key = None
        for record in self.convergence_history:
            key = (record['frame'], record['substep'])
            if key != current_key:
                current_key = key

        # Get last iteration for each substep
        grouped = defaultdict(list)
        for record in self.convergence_history:
            key = (record['frame'], record['substep'])
            grouped[key].append(record)

        frames = []
        substeps = []
        hydro_finals = []
        dev_finals = []

        for (frame, substep), records in sorted(grouped.items()):
            frames.append(frame)
            substeps.append(substep)
            hydro_finals.append(records[-1]['hydro_error'])
            dev_finals.append(records[-1]['dev_error'])

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Hydrostatic final errors
        x_labels = [f'F{f}S{s}' for f, s in zip(frames, substeps)]
        x_pos = np.arange(len(x_labels))

        ax1.bar(x_pos, hydro_finals, color='#2E86AB',
                alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Final Hydrostatic Error', fontsize=11)
        ax1.set_title('Final Constraint Errors per Substep',
                      fontsize=13, fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_xticks(x_pos[::max(1, len(x_pos)//20)])
        ax1.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), max(1, len(x_pos)//20))],
                            rotation=45, ha='right', fontsize=8)

        # Deviatoric final errors
        ax2.bar(x_pos, dev_finals, color='#A23B72',
                alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Frame and Substep', fontsize=11)
        ax2.set_ylabel('Final Deviatoric Error', fontsize=11)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticks(x_pos[::max(1, len(x_pos)//20)])
        ax2.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), max(1, len(x_pos)//20))],
                            rotation=45, ha='right', fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Convergence summary plot saved to: {save_path}")
        plt.close()

    def get_convergence_stats(self):
        """Get statistics about constraint convergence"""
        if not self.monitor_convergence or len(self.convergence_history) == 0:
            return None

        import numpy as np

        # Group by frame-substep-iter
        final_errors = {}
        for entry in self.convergence_history:
            key = (entry['frame'], entry['substep'])
            if key not in final_errors:
                final_errors[key] = {'hydro': [], 'dev': []}
            final_errors[key]['hydro'].append(entry['hydro_error'])
            final_errors[key]['dev'].append(entry['dev_error'])

        # Compute final (last iteration) errors for each substep
        final_hydro = []
        final_dev = []
        for key, errors in final_errors.items():
            final_hydro.append(errors['hydro'][-1])
            final_dev.append(errors['dev'][-1])

        stats = {
            'total_records': len(self.convergence_history),
            'num_substeps': len(final_errors),
            'hydro': {
                'mean_final': np.mean(final_hydro),
                'max_final': np.max(final_hydro),
                'min_final': np.min(final_hydro),
            },
            'dev': {
                'mean_final': np.mean(final_dev),
                'max_final': np.max(final_dev),
                'min_final': np.min(final_dev),
            }
        }

        return stats

    def get_compute_timing_stats(self):
        """Get compute-only timing stats collected from solve_step."""
        avg_solve_step = self.solve_compute_time_total / \
            self.solve_compute_calls if self.solve_compute_calls > 0 else 0.0
        avg_iteration = self.solve_compute_time_total / \
            self.solve_compute_iterations if self.solve_compute_iterations > 0 else 0.0
        return {
            'solve_step_compute_count': self.solve_compute_calls,
            'constraint_iteration_count': self.solve_compute_iterations,
            'total_compute_time_sec': self.solve_compute_time_total,
            'avg_compute_time_per_solve_step_sec': avg_solve_step,
            'avg_compute_time_per_iteration_sec': avg_iteration,
        }

    def plot_substep_rho(self, frame, substep, rho_data, save_path, export_data=True):
        """
        Plot rho_hat, rho_inst, r_hat, and omega vs iteration for one substep.

        Args:
            frame: Frame index
            substep: Substep index
            rho_data: List of dicts with rho records for this substep
            save_path: Path to save the plot
            export_data: Whether to export rho data to JSON file
        """
        if not rho_data:
            return

        import matplotlib.pyplot as plt
        import numpy as np
        import json
        from pathlib import Path

        iterations = [d['iter'] for d in rho_data]
        iter_compute_times_sec = [d['iter_compute_time_sec']
                                  for d in rho_data]
        rho_hat_vals = [d['rho_hat'] for d in rho_data]
        rho_inst_vals = [d['rho_inst'] for d in rho_data]
        r_hat_vals = [d['r_hat'] for d in rho_data]
        omega_vals = [d['omega'] for d in rho_data]

        if export_data:
            data_path = Path(save_path).with_suffix('.json')
            export_dict = {
                'frame': frame,
                'substep': substep,
                'iterations': iterations,
                'iter_compute_times_sec': iter_compute_times_sec,
                'rho_hat': rho_hat_vals,
                'rho_inst': rho_inst_vals,
                'r_hat': r_hat_vals,
                'omega': omega_vals,
                'records': rho_data,
                'solver_type': self.__class__.__name__,
                'compute_timing': self.get_compute_timing_stats(),
                'config': {
                    'constraints_iter': self.constraints_iter,
                    'cheb_enable': self.cheb_enable,
                    'cheb_rho': self.cheb_rho,
                    'dynamic_cheb_rho': self.dynamic_cheb_rho,
                    'cheb_gamma': self.cheb_gamma,
                    'cheb_warmup': self.cheb_warmup,
                    'rho_min': self.rho_min,
                    'rho_max': self.rho_max,
                    'rho_beta': self.rho_beta,
                }
            }
            with open(data_path, 'w') as f:
                json.dump(export_dict, f, indent=2)

        base_path = Path(save_path)
        stem = base_path.stem
        suffix = base_path.suffix if base_path.suffix else '.png'

        def save_single_plot(y_vals, title, y_label, out_name, color, marker):
            fig, ax = plt.subplots(1, 1, figsize=(7, 4))
            ax.plot(iterations, y_vals, marker + '-',
                    color=color, linewidth=2, markersize=4, alpha=0.9)
            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(out_name, dpi=150, bbox_inches='tight')
            plt.close()

        save_single_plot(
            rho_hat_vals,
            f'Frame {frame}, Substep {substep} - rho_hat',
            'rho_hat',
            base_path.with_name(f'{stem}_rho_hat{suffix}'),
            '#2E86AB',
            'o')

        if any(v is not None for v in rho_inst_vals):
            rho_inst_plot = [np.nan if v is None else v for v in rho_inst_vals]
            save_single_plot(
                rho_inst_plot,
                f'Frame {frame}, Substep {substep} - rho_inst',
                'rho_inst',
                base_path.with_name(f'{stem}_rho_inst{suffix}'),
                '#A23B72',
                's')

        save_single_plot(
            r_hat_vals,
            f'Frame {frame}, Substep {substep} - r_hat',
            'r_hat',
            base_path.with_name(f'{stem}_r_hat{suffix}'),
            '#F4A261',
            '^')

        if any(v is not None for v in omega_vals):
            omega_plot = [np.nan if v is None else v for v in omega_vals]
            save_single_plot(
                omega_plot,
                f'Frame {frame}, Substep {substep} - omega',
                'omega',
                base_path.with_name(f'{stem}_omega{suffix}'),
                '#6A994E',
                'd')
