import taichi as ti
import meshtaichi_patcher as Patcher
from xpbd_gpu.constants import EPSILON


@ti.func
def calc_tet_volume(p0, p1, p2, p3):
    return (p1 - p0).cross(p2 - p0).dot(p3 - p0) / 6.0


@ti.data_oriented
class XPBDSolverBase:
    """Base class for deformable object solvers"""

    def __init__(self,
                 rest_pose,
                 sdf,
                 scale=1.0,
                 offset=(0.0, 0.0, 0.0),
                 frame_dt=1e-2,
                 dt=1e-2,
                 substeps_num=20,
                 reorder_all=False,
                 block_size=128):

        # Time stepping parameters
        self.time = 0.0
        self.frame_dt = frame_dt
        self.dt = dt
        self.substeps_num = substeps_num

        # Physics constants
        self.gravity = ti.Vector.field(3, ti.f32, shape=())
        self.gravity[None] = ti.Vector([0.0, -9.8, 0.0])
        self.damping = 5.0
        self.total_mass = 1.0

        # Performance parameters
        self.block_size = block_size

        # Load mesh with necessary relations
        self.mesh = Patcher.load_mesh(rest_pose, relations=["VV", "CV"])

        # Basic vertex data that all solvers need
        self.mesh.verts.place({
            'x': ti.math.vec3,      # current position
            'pred_x': ti.math.vec3,  # predicted position
            'v': ti.math.vec3,      # velocity
            'm': ti.f32,            # mass
            'inv_m': ti.f32,        # inverse mass
        }, reorder=reorder_all)

        # Initialize vertex positions from mesh
        self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())

        # Visualization indices for rendering tetrahedra
        self.indices = ti.field(
            dtype=ti.u32, shape=len(self.mesh.cells) * 4 * 3)

        # Collision detection
        if isinstance(sdf, (list, tuple)):
            self.sdfs = [s for s in sdf if s is not None]
        else:
            self.sdfs = [sdf] if sdf is not None else []

        if len(self.sdfs) == 0:
            raise ValueError("At least one SDF object must be provided.")

        # Backward-compat: keep a primary SDF handle.
        self.sdf = self.sdfs[0]

        # Initialize visualization indices
        self.initialize_visual_indices()

        # Let derived classes handle specific initialization
        self.initialize_specific(scale, offset)

    @ti.kernel
    def initialize_visual_indices(self):
        """Initialize indices for tetrahedral mesh visualization"""
        for c in self.mesh.cells:
            # Face indices for each tetrahedron (4 triangular faces)
            ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]
            for i in ti.static(range(4)):
                for j in ti.static(range(3)):
                    self.indices[(c.id * 4 + i) * 3 +
                                 j] = c.verts[ind[i][j]].id

    @ti.kernel
    def _compute_lumped_mass_kernel(self):
        """Compute vertex lumped masses based on tetrahedral volumes"""
        total_volume = 0.0

        # Reset masses
        for v in self.mesh.verts:
            v.m = 0.0

        # Accumulate mass from cells
        for c in self.mesh.cells:
            # Calculate cell volume
            vol = calc_tet_volume(c.verts[0].x, c.verts[1].x,
                                  c.verts[2].x, c.verts[3].x)

            if vol <= EPSILON:
                print(
                    f"Warning: Cell {c.id} has negative or zero volume; volume= {vol:.6f}")

            # Distribute cell mass equally to vertices
            one_fourth_volume = vol * 0.25
            ti.atomic_add(c.verts[0].m, one_fourth_volume)
            ti.atomic_add(c.verts[1].m, one_fourth_volume)
            ti.atomic_add(c.verts[2].m, one_fourth_volume)
            ti.atomic_add(c.verts[3].m, one_fourth_volume)

            total_volume += vol

        # Scale masses to achieve target total mass
        mass_scale = self.total_mass / total_volume

        for v in self.mesh.verts:
            v.m *= mass_scale
            v.inv_m = 1.0 / v.m if v.m > 0.0 else 0.0

    @ti.kernel
    def _apply_fixed_vertices_for_sdf(self, sdf: ti.template()):
        """Set inv_m=0 for vertices initially inside a fixed SDF."""
        for v in self.mesh.verts:
            if v.inv_m > 0.0:
                fixed, inside, dotnv, diff_vel, n = sdf.check(v.x, v.v)
                if fixed and inside:
                    v.inv_m = 0.0

    def compute_lumped_mass(self):
        """Compute masses, then apply fixed-point pinning from all configured SDFs."""
        self._compute_lumped_mass_kernel()
        for sdf in self.sdfs:
            self._apply_fixed_vertices_for_sdf(sdf)

    @ti.kernel
    def apply_external_forces(self, dt: ti.f32):
        """Apply external forces (gravity) and predict positions"""
        for v0 in self.mesh.verts:
            # Algorithm I line 2: predict y_tilde from x_n, v_n, and f_ext.
            if v0.inv_m > 0.0:
                v0.v += self.gravity[None] * dt
            v0.pred_x = v0.x + v0.v * dt

    @ti.kernel
    def advance(self, dt: ti.f32, damping: ti.f32):
        """Update positions and velocities with damping"""
        for v0 in self.mesh.verts:
            if v0.inv_m <= 0.0:
                v0.pred_x = v0.x  # Fixed vertices don't move
            else:
                # Algorithm I line 21: recover v_{n+1} from the projected position.
                raw_v = (v0.pred_x - v0.x) / dt
                # Apply exponential damping
                v0.v = raw_v * ti.exp(-dt * damping)
                # Update position
                v0.x = v0.pred_x

    @ti.kernel
    def _solve_sdf_collision_one(self, dt: ti.f32, sdf: ti.template()):
        """Handle collisions for a single signed distance field."""
        for v in self.mesh.verts:
            if v.inv_m > 0.0:
                fixed, inside, dotnv, diff_vel, n = sdf.check(v.pred_x, v.v)
                # `fixed` describes whether the obstacle itself is fixed/moving.
                # Collision projection should still apply for fixed obstacles.
                if inside:
                    penetration_depth = -sdf.dist(v.pred_x)
                    if penetration_depth > 0.0:
                        # Push vertex out of collision object
                        v.pred_x += n * penetration_depth

    def solve_sdf_collision(self, dt):
        """Handle collisions by sequentially projecting against each configured SDF."""
        for sdf in self.sdfs:
            self._solve_sdf_collision_one(dt, sdf)

    def solve(self):
        """Main solve loop - advances simulation by one frame"""
        frame_time_left = self.frame_dt
        while frame_time_left > 0.0:
            dt0 = min(self.dt, frame_time_left)
            frame_time_left -= dt0

            sub_dt = dt0 / self.substeps_num
            for _ in range(self.substeps_num):
                self.apply_external_forces(sub_dt)
                self.solve_step(sub_dt)
                self.advance(sub_dt, self.damping)

            self.time += dt0

    # Abstract methods to be implemented by derived classes
    def initialize_specific(self, scale, offset):
        """Initialize solver-specific data structures"""
        raise NotImplementedError(
            "Derived classes must implement initialize_specific")

    def solve_step(self, dt):
        """Perform one substep of the solver"""
        raise NotImplementedError("Derived classes must implement solve_step")
