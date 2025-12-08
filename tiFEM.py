# pyright: ignore[reportInvalidTypeForm]

import numpy as np

import taichi as ti

ti.init(arch=ti.cuda)

headless_render = True

E, nu = 5e4, 0.0
mu, la = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # lambda = 0
density = 1000.0
# dt = 1e-4
dt = 1e-2
g_current_time = 0.0
g_current_step = 0


n_verts = 4
dx = 1.0 
n_cells = 1
gravity_constant = -9.8

tet_indices = ti.Vector.field(4, dtype=ti.i32, shape=n_cells)
tri_indices = ti.Vector.field(3, dtype=ti.i32, shape=4*n_cells)
edge_indices = ti.Vector.field(2, dtype=ti.i32, shape=6*n_cells)
pos = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
pos0 = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
prev_pos = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
vel = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
force = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
mul_ans = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
mass = ti.field(dtype=ti.f32, shape=n_verts)

rest_matrix = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_cells)
rest_volume = ti.field(dtype=ti.f32, shape=n_cells)
deformation_gradient = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_cells)
PK1 = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_cells)

F_mul_ans = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
F_b = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
F_r0 = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
F_p0 = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)



@ti.kernel
def get_vertices():
    # set rest positions for a single tetrahedron
    pos0[0] = ti.Vector([0.0, 0.0, 0.0])
    pos0[1] = ti.Vector([dx, 0.0, 0.0])
    pos0[2] = ti.Vector([0.0, dx, 0.0])
    pos0[3] = ti.Vector([0.0, 0.0, dx])


@ti.kernel
def get_indices():
    # set topology for a single tetrahedron
    tet_indices[0] = ti.Vector([0, 1, 2, 3])
    
    # four triangle faces
    tri_indices[0] = ti.Vector([0, 1, 2])
    tri_indices[1] = ti.Vector([0, 1, 3])
    tri_indices[2] = ti.Vector([0, 2, 3])
    tri_indices[3] = ti.Vector([1, 2, 3])
    
    # six edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    edge_indices[0] = ti.Vector([0, 1])
    edge_indices[1] = ti.Vector([0, 2])
    edge_indices[2] = ti.Vector([0, 3])
    edge_indices[3] = ti.Vector([1, 2])
    edge_indices[4] = ti.Vector([1, 3])
    edge_indices[5] = ti.Vector([2, 3])

@ti.func
def Ds(verts):
    return ti.Matrix.cols([pos[verts[i]] - pos[verts[3]] for i in range(3)])

@ti.func
def ssvd(F):
    U, sig, V = ti.svd(F)
    if U.determinant() < 0:
        for i in ti.static(range(3)):
            U[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    if V.determinant() < 0:
        for i in ti.static(range(3)):
            V[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    return U, sig, V


@ti.func
def get_force_func(c, verts):
    F = Ds(verts) @ rest_matrix[c]
    deformation_gradient[c] = F # record deformation gradient
    U, sig, V = ssvd(F)
    R = U @ V.transpose()
    I = ti.Matrix.identity(ti.f32, 3)
    # Linear Elasticity
    # P = mu * (F+F.transpose()-2*I) + la * (F-I).trace()*I

    # StVK
    # E = 0.5 * (F.transpose() @ F - I)
    # P = F @ (2*mu*E+la*E.trace()*I)

    # CR(Corotational Linear)
    # P = 2 * mu * (F - R) + la * (R.transpose()@F - I).trace()*R

    # As-Rigid-As-Possible (same with CR when la=0)
    P = 2 * mu * (F - R)

    # Neo-Hookean
    # FT_inv = F.inverse().transpose()
    # J = F.determinant()
    # P = mu * (F - FT_inv) + la * ti.log(J) * FT_inv

    PK1[c] = P  # record first Piola-Kirchhoff stress

    H = -rest_volume[c] * P @ rest_matrix[c].transpose()
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            force[verts[i]][j] += H[j, i]
            force[verts[3]][j] -= H[j, i]


@ti.kernel  
def get_force():
    for c in tet_indices:
        get_force_func(c, tet_indices[c])
    for u in force:
        force[u].y += gravity_constant * mass[u]



@ti.kernel
def matmul_cell(ret: ti.template(), vel: ti.template()):
    for i in ret:
        ret[i] = vel[i] * mass[i]
    for c in tet_indices:
        verts = tet_indices[c]
        W_c = rest_volume[c]
        B_c = rest_matrix[c]
        for u in range(4):
            for d in range(3):
                dD = ti.Matrix.zero(ti.f32, 3, 3)
                if u == 3:
                    for j in range(3):
                        dD[d, j] = -1
                else:
                    dD[d, u] = 1
                dF = dD @ B_c
                dP = 2.0 * mu * dF
                dH = -W_c * dP @ B_c.transpose()
                for i in range(3):
                    for j in range(3):
                        tmp = vel[verts[i]][j] - vel[verts[3]][j]
                        ret[verts[u]][d] += -(dt**2) * dH[j, i] * tmp


@ti.kernel
def get_b():
    for i in F_b:
        F_b[i] = mass[i] * vel[i] + dt * force[i]



@ti.kernel
def add(ans: ti.template(), a: ti.template(), k: ti.f32, b: ti.template()):
    for i in ans:
        ans[i] = a[i] + k * b[i]


@ti.kernel
def dot(a: ti.template(), b: ti.template()) -> ti.f32: 
    ans = 0.0
    for i in a:
        ans += a[i].dot(b[i])
    return ans


def cg():
    def mul(x):
        matmul_cell(F_mul_ans, x)
        return F_mul_ans

    get_force()
    get_b()
    mul(vel)
    add(F_r0, F_b, -1, mul(vel))

    d = F_p0
    d.copy_from(F_r0)
    r_2 = dot(F_r0, F_r0)
    n_iter = 50
    epsilon = 1e-6
    r_2_init = r_2
    r_2_new = r_2
    for _ in range(n_iter):
        q = mul(d)
        alpha = r_2_new / dot(d, q)
        add(vel, vel, alpha, d)
        add(F_r0, F_r0, -alpha, q)
        r_2 = r_2_new
        r_2_new = dot(F_r0, F_r0)
        if r_2_new <= r_2_init * epsilon**2:
            break
        beta = r_2_new / r_2
        add(d, F_r0, beta, d)
    # cg iteration finally output vel
    force.fill(0)
    add(pos, pos, dt, vel)


@ti.kernel
def advect():
    for p in pos:
        vel[p] += dt * (force[p] / mass[p])
        pos[p] += dt * vel[p]
        force[p] = ti.Vector([0, 0, 0])

@ti.kernel
def integrate_vel_explicit_euler():
    for p in pos:
        pos[p] += dt * vel[p]


@ti.kernel
def init():
    g_current_step = 0
    g_current_time = 0.0
    for u in pos:
        pos[u] = pos0[u]
        vel[u] = [0.0] * 3
        force[u] = [0.0] * 3
        mass[u] = 0.0
    for c in tet_indices:
        F = Ds(tet_indices[c])
        rest_matrix[c] = F.inverse()
        rest_volume[c] = ti.abs(F.determinant()) / 6
        for i in range(4):
            mass[tet_indices[c][i]] += rest_volume[c] / 4 * density
    for u in pos:
        pos[u].y += 1.0



@ti.kernel
def floor_bound():
    for u in pos:
        if pos[u].y < 0:
            pos[u].y = 0
            if vel[u].y < 0:
                vel[u].y = 0


# implicit substep
def substep_implicit():
    cg()
    floor_bound()

# uncomment to see the explicit substep
def substep_explicit():
    for i in range(10):
        get_force()
        advect()
    floor_bound()

def substep():
    substep_implicit()
    # substep_explicit()

@ti.kernel
def get_vol_err() -> ti.f32:
    err = 0.0
    for c in tet_indices:
        F = Ds(tet_indices[c])
        vol = ti.abs(F.determinant()) / 6
        err += (vol - rest_volume[c])**2
    return err


@ti.kernel
def compute_potential_energy() -> ti.f32:
    """计算总势能：弹性势能 + 重力势能"""
    elastic_energy = 0.0
    gravity_energy = 0.0
    
    # 计算弹性势能
    for c in tet_indices:
        verts = tet_indices[c]
        F = Ds(verts) @ rest_matrix[c]
        # Neo-Hookean能量密度: W = mu * (tr(F^T F) - 3)
        I_1 = (F.transpose() @ F).trace()
        psi = mu * (I_1 - 3.0)
        elastic_energy += psi * rest_volume[c]
    
    # 计算重力势能
    for u in pos:
        gravity_energy += mass[u] * 9.8 * pos[u].y
    
    return elastic_energy + gravity_energy


def render(substep,init):
    
    def T(a):
        phi, theta = np.radians(28), np.radians(32)

        a = a - 0.2
        x, y, z = a[:, 0], a[:, 1], a[:, 2]
        c, s = np.cos(phi), np.sin(phi)
        C, S = np.cos(theta), np.sin(theta)
        x, z = x * c + z * s, z * c - x * s
        u, v = x, y * C + z * S
        return np.array([u, v]).swapaxes(0, 1) + 0.5

    gui = ti.GUI("Implicit FEM", show_gui=not headless_render)  # 启用GUI显示
    pause = False
    frame_id = 0
    vol_err = 0.0
    global g_current_step, g_current_time
    while gui.running:
        if not pause:
            substep()
            if frame_id % 20 == 0 and frame_id > 0:
                vol_err = get_vol_err()
                print(f"Frame {frame_id}, Volume Error: {vol_err:.2e}")
            frame_id += 1
            g_current_step += 1
            g_current_time += dt

        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                break
            if gui.event.key == ti.GUI.SPACE:
                pause = not pause
        if gui.is_pressed("r"):
            init()
        gui.clear(0xFFFFFF)
        
        # draw ground grid on y=0
        grid_size = 10
        grid_step = 0.2
        for i in range(grid_size + 1):
            x = i * grid_step - 1
            # lines along x direction
            p1 = T(np.array([[x, 0, -1]]) / 3)[0]
            p2 = T(np.array([[x, 0, 1]]) / 3)[0]
            gui.line(p1, p2, radius=1, color=0xCCCCCC)
            
            z = i * grid_step - 1
            # lines along z direction
            p1 = T(np.array([[-1, 0, z]]) / 3)[0]
            p2 = T(np.array([[1, 0, z]]) / 3)[0]
            gui.line(p1, p2, radius=1, color=0xCCCCCC)
        
        # project vertices
        vertices_2d = T(pos.to_numpy() / 3)
        
        # draw triangle faces (semi-transparent)
        for i in range(4):
            face = tri_indices.to_numpy()[i]
            triangle = vertices_2d[face]
            gui.triangle(triangle[0], triangle[1], triangle[2], color=0xEECCAA)
        
        # draw wireframe edges
        for i in range(6):
            edge = edge_indices.to_numpy()[i]
            gui.line(vertices_2d[edge[0]], vertices_2d[edge[1]], radius=2, color=0x0000FF)
        
        # draw vertices
        gui.circles(vertices_2d, radius=5, color=0xBA543A)
        
        gui.text(f"Volume Error: {vol_err:.2e}", pos=(0.02, 0.95), color=0x000000)
        gui.text(f"step: {g_current_step:.2e}", pos=(0.02, 0.90), color=0x000000)
        gui.text(f"time: {g_current_time:.2e} s", pos=(0.02, 0.85), color=0x000000)
        if headless_render:
            gui.show(f"pic/{g_current_step:06d}.png")
        else:
            gui.show()


def main():
    get_vertices()
    get_indices()
    init()
    render(substep,init)

    # def T(a):
    #     phi, theta = np.radians(28), np.radians(32)

    #     a = a - 0.2
    #     x, y, z = a[:, 0], a[:, 1], a[:, 2]
    #     c, s = np.cos(phi), np.sin(phi)
    #     C, S = np.cos(theta), np.sin(theta)
    #     x, z = x * c + z * s, z * c - x * s
    #     u, v = x, y * C + z * S
    #     return np.array([u, v]).swapaxes(0, 1) + 0.5

    # gui = ti.GUI("Implicit FEM")  # 启用GUI显示
    # pause = False
    # frame_id = 0
    # vol_err = 0.0
    # while gui.running:
    #     if not pause:
    #         substep()
    #         if frame_id % 20 == 0 and frame_id > 0:
    #             vol_err = get_vol_err()
    #             print(f"Frame {frame_id}, Volume Error: {vol_err:.2e}")
    #         frame_id += 1
    #     if gui.get_event(ti.GUI.PRESS):
    #         if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
    #             break
    #         if gui.event.key == ti.GUI.SPACE:
    #             pause = not pause
    #     if gui.is_pressed("r"):
    #         init()
    #     gui.clear(0xFFFFFF)
        
    #     # draw ground grid on y=0
    #     grid_size = 10
    #     grid_step = 0.2
    #     for i in range(grid_size + 1):
    #         x = i * grid_step - 1
    #         # lines along x direction
    #         p1 = T(np.array([[x, 0, -1]]) / 3)[0]
    #         p2 = T(np.array([[x, 0, 1]]) / 3)[0]
    #         gui.line(p1, p2, radius=1, color=0xCCCCCC)
            
    #         z = i * grid_step - 1
    #         # lines along z direction
    #         p1 = T(np.array([[-1, 0, z]]) / 3)[0]
    #         p2 = T(np.array([[1, 0, z]]) / 3)[0]
    #         gui.line(p1, p2, radius=1, color=0xCCCCCC)
        
    #     # project vertices
    #     vertices_2d = T(pos.to_numpy() / 3)
        
    #     # draw triangle faces (semi-transparent)
    #     for i in range(4):
    #         face = tri_indices.to_numpy()[i]
    #         triangle = vertices_2d[face]
    #         gui.triangle(triangle[0], triangle[1], triangle[2], color=0xEECCAA)
        
    #     # draw wireframe edges
    #     for i in range(6):
    #         edge = edge_indices.to_numpy()[i]
    #         gui.line(vertices_2d[edge[0]], vertices_2d[edge[1]], radius=2, color=0x0000FF)
        
    #     # draw vertices
    #     gui.circles(vertices_2d, radius=5, color=0xBA543A)
        
    #     gui.text(f"Volume Error: {vol_err:.2e}", pos=(0.02, 0.95), color=0x000000)
    #     gui.text(f"step: {frame_id:.2e}", pos=(0.02, 0.90), color=0x000000)
    #     gui.text(f"time: {frame_id * dt:.2e} s", pos=(0.02, 0.85), color=0x000000)
    #     # gui.show(f"pic/{frame_id:06d}.png")
    #     gui.show()



if __name__ == "__main__":
    main()
