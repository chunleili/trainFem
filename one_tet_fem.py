import numpy as np
import taichi as ti

ti.init(arch=ti.cuda)

verbose = True
# 材料参数
# E, nu = 5e4, 0.0
E, nu = 5e4, 0.0
mu, la = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
density = 1000.0
dt = 1e-2
damping = 0.1  # 阻尼系数

# 只有一个四面体，4个顶点
n_verts = 4
n_cells = 1

# 顶点索引 (一个四面体有4个顶点)
F_vertices = ti.Vector.field(4, dtype=ti.i32, shape=n_cells)

# 位置、速度、力、质量
F_x = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
F_ox = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
F_v = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
F_f = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
F_mul_ans = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
F_m = ti.field(dtype=ti.f32, shape=n_verts)

# 参考构型的逆和体积
F_B = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_cells)
F_W = ti.field(dtype=ti.f32, shape=n_cells)


@ti.kernel
def get_vertices():
    """初始化一个四面体的顶点"""
    # 四面体的四个顶点
    F_vertices[0] = ti.Vector([0, 1, 2, 3])
    # 设置初始位置 (一个标准四面体)
    F_ox[0] = ti.Vector([0.0, 0.0, 0.0])
    F_ox[1] = ti.Vector([1.0, 0.0, 0.0])
    F_ox[2] = ti.Vector([0.0, 1.0, 0.0])
    F_ox[3] = ti.Vector([0.0, 0.0, 1.0])



@ti.func
def Ds(verts):
    return ti.Matrix.cols([F_x[verts[i]] - F_x[verts[3]] for i in range(3)])


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
    F = Ds(verts) @ F_B[c]
    P = ti.Matrix.zero(ti.f32, 3, 3)
    U, sig, V = ssvd(F)
    P = 2 * mu * (F - U @ V.transpose())
    H = -F_W[c] * P @ F_B[c].transpose()
    for i in ti.static(range(3)):
        force = ti.Vector([H[j, i] for j in range(3)])
        F_f[verts[i]] += force
        F_f[verts[3]] -= force



@ti.kernel
def get_force():
    for c in F_vertices:
        get_force_func(c, F_vertices[c])
    for u in F_f:
        F_f[u].y -= 9.8 * F_m[u]
        # 添加速度阻尼力
        F_f[u] -= damping * F_m[u] * F_v[u]




@ti.kernel
def matmul_cell(ret: ti.template(), vel: ti.template()):
    for i in ret:
        ret[i] = vel[i] * F_m[i] * (1.0 + damping * dt)  # 质量矩阵 + 阻尼矩阵
    for c in F_vertices:
        verts = F_vertices[c]
        W_c = F_W[c]
        B_c = F_B[c]
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
def add(ans: ti.template(), a: ti.template(), k: ti.f32, b: ti.template()):
    for i in ans:
        ans[i] = a[i] + k * b[i]


@ti.kernel
def dot(a: ti.template(), b: ti.template()) -> ti.f32:
    ans = 0.0
    for i in a:
        ans += a[i].dot(b[i])
    return ans



# CG求解器用的临时场
F_b = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
F_r0 = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
F_p0 = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)


@ti.kernel
def compute_potential_energy() -> ti.f32:
    """计算总势能：弹性势能 + 重力势能"""
    elastic_energy = 0.0
    gravity_energy = 0.0
    
    # 计算弹性势能
    for c in F_vertices:
        verts = F_vertices[c]
        F = Ds(verts) @ F_B[c]
        # Neo-Hookean能量密度: W = mu * (tr(F^T F) - 3)
        I_1 = (F.transpose() @ F).trace()
        psi = mu * (I_1 - 3.0)
        elastic_energy += psi * F_W[c]
    
    # 计算重力势能
    for u in F_x:
        gravity_energy += F_m[u] * 9.8 * F_x[u].y
    
    return elastic_energy + gravity_energy


@ti.kernel
def compute_volume_error() -> ti.f32:
    """计算体积误差：(当前体积 - 初始体积) / 初始体积"""
    vol_err = 0.0
    for c in F_vertices:
        verts = F_vertices[c]
        F = Ds(verts) @ F_B[c]
        J = F.determinant()  # 体积比 = det(F)
        # 体积误差 = |J - 1|，理想情况下不可压缩材料 J = 1
        vol_err += ti.abs(J - 1.0)
    return vol_err




@ti.kernel
def get_b():
    for i in F_b:
        F_b[i] = F_m[i] * F_v[i] + dt * F_f[i]


def cg():
    """共轭梯度法求解隐式时间积分"""
    def mul(x):
        matmul_cell(F_mul_ans, x)
        return F_mul_ans

    get_force()
    get_b()
    mul(F_v)
    add(F_r0, F_b, -1, mul(F_v))

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
        add(F_v, F_v, alpha, d)
        add(F_r0, F_r0, -alpha, q)
        r_2 = r_2_new
        r_2_new = dot(F_r0, F_r0)
        if r_2_new <= r_2_init * epsilon**2:
            break
        beta = r_2_new / r_2
        add(d, F_r0, beta, d)
    F_f.fill(0)
    add(F_x, F_x, dt, F_v)




@ti.kernel
def init():
    for u in F_x:
        F_x[u] = F_ox[u]
        F_v[u] = [0.0] * 3
        F_f[u] = [0.0] * 3
        F_m[u] = 0.0
    for c in F_vertices:
        F = Ds(F_vertices[c])
        F_B[c] = F.inverse()
        F_W[c] = ti.abs(F.determinant()) / 6
        for i in range(4):
            F_m[F_vertices[c][i]] += F_W[c] / 4 * density
    # 抬高四面体位置，避免初始就在地面下
    for u in F_x:
        F_x[u].y += 1.0




@ti.kernel
def floor_bound():
    for u in F_x:
        if F_x[u].y < 0:
            F_x[u].y = 0
            if F_v[u].y < 0:
                F_v[u].y = 0


# 四面体的索引 (4个面，每个面3个顶点)
indices = ti.field(ti.i32, shape=4 * 3)


@ti.kernel
def get_indices():
    """设置四面体的4个三角形面"""
    # 面0: 顶点 0, 1, 2
    indices[0], indices[1], indices[2] = 0, 2, 1
    # 面1: 顶点 0, 1, 3
    indices[3], indices[4], indices[5] = 0, 1, 3
    # 面2: 顶点 0, 2, 3
    indices[6], indices[7], indices[8] = 0, 3, 2
    # 面3: 顶点 1, 2, 3
    indices[9], indices[10], indices[11] = 1, 2, 3


def substep():
    """时间步进"""
    cg()
    floor_bound()
    if verbose:
        # 计算并打印势能和体积误差
        pe = compute_potential_energy()
        vol_err = compute_volume_error()
        print(f"Potential Energy: {pe:.6f}, Volume Error: {vol_err:.6f}")



def main():
    get_vertices()
    init()
    get_indices()

    # 使用GGUI渲染
    res = (800, 600)
    window = ti.ui.Window("One Tetrahedron FEM", res, vsync=True)

    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(2.0, 2.0, 3.0)
    camera.lookat(0.5, 1.0, 0.5)
    camera.fov(55)

    while window.running:
        substep()
        
        if window.is_pressed("r"):
            init()
        if window.is_pressed(ti.GUI.ESCAPE):
            break

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        scene.ambient_light((0.9,) * 3)
        scene.point_light(pos=(0.5, 5.0, 0.5), color=(1.0, 1.0, 1.0))
        scene.point_light(pos=(5.0, 5.0, 5.0), color=(1.0, 1.0, 1.0))

        scene.mesh(F_x, indices, color=(0.73, 0.33, 0.23), show_wireframe=True)

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()

