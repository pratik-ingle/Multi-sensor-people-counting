"""Headless OpenGL (EGL) renderer for the thermal cameras.

Pass 1  rasterise the 3D scene (furniture, posed people, seat imprints) into a high-resolution
        pinhole image of LWIR radiance, wide enough to contain the whole lens field of view.
Pass 2  for every sensor pixel, sample that image along the real (distorted) lens rays at a 6x6
        grid spanning +-1.5 px and weight by pixel footprint (*) optics blur -> pixel radiance.
Only the small sensor images are read back to the CPU.
"""
from __future__ import annotations

import threading

import moderngl
import numpy as np

from .geometry import Pose
from .thermal_camera import Optics, view_matrix

_CTX: dict[int, moderngl.Context] = {}


def gl_context() -> moderngl.Context:
    """One EGL context per thread (OpenGL contexts are bound to the thread that uses them)."""
    tid = threading.get_ident()
    if tid not in _CTX:
        _CTX[tid] = moderngl.create_context(standalone=True, backend="egl")
    return _CTX[tid]


_VS1 = """
#version 330
uniform mat4 u_mvp;
uniform mat4 u_view;
in vec3 in_pos;
in float in_rad;
in float in_id;
out float v_rad;
flat out float v_id;
out float v_dist;
void main() {
    vec4 p = vec4(in_pos, 1.0);
    gl_Position = u_mvp * p;
    v_rad = in_rad;
    v_id = in_id;
    v_dist = length((u_view * p).xyz);
}
"""
_FS1 = """
#version 330
in float v_rad;
flat in float v_id;
in float v_dist;
layout(location = 0) out vec4 f_out;
void main() { f_out = vec4(v_rad, v_dist, v_id, 1.0); }
"""
_VS2 = """
#version 330
in vec2 in_xy;
void main() { gl_Position = vec4(in_xy, 0.0, 1.0); }
"""
_FS2 = """
#version 330
uniform sampler2D u_rad;
uniform sampler2D u_uv;
uniform int u_S;
uniform float u_w[64];
uniform int u_y0;
out float f_out;
void main() {
    ivec2 pix = ivec2(int(gl_FragCoord.x), int(gl_FragCoord.y) - u_y0);
    float acc = 0.0;
    for (int b = 0; b < u_S; b++) {
        for (int a = 0; a < u_S; a++) {
            vec2 uv = texelFetch(u_uv, ivec2(pix.x * u_S + a, pix.y * u_S + b), 0).xy;
            acc += u_w[b * u_S + a] * texture(u_rad, uv).r;
        }
    }
    f_out = acc;
}
"""


def _perspective(tx: float, ty: float, near=0.05, far=60.0) -> np.ndarray:
    return np.array([[1 / tx, 0, 0, 0], [0, 1 / ty, 0, 0],
                     [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)], [0, 0, -1, 0]])


class ThermalRenderer:
    def __init__(self, optics: Optics, poses: list[Pose], static_V, static_F, n_human_verts: int, human_F,
                 max_people: int = 400, max_imprints: int = 256, oversample: float = 4.0, S: int = 6):
        self.ctx = ctx = gl_context()
        self.optics = optics
        self.W, self.H = W, H = optics.W, optics.H
        self.n_cams = len(poses)
        self.views = [view_matrix(p) for p in poses]

        # Sample rays of every sensor pixel, then the pinhole frustum that contains them all.
        offs = -1.5 + 3.0 * (np.arange(S) + 0.5) / S
        xs = (np.arange(W)[:, None] + 0.5 + offs[None, :]).ravel()
        ys = (np.arange(H)[:, None] + 0.5 + offs[None, :]).ravel()
        X, Y = np.meshgrid(xs, ys)
        d = optics.rays(X, Y)
        px, py = d[..., 0] / d[..., 2], d[..., 1] / d[..., 2]
        self.tx, self.ty = 1.02 * np.abs(px).max(), 1.02 * np.abs(py).max()
        uv = np.stack([0.5 + 0.5 * px / self.tx, 0.5 + 0.5 * py / self.ty], axis=-1).astype(np.float32)
        sig2 = 1.0 / 12.0 + optics.p.psf_sigma_px ** 2
        w = np.exp(-(offs[None, :] ** 2 + offs[:, None] ** 2) / (2 * sig2)).ravel()
        w = (w / w.sum()).astype(np.float32)
        self.W1 = int(min(2048, np.ceil(2 * self.tx * optics.f * oversample)))
        self.H1 = int(min(2048, np.ceil(2 * self.ty * optics.f * oversample)))
        self.P = _perspective(self.tx, self.ty)

        self.prog1 = ctx.program(vertex_shader=_VS1, fragment_shader=_FS1)
        self.prog2 = ctx.program(vertex_shader=_VS2, fragment_shader=_FS2)

        # static scene
        self.n_static = len(static_V)
        self.static_buf = np.zeros((self.n_static, 5), np.float32)
        self.static_buf[:, :3] = static_V
        self.vbo_static = ctx.buffer(self.static_buf.tobytes())
        ibo = ctx.buffer(np.asarray(static_F, np.uint32).tobytes())
        self.vao_static = ctx.vertex_array(self.prog1, [(self.vbo_static, "3f 1f 1f", "in_pos", "in_rad", "in_id")],
                                           index_buffer=ibo, index_element_size=4)
        # people (index buffer prepared for max_people copies of the body template)
        self.nv_h = n_human_verts
        self.ni_h = human_F.size
        self.max_people = max_people
        self.vbo_people = ctx.buffer(reserve=max_people * n_human_verts * 20)
        idx = (np.asarray(human_F, np.uint32)[None] + (np.arange(max_people, dtype=np.uint32) * n_human_verts)[:, None, None])
        ibo_p = ctx.buffer(idx.astype(np.uint32).tobytes())
        self.vao_people = ctx.vertex_array(self.prog1, [(self.vbo_people, "3f 1f 1f", "in_pos", "in_rad", "in_id")],
                                           index_buffer=ibo_p, index_element_size=4)
        # imprint quads
        self.max_quads = max_imprints * 2
        self.vbo_imp = ctx.buffer(reserve=self.max_quads * 4 * 20)
        q = np.array([0, 1, 2, 0, 2, 3], np.uint32)
        ibo_i = ctx.buffer((q[None] + 4 * np.arange(self.max_quads, dtype=np.uint32)[:, None]).astype(np.uint32).tobytes())
        self.vao_imp = ctx.vertex_array(self.prog1, [(self.vbo_imp, "3f 1f 1f", "in_pos", "in_rad", "in_id")],
                                        index_buffer=ibo_i, index_element_size=4)

        # pass-1 target
        self.tex1 = ctx.texture((self.W1, self.H1), 4, dtype="f4")
        self.tex1.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex1.repeat_x = self.tex1.repeat_y = False
        self.fbo1 = ctx.framebuffer(color_attachments=[self.tex1], depth_attachment=ctx.depth_renderbuffer((self.W1, self.H1)))
        # pass-2 resources
        self.uvtex = ctx.texture((W * S, H * S), 2, data=uv.tobytes(), dtype="f4")
        self.uvtex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.atlas = ctx.texture((W, H * self.n_cams), 1, dtype="f4")
        self.fbo2 = ctx.framebuffer(color_attachments=[self.atlas])
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32)
        self.vao2 = ctx.vertex_array(self.prog2, [(ctx.buffer(quad.tobytes()), "2f", "in_xy")])
        self.prog2["u_S"].value = S
        self.prog2["u_w"].write(np.pad(w, (0, 64 - len(w))).astype(np.float32).tobytes())
        self.prog2["u_rad"].value = 0
        self.prog2["u_uv"].value = 1

    def set_static_radiance(self, rad: np.ndarray):
        self.static_buf[:, 3] = rad
        self.vbo_static.write(self.static_buf.tobytes())

    def render(self, cams, people_V=None, people_rad=None, people_id=None, imp_V=None, imp_rad=None,
               background_rad: float = 0.0) -> np.ndarray:
        """Radiance images (len(cams), H, W), row 0 = top of the image."""
        ctx = self.ctx
        n_people = 0 if people_V is None else len(people_V) // self.nv_h
        n_people = min(n_people, self.max_people)
        if n_people:
            buf = np.empty((n_people * self.nv_h, 5), np.float32)
            buf[:, :3] = people_V[:n_people * self.nv_h]
            buf[:, 3] = people_rad[:n_people * self.nv_h]
            buf[:, 4] = people_id[:n_people * self.nv_h]
            self.vbo_people.write(buf.tobytes())
        n_quads = 0 if imp_V is None else min(len(imp_V) // 4, self.max_quads)
        if n_quads:
            buf = np.zeros((n_quads * 4, 5), np.float32)
            buf[:, :3] = imp_V[:n_quads * 4]
            buf[:, 3] = imp_rad[:n_quads * 4]
            self.vbo_imp.write(buf.tobytes())

        for k, c in enumerate(cams):
            ctx.enable(moderngl.DEPTH_TEST)
            self.fbo1.use()
            self.fbo1.clear(background_rad, 0.0, 0.0, 1.0, depth=1.0)
            V = self.views[c]
            self.prog1["u_mvp"].write((self.P @ V).T.astype(np.float32).tobytes())
            self.prog1["u_view"].write(V.T.astype(np.float32).tobytes())
            self.vao_static.render(moderngl.TRIANGLES)
            if n_people:
                self.vao_people.render(moderngl.TRIANGLES, vertices=n_people * self.ni_h)
            if n_quads:
                self.vao_imp.render(moderngl.TRIANGLES, vertices=n_quads * 6)
            ctx.disable(moderngl.DEPTH_TEST)
            self.fbo2.viewport = (0, k * self.H, self.W, self.H)
            self.fbo2.use()
            self.tex1.use(location=0)
            self.uvtex.use(location=1)
            self.prog2["u_y0"].value = k * self.H
            self.vao2.render(moderngl.TRIANGLE_STRIP)
        raw = self.fbo2.read(viewport=(0, 0, self.W, self.H * len(cams)), components=1, dtype="f4")
        img = np.frombuffer(raw, dtype=np.float32).reshape(len(cams), self.H, self.W)
        return img[:, ::-1, :].copy()

    def release(self):
        for obj in (self.vbo_static, self.vbo_people, self.vbo_imp, self.vao_static, self.vao_people, self.vao_imp,
                    self.tex1, self.fbo1, self.uvtex, self.atlas, self.fbo2, self.vao2, self.prog1, self.prog2):
            try:
                obj.release()
            except Exception:
                pass
