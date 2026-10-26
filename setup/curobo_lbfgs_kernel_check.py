"""Check cuRobo's fused L-BFGS CUDA kernel against its own PyTorch version.

cuRobo computes the L-BFGS step either with the CUDA kernel (LBFGScu, the
default) or with plain PyTorch (jit_lbfgs_update_buffers +
jit_lbfgs_compute_step_direction, used when use_cuda_kernel=False). Both
should give the same step. This runs both on random inputs for every history
size the kernel is compiled for, and for problem sizes below and above 32
(sizes above 32 use the cross-warp reduction that traps on H100).

Run from RoboTwin/, on GPU 6:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
    .venv/bin/python ../setup/curobo_lbfgs_kernel_check.py
"""

import torch

import curobo
from curobo.curobolib.opt import LBFGScu
from curobo.opt.newton.lbfgs import jit_lbfgs_compute_step_direction, jit_lbfgs_update_buffers

print(f"curobo from {curobo.__file__}, {torch.cuda.get_device_name(0)}", flush=True)
torch.manual_seed(0)
dev, b, eps = "cuda", 4, 0.01
worst = 0.0
for m in [5, 6, 7, 15, 27, 31]:  # history sizes with a compiled kernel
    for v in [7, 32, 33, 64, 100, 192, 224, 384]:  # d_opt = horizon * dof
        if v * m + 33 >= 12000:  # cuRobo falls back to PyTorch here anyway
            continue
        s, y = torch.randn(m, b, v, 1, device=dev), torch.randn(m, b, v, 1, device=dev)
        rho = torch.rand(m, b, 1, 1, device=dev)
        x0, g0 = torch.randn(b, v, 1, device=dev), torch.randn(b, v, 1, device=dev)
        q, gq = torch.randn(b, v, device=dev), torch.randn(b, 1, v, device=dev)
        for shared in [True, False]:
            # CUDA kernel (updates its copies of the buffers in place).
            A = [t.clone() for t in (rho, y, s, x0, g0)]
            dq_cuda = LBFGScu.apply(torch.zeros(b, v, device=dev), *A[:3], q, gq, *A[3:], eps, True, shared)
            # PyTorch reference.
            sB, yB, rhoB, x0B, g0B = jit_lbfgs_update_buffers(
                q, gq, s.clone(), y.clone(), rho.clone(), x0.clone(), g0.clone(), True
            )
            alpha = torch.zeros(m, b, 1, 1, device=dev)
            dq_ref = jit_lbfgs_compute_step_direction(alpha, rhoB, yB, sB, gq, m, eps, True).view(b, v)
            torch.cuda.synchronize()
            err = ((dq_cuda - dq_ref).abs().max() / dq_ref.abs().max()).item()
            worst = max(worst, err)
            if err > 1e-3:
                print(f"MISMATCH m={m} v_dim={v} shared={shared}: rel err {err:.2e}", flush=True)
print(f"RESULT max relative error over all cases: {worst:.2e} ({'OK' if worst < 1e-3 else 'FAIL'})", flush=True)
