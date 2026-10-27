# Đối chiếu source trước khi triển khai

## Bổ sung architecture research 26/10/2026

Đợt search RGB-D tái sử dụng CP/self/cross, geometry XYZ+normal+validity và reliability gates đã audit. `ReliabilityGate` legacy đã có channel/spatial weighting nhưng chưa nhận measured-depth validity trực tiếp. Biến thể validity thêm một quality channel, giữ recurrence và output projections; fallback toàn-invalid dùng RGB self với nhánh Depth bằng 0 trước learned fuse. Đây là thiết kế local, không gọi là MS2Fusion nguyên bản.

Decoder hiện tại project/resize/mean rồi hai Conv3x3–GroupNorm(1)–GELU. Các ablation config-driven thay activation, residual, merge; concat projection khởi tạo bằng mean để kiểm tra bảo toàn baseline trước học. Detail skip trước refinement là giả thuyết riêng, không đồng nghĩa HRNet đầy đủ.

[CMX](https://arxiv.org/abs/2203.04838) mô tả cross-modal feature rectification rồi feature fusion; được dùng để đặt giả thuyết compensation, chưa copy implementation hoặc chứng minh hiệu quả CCS2. [Integral regression](https://arxiv.org/abs/1711.08229) hỗ trợ nghiên cứu supervision tọa độ khả vi; cửa sổ argmax local hiện tại phải kiểm tra gradient khi peak sai xa GT. Source VMamba pinned dưới đây vẫn là căn cứ ordering; không thay selective-scan kernel.

Local residual hiện có ID/zero-init/predicted ROI và fused-feature input. Biến thể Conv dùng cùng API/ROI/merge thay core SSM; RGB-detail-only không được gọi là Local RGB-D trực tiếp. Search Local hai nhánh ROI sẽ mở sau khi kiểm chứng control và incremental cost.

Trạng thái này là research/implementation; correctness/profile và accuracy được ghi riêng trong campaign registry, không suy ra hiệu quả từ paper hoặc tên module.

### Architecture audit thực thi và profile đầu tiên

Luồng thực tế: RGB SE-ResNet → bốn scale; depth+K → XYZ/normal/validity → geometry residual encoder → bốn scale cùng stride; mỗi scale chạy self/cross CP readout và reliability compensation; decoder project/resize/merge → Conv refinement → chín heatmap. Local residual crop từ Global argmax, thêm semantic ID, predict delta và scatter trên lưới Global. Không có detector ngoài và không có một SSM thứ ba bắt buộc sau fused decoder.

| Khối | Spatial/normalization | Hướng kiểm chứng |
|---|---|---|
| RGB backbone | Pretrained SE-ResNet-50, optional RGB stem stride 2 | Giữ weights/hash; không thay backbone toàn bộ trước profile |
| Geometry | Input 7 channel metric XYZ, normal, validity; stem 7x7/2 + pool; GroupNorm(1), SiLU residual | Gradient/validity, input lẻ và sampling grid |
| CP coordinator | Independent modality projection/SSM parameters, shared local depthwise Conv, recipient LayerNorm/projection | C-only exchange, self/cross đúng nghĩa; directions 1/4 |
| Reliability | Spatial/channel sigmoid từ self/cross/difference | Thêm measured validity là ablation; gate không phải calibrated confidence |
| Decoder | 4x32 channel, mean mặc định, width 32, 2 Conv3x3/GN(1)/GELU | Concat initialized-mean, activation, residual và detail-before |
| Local | One predicted ROI, ID, zero-init delta, gate logit 0 | Frozen Global, Conv/SSM cores, support/identity và incremental cost |

GPU profile Slurm 6125, A5000: control 27.290.489 parameters; concat thêm 4.096 parameters; validity gate thêm 256 parameters. Temporary b8 step P50 control 56,31 ms, concat 53,97 ms, SiLU 56,59 ms, residual 56,75 ms, validity 58,52 ms, detail-before 57,47 ms, scan4 67,22 ms. Frozen-Global Local SSM/Conv lần lượt 20,11/18,65 ms. Đây là 25 steps trên một batch, chỉ dùng dự toán; chưa đủ benchmark speedup hoặc convergence.

Inclusive GPU-event component P50 control: RGB encoder khoảng 2,89 ms, decoder 0,60 ms, fusion scale đầu 1,18 ms. Không cộng các timing lồng nhau thành tổng pipeline. Full component/activation-memory/gradient rows nằm trong `runs/improve_20261025/architecture/profile.json`; scan FLOPs chưa được công cụ này đếm nên để null thay vì báo số thiếu toán tử.

Đợt pilot dùng snapshot riêng; các cải tiến diagnostic GT depth/sigma/resume tiếp theo ở working tree không sửa snapshot đang chạy. Synthetic annotation có `keypoints_2d[*].depth_m`; diagnostic mới ưu tiên giá trị đó và gắn provenance, không sửa prediction/mm lịch sử vốn dùng depth sensor tại nhãn.

### Modality diagnostic thực đo trên candidate lịch sử

Slurm GPU 6161 xuất prediction cho subset validation cố định theo UID hash, 256 ảnh mỗi nguồn; CPU 6163 bootstrap 2.000 lượt theo render/parent, không chạy lại model. Candidate pretrained seed 20261025:

| Intervention | OLD Δmean px [CI95] | NEW Δmean px [CI95] |
|---|---|---|
| Depth toàn invalid | +0,32462 [0,26906; 0,38085] | +0,21875 [0,17267; 0,27478] |
| Geometry features bằng 0 | +1,40952 [1,32703; 1,49559] | +0,84790 [0,78549; 0,91697] |
| RGB features bằng 0 | +13,12385 [10,74017; 16,18580] | +313,32082 [274,26025; 349,41335] |

Delta dương nghĩa clean RGB-D tốt hơn intervention. Các phép feature-zero tạo phân phối nhân tạo, không phải baseline single-modality được train; depth-invalid có liên hệ với dropout training nhưng vẫn là diagnostic. Không gọi đây là chứng minh nhân quả hoặc generalization độc lập. Kết quả chỉ áp dụng candidate lịch sử này; kiến trúc thắng search cần diagnostics riêng trên nhiều seed. Gradient/gates và GT-depth mm có trong `architecture/tasks/modality_F_candidate_20261025/modality.json`; summary CPU tại `architecture/modality_summary.json`.

Snapshot tải ngày 2026-09-14, giữ nguyên tại `references/`. Mỗi `*_tree.json`
lưu repo, commit SHA và danh sách file. Chỉ đọc các đoạn liên quan bên dưới,
không khẳng định đã audit toàn bộ repository hoặc chạy detector upstream.
Các file source tải về chỉ phục vụ đối chiếu, không nằm trong package cài đặt.

| Repo | Commit |
| --- | --- |
| MS2Fusion | `1c4225b8d4804736184be512e7dd006e641ff988` |
| MambaSOD | `8e0e5f40448456ae307d573ea03ddf858d4edf3e` |
| Fusion-Mamba | `2735a278fdc1bf79d7ff6d3f81481221ed3ffc1c` |
| Mamba | `e9594ce1c732d97440f0332fdc43170a2294dbfa` |
| VMamba | `2ed52ead062a51a64521ed3871d52914bf532876` |

## MS2Fusion: CP và SP nằm chung trong class thực tế

Đọc constructor và forward của `DBISSF_Attention`, `DBISSF_SS`, `DBISSF`,
`SSF` trong [RGB-T OD/models/common.py](https://github.com/61s61min/MS2Fusion/blob/1c4225b8d4804736184be512e7dd006e641ff988/RGB-T%20OD/models/common.py#L1986).
Không có class `CPSSM`/`SPSSM` riêng trong đoạn source này; các tên module local
không được xem là tên class chính thức của paper. Kết luận dưới đây dựa trên
object-detection implementation này, không suy rộng rằng mọi task trong repo giống nhau.

`DBISSF_Attention.forward` nhận bốn tensor `[B,L,d_inner]`, theo thứ tự
`x_rgb, x_share_rgb, x_e, x_share_e`. Nó sinh tham số riêng bằng `x_proj_1/2`,
`dt_proj_1/2`, `A_log_1/2`, `D_1/2`. Tại dòng 2148–2160, `Cross=True` thực hiện:

```
y_rgb = scan(x_rgb, dt_rgb, A_rgb, B_rgb, C_e, D_1)
y_e   = scan(x_e,   dt_e,   A_e,   B_e,   C_rgb, D_2)
```

Bias Delta và softplus vẫn thuộc recipient. Chỉ `C` được đổi. `Cross=False`
scan với `C_rgb`/`C_e` riêng. Source trả cross **hoặc** self tùy cờ; hai output
còn lại là **shared branch**, không phải self branch. Module mới tính cả self
và cross đồng thời, dùng cùng tham số recipient và cùng output norm/projection.
Do đó bốn output mới đúng ý nghĩa `rgb_self/rgb_cross/depth_self/depth_cross`,
nhưng không đồng nghĩa với bốn output upstream.

SP được nghiên cứu riêng: `DBISSF_SS.forward` (dòng 2311) áp dụng cùng
`in_proj_share` cho hai feature ảnh `[B,C,H,W]`. Sau permutation, projection,
shared depthwise convolution và SiLU, nó flatten thành `[B,HW,d_inner]`.
Trong `DBISSF_Attention`, hai shared embedding được **cộng**, sau đó
`x_proj_share` sinh `dt_share/B_share/C_share`. `dt_proj_share`, `A_log_share`
và `D_share` cũng dùng chung. Hai scan SP dùng **x_rgb và x_e riêng** làm input,
với cùng toàn bộ bộ tham số shared; không scan một tensor cộng rồi nhân đôi output.
`out_norm_share` và `out_proj_share` dùng chung, trả hai feature shared riêng.

Điểm kết hợp: `SSF.forward` (dòng 2525) ghép cross/shared của RGB qua
`cm_fusion_rgb`, cross/shared của modality e qua `cm_fusion_e`, rồi ghép hai
modality qua `cm_fusion`; cả ba là `CMSSF`. Không mang chuỗi fusion này sang
baseline yêu cầu. `DBISSF._forward` còn có cách cộng shared output với chính
DropPath của nó; không tái sử dụng hành vi này.

`SPSSM` hiện được triển khai độc lập trong `sp_ssm.py`. `MS2FusionBranches`
luôn chạy cả CP và SP, giữ shared embedding/convolution/norm/projection theo
`DBISSF_SS(scan=False, dropout=0)`, trả bốn feature cross/shared của source.
Test đối chiếu dùng trực tiếp AST hai class upstream và cùng weights để kiểm
tra forward và gradient. `RGBDCompensation` là kiến trúc riêng self/cross với
reliability; không gọi kiến trúc đó là MS2Fusion đầy đủ.

## MambaSOD: wrapper RGB-D và Mamba chỉnh sửa

Đã đọc:

- [Cross_Model_Mamba.py](https://github.com/YueZhan721/MambaSOD/blob/8e0e5f40448456ae307d573ea03ddf858d4edf3e/models/CMMamba/Cross_Model_Mamba.py#L158):
  `PatchEmbed`, `PatchUnEmbed`, `CrossMamba`, `CrossMamba_`.
- [Pan_Mamba.py](https://github.com/YueZhan721/MambaSOD/blob/8e0e5f40448456ae307d573ea03ddf858d4edf3e/models/CMMamba/Pan_Mamba.py#L192):
  `SingleMambaBlock`, `TokenSwapMamba`, `CrossMamba`.
- [mamba_simple.py](https://github.com/YueZhan721/MambaSOD/blob/8e0e5f40448456ae307d573ea03ddf858d4edf3e/models/CMMamba/mamba_simple.py#L198):
  `Mamba.forward`, đặc biệt nhánh fast path `bimamba_type="v3"`.

`PatchEmbed` chuyển ảnh thành token `[B,HW,C]`; `CrossMamba_` nhận hai sequence,
concat/project feature rồi gọi Mamba với `extra_emb` và `concated`. Nhánh v3
scan riêng ba embedding rồi dùng output concatenated branch để nhân các output
modality. Đây không phải chứng cứ cho C-only exchange; chứng cứ đó đến từ
MS2Fusion. `CrossMamba_` trả feature ảnh `[B,C,H,W]` và suy H/W bằng căn bậc hai
token count; `Pan_Mamba.CrossMamba` có reshape cố định 128×128. Code mới lưu
H/W thật và kiểm thử ảnh chữ nhật. Không copy channel swapping hoặc wrapper này.

## Fusion-Mamba: gated exchange khác reliability gate

Đọc [models/common.py, Cross_modal_Vmaba_module.forward và Cross_Block](https://github.com/EhanDong/Fusion-Mamba/blob/2735a278fdc1bf79d7ff6d3f81481221ed3ffc1c/models/common.py#L1211).
Module nhận/trả hai tensor `[B,H,W,C]`, tách input projection thành `x` và `z`,
scan các feature, normalize và nhân với `SiLU(z)`. Source tính `y1` đã trộn
trước khi dùng chính `y1` đó để tính `y2`, nên không mô tả source như hai phép
trộn đối xứng độc lập. SiLU không bị chặn trong [0,1].

Chỉ sử dụng ý tưởng điều tiết trao đổi bằng feature gate. Hai sigmoid gate mới
nhìn `[F_self,F_cross,abs(F_cross-F_self)]` và bù residual là thiết kế cho task
này, không phải module nguyên bản từ Fusion-Mamba hoặc MS2Fusion.

## Mamba canonical và VMamba

Đọc [selective_scan_interface.py: selective_scan_fn/ref](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/selective_scan_interface.py#L118)
và [mamba_simple.py: Mamba.forward](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba_simple.py#L119).
Scan nhận `x/Delta [B,d,L]`, `A [d,N]`, variable `B/C [B,N,L]`, skip `D [d]`,
trả `[B,d,L]`. Mamba wrapper nhận/trả `[B,L,C]`, có thêm input projection,
causal convolution, gating và output projection. Simple SSM mới chỉ phụ trách
state-space operation và parameter generation, không nhận modality thứ hai.

Với mỗi token/channel:

```
Delta_t = softplus(project_delta(x_t))
A = -exp(A_log)
h_t = exp(Delta_t * A) * h_(t-1) + Delta_t * B_t * x_t
y_t = sum_N(C_t * h_t) + D * x_t
```

`x_t`: input; `h_t`: hidden state; `A`: transition; `B_t`: input-to-state;
`C_t`: readout; `Delta_t`: selective time step; `D`: direct skip.
Nhân `Delta*B` khớp Mamba reference, không tuyên bố là discretization ZOH chính
xác của B. Khởi tạo A âm, D bằng 1 và Delta ban đầu khoảng 0.001–0.1 theo
nguyên tắc source. Không sử dụng thêm z gate trong SimpleSelectiveSSM.

VMamba đã đọc [csm_triton.py: cross_scan_fwd/cross_merge_fwd](https://github.com/MzeroMiko/VMamba/blob/2ed52ead062a51a64521ed3871d52914bf532876/classification/models/csm_triton.py#L22)
và [csms6s.py: selective_scan_torch](https://github.com/MzeroMiko/VMamba/blob/2ed52ead062a51a64521ed3871d52914bf532876/classification/models/csms6s.py#L25).
Spatial scan `[B,C,H,W] → [B,4,C,HW]` theo row, column và hai hướng ngược;
merge đảo thứ tự/transpose trước khi tổng hợp. Code mới có cùng thứ tự, nhưng
lấy **trung bình** và chia sẻ SSM weights giữa bốn hướng. Đây là lựa chọn local,
không tuyên bố tái tạo đầy đủ VMamba. `vmamba.py` được tải làm tài liệu bổ sung,
không dùng nó làm bằng chứng cho một class chưa được đọc chi tiết.

## Phần viết mới và phạm vi

CP-SSM hiện được biểu diễn bằng `CPSSM(nn.Module)` trong `cp_ssm.py`, nhận hai
sequence `[B,C,L]`, tự sinh tham số hai modality và trả hai cross sequences.
Block sở hữu `rgb_ssm/depth_ssm` và thực hiện đầy đủ hai selective scans với C
được trao đổi. `exchange_readouts` chỉ là helper nội bộ của phép tương tác.
`CrossParametricSSM` trong `cross_parametric.py` là coordinator ảnh: chuẩn bị
sequence, tính self features và gọi `CPSSM.forward` cho cross features, rồi
normalize/project/merge về feature map. Tham số sinh từ token được truyền lại
cho CPSSM để hai nhánh self/cross dùng cùng giá trị và tránh projection lặp.

`models/fusion/ssm` chứa recurrence và spatial ordering độc lập; `ms2fusion`
chứa quy tắc trao đổi readout và CP feature wrapper; `reliability` chứa gate và
compensation mới. `rgbd_compensation.py` project channel, kiểm tra alignment,
trả aux và concat → Conv1×1 → channel LayerNorm → GELU. Wrapper multiscale có
block độc lập cho mỗi scale. Không có Mamba thứ ba sau compensation.

Self/cross dùng chung recipient pre-processing, A/B/Delta/D, output LayerNorm
và projection. Chỉ donor C thay đổi, nên phép trừ không bị trộn với thay đổi
trọng số output branch. Không cộng thêm self+cross trước residual vì công thức
cụ thể trong yêu cầu là `self + sigmoid_gate * (cross-self)`.

Backbone `C:/DATA/code/fusion_code/fusion/se_resnet.py` được giữ nguyên.
Không tạo backbone/decoder/head/full network mới trong workspace trống này.
Phần nối hai encoder độc lập được minh họa trong README, không thay đổi pipeline
train/inference của `fusion_code`.

### G–D–L continuation (2026-10-26)

The search uses explicit `global_ssm.topology=g0/g1/g2` and `decoder.kind=d0/d1/d2`;
legacy defaults and checkpoint parameter paths remain supported. G0 uses the existing
shared-weight four-direction Cross-C scans and directional self/cross compensation.
G1 batches valid 8×8 windows at F1/F2 without padded tokens entering SSM recurrences.
G2 adds an adaptive gated depthwise CNN residual with a zero-initialized output
projection, preserving the initial SS2D output. These are task-specific extensions.

D1 is an additive lateral/top-down separable-convolution FPN, distinct from the old
concat decoder. D2 uses both RGB and geometry stride-two stems to refine 90×160
coarse heatmaps into 180×320 predictions. Local predicted centers use the final Global
heatmap grid and pixel-center-normalized sampling; coarse integer indices are never
reused on the fine grid. L2 combines CNN and four-direction SSM features with an
adaptive gate and zero-initialized residual head. Diagnostics inspect gradients after
optimizer steps, gate saturation, branch masking, and masked-branch accuracy.

All training, tests, profiling and inference run through Slurm. The user removed
the previous GPU-hour ceiling; accounting and the four-GPU concurrency limit remain.
Locked legacy baseline and candidate checkpoints remain comparison references;
synthetic test and D455 evaluation require the final checkpoint lock.

### Persistent G–D–L automation (2026-10-27)

The Slurm CPU controller persists a successor with `afterany` independently of chat.
Submission tokens are written before `sbatch` and matched against live queue and
accounting comments after a crash. Temporary Slurm failures retain the current phase.
Completed checkpoints receive CPU checks of architecture keys/shapes, config/source/
manifest hashes, validation metrics, optimizer, scheduler recipe, RNG and sampler.
Timeout/preemption slices resume from audited checkpoints without reducing the horizon.
A failed optional candidate is recorded and excluded; a failed mandatory control or
unresolved correctness check stops with an explicit technical blocker. The canonical
`architecture_gdl/registry.json` records decisions, winners, paths, verification and jobs.
Official training remains conditional on freeze and native GPU verification; final
synthetic test/D455 evaluation remains conditional on the three-checkpoint final lock.

The user raised campaign concurrency to six GPUs on 2026-10-27. The controller
reads the authorized limit from persistent state and counts pending/running GPU
devices, including the user's other allocations. Decoder and Local pilots can run
three variants across both confirmed seeds (six independent one-GPU tasks). Each
task uses its own seed's verified frozen parent, with the same 15K horizon, 8K
subset and validation protocol. Every seed must pass the existing comparator
before a variant is ranked by mean balanced validation error across seeds. Full
confirmation and the three official seeds remain unchanged. Activation supplements
an in-flight pilot stage without changing existing task JSON, source or workers.
