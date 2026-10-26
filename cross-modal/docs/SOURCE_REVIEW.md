# Đối chiếu source trước khi triển khai

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
