# Hướng Dẫn Huấn Luyện & Đánh Giá Baseline SHIELD (ICML 2025)

File này hướng dẫn chi tiết cách chạy baseline **SHIELD** (*"SHIELD: Multi-task Multi-distribution Vehicle Routing Solver with Sparsity and Hierarchy"*, Goh et al., ICML 2025) trong codebase hiện tại.

---

## 1. Tóm tắt kiến trúc SHIELD đã tích hợp

* **Encoder**: 6 tầng Transformer MoE (4 experts, top-2 routing ở cấp độ node, embedding dimension 128, 8 heads, FF 512).
* **Clustering**: `ContextClusteringLayer` với $N_c = 5$ cluster centers học được, $B = 5$ vòng lặp soft-clustering kết hợp task prompt $\gamma \in \mathbb{R}^5$ (`[C, O, B, L, TW]`).
* **Decoder**: Sâu 3 tầng (`decoder_layers=3`) sử dụng **Mixture-of-Depths (MoD)** với $\beta = 0.10$ (chỉ top-10% agents được chọn tính toán attention mỗi bước, 90% còn lại dùng residual bypass để tiết kiệm compute).
* **Tracking cụm động**: Cập nhật trạng thái cụm cho từng agent sau mỗi bước chọn node theo Eq. 8.

---

## 2. Lệnh huấn luyện Full SHIELD (5000 Epochs)

### 2.1 Chạy trực tiếp trên Terminal (GPU 0)
```bash
python train.py \
    --model_type SHIELD \
    --problem Train_ALL \
    --problem_size 50 \
    --pomo_size 50 \
    --train_batch_size 128 \
    --train_episodes 20000 \
    --epochs 5000 \
    --embedding_dim 128 \
    --encoder_layer_num 6 \
    --decoder_layers 3 \
    --use_mod \
    --mod_beta 0.10 \
    --use_cluster \
    --n_clusters 5 \
    --cluster_iters 5 \
    --use_task_prompt \
    --task_dim 5 \
    --lr 1e-4 \
    --weight_decay 1e-6 \
    --gpu_id 0
```

> **Ghi chú tiện lợi:** Vì tất cả các tham số kiến trúc trên đã được thiết lập làm **giá trị mặc định** trong parser, bạn có thể chạy phiên bản rút gọn tương đương 100%:
> ```bash
> python train.py --model_type SHIELD --epochs 5000 --gpu_id 0
> ```

---

### 2.2 Chạy nền trên Server (Nohup / Background)
Khuyến nghị sử dụng `nohup` để tiến trình không bị ngắt khi đóng SSH:
```bash
nohup python train.py \
    --model_type SHIELD \
    --epochs 5000 \
    --gpu_id 0 \
    > train_shield_5000.log 2>&1 &
```

* **Theo dõi tiến độ huấn luyện:**
```bash
tail -f train_shield_5000.log
```

* **Xem GPU hoạt động:**
```bash
watch -n 1 nvidia-smi
```

---

## 3. Lệnh kiểm thử & Đánh giá (Evaluation)

Sau khi huấn luyện xong, checkpoint được lưu tự động tại `results/<timestamp>/epoch-5000.pt`.

### 3.1 Đánh giá trên toàn bộ 16 biến thể VRP (`ALL`)
Chạy đánh giá greedy không augmentation (nhanh):
```bash
python test.py \
    --model_type SHIELD \
    --problem ALL \
    --problem_size 50 \
    --checkpoint results/<timestamp>/epoch-5000.pt \
    --test_episodes 1000 \
    --test_batch_size 100 \
    --aug_factor 1 \
    --gpu_id 0
```

Chạy đánh giá đầy đủ với **8-fold Instance Augmentation** (cho kết quả tốt nhất như paper):
```bash
python test.py \
    --model_type SHIELD \
    --problem ALL \
    --problem_size 50 \
    --checkpoint results/<timestamp>/epoch-5000.pt \
    --test_episodes 1000 \
    --test_batch_size 100 \
    --aug_factor 8 \
    --gpu_id 0
```

### 3.2 Đánh giá riêng từng bài toán (ví dụ: CVRP)
```bash
python test.py \
    --model_type SHIELD \
    --problem CVRP \
    --problem_size 50 \
    --checkpoint results/<timestamp>/epoch-5000.pt \
    --test_episodes 1000 \
    --test_batch_size 100 \
    --aug_factor 8 \
    --gpu_id 0
```

---

## 4. Các cấu hình so sánh (Ablation Study)

Bạn có thể chạy các biến thể ablation của SHIELD bằng các cờ có sẵn:

1. **MVMoE (Gốc - 1 layer decoder)**:
   ```bash
   python train.py --model_type MOE --epochs 5000 --gpu_id 0
   ```

2. **MVMoE-Deeper (3 layer dense decoder, không dùng MoD & Cluster)**:
   ```bash
   python train.py --model_type SHIELD --no-use_mod --no-use_cluster --decoder_layers 3 --epochs 5000 --gpu_id 0
   ```

3. **SHIELD-MoD (3 layer MoD decoder, không dùng Cluster)**:
   ```bash
   python train.py --model_type SHIELD --use_mod --mod_beta 0.10 --no-use_cluster --decoder_layers 3 --epochs 5000 --gpu_id 0
   ```

4. **SHIELD Full (Đầy đủ MoD + Clustering + Task prompt)**:
   ```bash
   python train.py --model_type SHIELD --use_mod --mod_beta 0.10 --use_cluster --n_clusters 5 --decoder_layers 3 --epochs 5000 --gpu_id 0
   ```

---

## 5. Chạy lại bộ kiểm tra tính đúng đắn (Sanity Verification)
Để kiểm tra lại tính toàn vẹn của mô hình (shape, gradient, routing, param count) trước khi chạy:
```bash
python test_shield_verification.py
```
