# Azure DRL Beginner Project

Đây là bộ project đã được sắp lại theo kiểu **người mới** để bạn làm đề tài:

**Ứng dụng Deep Reinforcement Learning để tối ưu hóa tiêu thụ năng lượng và quản lý tài nguyên động trong hệ thống Cloud Computing**

## Bên trong folder này có gì?

- `notebooks/01_sqlite_to_workload.ipynb`  
  Notebook để đọc file Azure Packing Trace (`.sqlite`) và biến nó thành `workload_real.csv`

- `notebooks/02_train_evaluate_ppo.ipynb`  
  Notebook để train PPO, chạy baseline và đánh giá kết quả

- `src/energy_env.py`  
  Môi trường mô phỏng cloud energy có:
  - trạng thái host **Active / Sleep / Off**
  - mô phỏng **VM consolidation** theo từng VM nhỏ + đếm migration
  - tách **IT power** và **Facility power**, tính **PUE**
  - mô hình nhiệt độ host + chỉ số hao mòn phần cứng

- `src/baselines.py`  
  Các baseline chính: **RoundRobin**, **BestFit**, **Threshold**, **Fixed-Keep**

- `src/analysis_utils.py`
  Hàm xuất biểu đồ để giải thích vì sao PPO tốt hơn (DVFS, power theo thời gian, PUE/nhiệt độ/hao mòn)

- `src/azure_workload_utils.py`  
  Hàm chuyển từ SQLite sang workload CSV

- `data/workload_real_sample_from_uploaded_sqlite.csv`  
  File mẫu mình đã tạo sẵn từ đúng file SQLite bạn upload để bạn nhìn format

- `outputs/workload_preview.png`  
  Hình xem nhanh workload mẫu

## Cách dùng ngắn gọn

### Bước 1. Cài thư viện
```bash
pip install -r requirements.txt
```

### Bước 2. Mở notebook đầu tiên
Mở:
```text
notebooks/01_sqlite_to_workload.ipynb
```

Rồi sửa đường dẫn `DB_PATH` thành file `.sqlite` của bạn, chạy từng cell.

Kết quả cuối cùng sẽ tạo ra:
```text
data/workload_real.csv
```

### Bước 3. Mở notebook thứ hai
Mở:
```text
notebooks/02_train_evaluate_ppo.ipynb
```

Notebook này sẽ:
- đọc `workload_real.csv`
- tạo môi trường RL
- train PPO
- so sánh với baseline (RoundRobin / BestFit / Threshold / Fixed)
- lưu bảng kết quả và biểu đồ

### Bước 4. Xuất biểu đồ phân tích PPO
Sau khi chạy xong một episode, dùng `env.trace`:

```python
from src.analysis_utils import trace_to_dataframe, export_core_plots

df_trace = trace_to_dataframe(env.trace)
paths = export_core_plots(df_trace, "outputs")
print(paths)
```

Sẽ tạo 3 biểu đồ:
- `demand_dvfs_over_time.png`
- `it_vs_facility_power.png`
- `pue_temperature_wear.png`

## Gợi ý cho AzurePackingTraceV1
Với file AzurePackingTraceV1, cách đơn giản và hợp lý nhất là:

- lấy bảng `vm` để biết VM sống từ lúc nào đến lúc nào
- lấy bảng `vmType` để biết loại VM dùng bao nhiêu `core`
- gộp `vmType` theo `vmTypeId`
- cộng tổng `core` của tất cả VM còn sống tại mỗi timestep

Kết quả là một chuỗi workload thật theo thời gian.

## Tại sao trong dữ liệu có starttime âm?
Trong file bạn upload, nhiều VM có `starttime < 0`. Điều đó thường được hiểu là:
- VM đã tồn tại trước khi cửa sổ trace bắt đầu
- nên khi dựng workload, ta coi các VM này là **đã active từ đầu cửa sổ quan sát**

Vì vậy notebook đang dùng cửa sổ:
- từ `0` đến `14` ngày
- mỗi timestep = `1 giờ`

=> tổng cộng `336` timestep.

## Lưu ý quan trọng
Bộ code này là **khung làm đúng hướng và chạy được**, nhưng để ra kết quả đẹp hơn bạn vẫn nên:
- tăng `total_timesteps`
- chỉnh reward nếu PPO chưa tiết kiệm điện tốt
- chạy nhiều seed khác nhau rồi lấy trung bình
- viết báo cáo giải thích cách dựng workload từ Azure
- bổ sung phần hiệu chỉnh PUE theo điều kiện DC thật nếu có dữ liệu đo thực

## Mẹo cho người mới
Nếu bạn là "tờ giấy trắng", hãy đi đúng thứ tự:
1. Chạy notebook 1 cho ra `workload_real.csv`
2. Nhìn biểu đồ workload xem có hợp lý không
3. Chạy notebook 2 để train
4. So sánh bảng metrics (energy/SLA/PUE/temp/migrations)
5. Xuất biểu đồ phân tích bằng `analysis_utils`
6. Mới bắt đầu chỉnh tham số

Chỉ cần đi đúng thứ tự này là không bị loạn.
