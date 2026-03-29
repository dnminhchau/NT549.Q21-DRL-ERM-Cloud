# NT549.Q21 - Deep Reinforcement Learning for Energy-efficient Resource Management in Cloud Computing
Đồ án này được thực hiện phục vụ mục đích học thuật trong khuôn khổ môn học tại **Trường Đại học Công nghệ Thông tin - ĐHQG TP.HCM**. Mọi nội dung, mã nguồn và kết quả nghiên cứu chỉ mang tính chất tham khảo và không được sử dụng cho mục đích thương mại.

## 📋 Mô tả dự án

**Chủ đề:** Ứng dụng Deep Reinforcement Learning để tối ưu hóa tiêu thụ năng lượng và quản lý tài nguyên động trong hệ thống Cloud Computing.

### 🎯 Nội dung dự kiến

- Xây dựng mô hình tiêu thụ năng lượng (**Energy Model**) dựa trên trạng thái CPU (DVFS) và trạng thái hoạt động của máy chủ (Active/Sleep/Off).
- Thiết kế và huấn luyện tác nhân **Deep Reinforcement Learning** cho bài toán điều phối tải công việc và hợp nhất máy ảo (VM Consolidation).
- Ứng dụng RL trong **quản lý nguồn động** (Power Management), điều chỉnh tần số và điện áp CPU nhằm giảm điện năng tiêu thụ trong các giai đoạn tải thấp.
- Xây dựng **hàm phần thưởng đa mục tiêu** (Multi-objective Reward) cân bằng giữa tiết kiệm năng lượng, độ trễ hệ thống và cam kết SLA.
- Thử nghiệm và đánh giá mô hình trên các bộ dữ liệu workload thực tế như **Google Cluster Data** hoặc **Azure Public Dataset**.

### 📈 Kết quả dự kiến

- Giảm từ **15–30%** tổng điện năng tiêu thụ so với các thuật toán điều phối truyền thống (Round Robin, Best Fit).
- Cải thiện chỉ số **PUE** (Power Usage Effectiveness), nâng cao hiệu quả sử dụng năng lượng toàn hệ thống.
- Đảm bảo chất lượng dịch vụ với tỷ lệ vi phạm **SLA** thấp hoặc tương đương các phương pháp hiện tại.
- Giảm nhiệt độ hoạt động và kéo dài tuổi thọ phần cứng nhờ phân bổ tải và quản lý nguồn thông minh.
- Xây dựng hệ thống giám sát trực quan thể hiện mối quan hệ giữa tải hệ thống, số máy chủ đang hoạt động và mức năng lượng tiết kiệm theo thời gian thực.

---

## 📦 Dataset

Do kích thước file quá lớn (>100MB), dataset không được lưu trực tiếp trong repository.

👉 Tải dataset tại: https://drive.google.com/drive/folders/1cdP3E-oYdnmFC0NTsnBnOJTCsm5xvUZY?usp=sharing

---

## 👥 Thành viên nhóm

| STT | Họ và tên | MSSV | Email |
|-----|-----------|------|-------|
| 1 | Đoàn Ngọc Minh Châu | 23521068 | 235120168@gm.uit.edu.vn |
| 2 | Huỳnh Thị Phương Nghi | 23521001 | 23521001@gm.uit.edu.vn |
| 3 | Võ Thị Hồng Phúc | 23521226 | 23521226@gm.uit.edu.vn |
