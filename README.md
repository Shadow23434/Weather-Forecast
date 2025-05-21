# WeatherForecast

Dự án **WeatherForecast** là một ứng dụng web dự báo thời tiết sử dụng Django, tích hợp các mô hình học máy (ML) để dự đoán và phân tích dữ liệu thời tiết.

## 🚀 Tính năng chính
- Dự báo thời tiết theo thành phố.
- Lấy dữ liệu thời tiết từ OpenWeatherMap, NOAA, Open-Meteo.
- Dự đoán nhiệt độ, mưa, v.v. bằng các mô hình ML: RandomForest, XGBoost, LightGBM, LSTM (TensorFlow).
- Giao diện web đơn giản, dễ sử dụng.

## 🛠️ Cài đặt

### 1. Clone project
```bash
git clone https://github.com/yourusername/WeatherForecast.git
cd WeatherForecast
```

### 2. Tạo và kích hoạt môi trường ảo
```bash
python -m venv myenv
# Windows PowerShell
.\myenv\Scripts\Activate
# hoặc CMD
myenv\Scripts\activate.bat
```

### 3. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### 4. Tạo file `.env` trong thư mục gốc và thêm các biến môi trường:
```env
OPEN_WEATHER_API_KEY=your_openweather_api_key
CURRENT_WEATHER_URL=https://api.openweathermap.org/data/2.5/
HISTORICAL_WEATHER_URL=https://archive-api.open-meteo.com/v1/archive
FLAG_URL=https://flagcdn.com/32x24/
NOAA_API_TOKEN=your_noaa_token
NOAA_API_BASE_URL=https://www.ncei.noaa.gov/cdo-web/api/v2/
NOAA_DATASET_ID=GHCND
```

### 5. Khởi tạo database (nếu cần)
```bash
python manage.py migrate
```

### 6. Chạy server
```bash
python manage.py runserver
```
Truy cập: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

---

## 📁 Cấu trúc thư mục

```
WeatherForecast/
│
├── myenv/                # Môi trường ảo (không commit lên git)
├── weatherForecast/      # Mã nguồn Django
│   ├── forecast/         # App chính: xử lý logic, ML, API
│   ├── weatherForecast/  # Cấu hình project Django
│   ├── manage.py
│   └── ...
├── data/                 # Dữ liệu thời tiết (csv, v.v.)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧑‍💻 Công nghệ sử dụng
- Django
- Pandas, Numpy
- scikit-learn, XGBoost, LightGBM, TensorFlow
- Requests
- python-environ

---

## 📌 Ghi chú
- Không commit thư mục `myenv/`, file `.env`, hoặc dữ liệu nhạy cảm lên git.
- Đảm bảo đã cài đủ các package trong `requirements.txt`.
  ```

---

## 📄 License
MIT License

---

**Nếu bạn cần thêm hướng dẫn chi tiết hoặc gặp lỗi khi cài đặt, hãy liên hệ!** 