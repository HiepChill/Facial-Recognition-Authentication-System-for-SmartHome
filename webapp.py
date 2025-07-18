import streamlit as st
import requests
import pandas as pd
from PIL import Image
from datetime import datetime, date
import json
import threading
import subprocess
import sys
import time
import os

# Cấu hình trang
st.set_page_config(
    page_title="Smart Home Security System",
    page_icon="🏠",
    layout="wide"
)

# URL API backend
API_BASE_URL = "http://127.0.0.1:8080"

# CSS đơn giản
st.markdown("""
<style>
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def show_success(message):
    st.markdown(f'<div class="success-box">✅ {message}</div>', unsafe_allow_html=True)

def show_error(message):
    st.markdown(f'<div class="error-box">❌ {message}</div>', unsafe_allow_html=True)

def show_info(message):
    st.markdown(f'<div class="info-box">ℹ️ {message}</div>', unsafe_allow_html=True)

def check_api_server():
    """Kiểm tra API server có đang chạy không"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_api_server():
    """Khởi động API server"""
    try:
        # Chạy uvicorn server
        subprocess.Popen([
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", "127.0.0.1", 
            "--port", "8080",
            "--reload"
        ])
        return True
    except Exception as e:
        st.error(f"Lỗi khởi động API server: {e}")
        return False

def api_call(method, endpoint, data=None, files=None):
    """Gọi API với error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, params=data, timeout=10)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files, timeout=30)
            else:
                response = requests.post(url, data=data, timeout=10)
        elif method == "PUT":
            response = requests.put(url, data=data, timeout=10)
        elif method == "DELETE":
            response = requests.delete(url, timeout=10)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get("detail", "Lỗi không xác định")
    except requests.exceptions.ConnectionError:
        return False, "Không thể kết nối đến server"
    except requests.exceptions.Timeout:
        return False, "Server phản hồi quá chậm"
    except Exception as e:
        return False, str(e)

# Header
st.title("🏠 Hệ thống an ninh thông minh")
st.write("---")

# Kiểm tra trạng thái API server
api_status = check_api_server()

if not api_status:
    st.warning("⚠️ API Server chưa chạy!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Khởi động API Server", type="primary"):
            with st.spinner("Đang khởi động API server..."):
                if start_api_server():
                    show_info("Đã gửi lệnh khởi động API server. Vui lòng đợi 5-10 giây rồi refresh trang.")
                    time.sleep(3)
                    st.rerun()
    
    with col2:
        if st.button("🔄 Kiểm tra lại"):
            st.rerun()
    
    st.info("""
    **Hướng dẫn khởi động thủ công:**
    1. Mở Terminal/Command Prompt mới
    2. Chạy lệnh: `uvicorn main:app --host 127.0.0.1 --port 8080 --reload`
    3. Refresh trang này
    """)
    
    # Không hiển thị menu nếu API chưa sẵn sàng
    st.stop()

# Sidebar menu
menu = st.sidebar.selectbox("Chọn chức năng:", [
    "📊 Tổng quan", 
    "👥 Quản lý người dùng", 
    "📸 Ảnh khuôn mặt", 
    "🎥 Camera trực tiếp", 
    "📋 Lịch sử"
])

# Hiển thị trạng thái API ở sidebar
with st.sidebar:
    st.write("---")
    if api_status:
        st.success("🟢 API Server: Hoạt động")
    else:
        st.error("🔴 API Server: Offline")
    
    if st.button("🔄 Refresh", key="sidebar_refresh"):
        st.rerun()

if menu == "📊 Tổng quan":
    st.header("📊 Tổng quan hệ thống")
    
    # Kiểm tra kết nối API
    success, data = api_call("GET", "/")
    
    if success:
        st.success("🟢 API Server đang hoạt động")
        
        # Lấy thống kê người dùng
        success_users, users_data = api_call("GET", "/users")
        
        if success_users:
            users = users_data.get("users", [])
            total_users = len(users)
            total_images = sum(user.get("image_count", 0) for user in users)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("👥 Số người dùng", total_users)
            with col2:
                st.metric("📸 Tổng ảnh", total_images)
            with col3:
                # Lấy lịch sử hôm nay
                today = datetime.now().strftime("%Y-%m-%d")
                success_history, history_data = api_call("GET", f"/recognition/history?date={today}")
                detections_today = len(history_data.get("history", [])) if success_history else 0
                st.metric("🔍 Nhận diện hôm nay", detections_today)
            
            # Hiển thị danh sách người dùng
            if users:
                st.subheader("👥 Danh sách người dùng")
                for user in users:
                    with st.expander(f"👤 {user['name']} (ID: {user['id']})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Số ảnh:** {user['image_count']}")
                        with col2:
                            st.write(f"**Ngày tạo:** {user['created_at'][:10]}")
                        if user.get('notes'):
                            st.write(f"**Ghi chú:** {user['notes']}")
            else:
                st.info("Chưa có người dùng nào trong hệ thống")
        else:
            show_error(f"Không thể tải danh sách người dùng: {users_data}")
    else:
        show_error(f"Không thể kết nối API Server: {data}")

elif menu == "👥 Quản lý người dùng":
    st.header("👥 Quản lý người dùng")
    
    tab1, tab2 = st.tabs(["➕ Thêm người dùng", "📝 Sửa/Xóa"])
    
    with tab1:
        st.subheader("➕ Thêm người dùng mới")
        
        with st.form("add_user"):
            col1, col2 = st.columns(2)
            with col1:
                user_id = st.text_input("ID người dùng *", placeholder="VD: 01")
            with col2:
                user_name = st.text_input("Tên *", placeholder="VD: Nguyễn Văn A")
            
            notes = st.text_area("Ghi chú", placeholder="Thông tin thêm...")
            
            if st.form_submit_button("➕ Tạo người dùng", type="primary"):
                if user_id and user_name:
                    with st.spinner("Đang tạo người dùng..."):
                        success, response = api_call("POST", "/users", {
                            "user_id": user_id,
                            "name": user_name,
                            "notes": notes
                        })
                    
                    if success:
                        show_success(f"Đã tạo người dùng {user_name} thành công!")
                        time.sleep(3)
                        st.rerun()
                    else:
                        show_error(f"Lỗi: {response}")
                else:
                    show_error("Vui lòng điền đầy đủ ID và tên!")
    
    with tab2:
        st.subheader("📝 Sửa/Xóa người dùng")
        
        # Lấy danh sách người dùng
        with st.spinner("Đang tải danh sách người dùng..."):
            success, users_data = api_call("GET", "/users")
        
        if success and users_data.get("users"):
            users = users_data["users"]
            user_names = [f"{user['id']} - {user['name']}" for user in users]
            
            selected = st.selectbox("Chọn người dùng:", user_names)
            
            if selected:
                user_id = selected.split(" - ")[0]
                
                # Lấy thông tin chi tiết
                success, user_detail = api_call("GET", f"/users/{user_id}")
                
                if success:
                    user_info = user_detail["user"]
                    
                    with st.form("edit_user"):
                        new_name = st.text_input("Tên mới:", value=user_info["name"])
                        new_notes = st.text_area("Ghi chú mới:", value=user_info.get("notes", ""))
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.form_submit_button("💾 Cập nhật", type="primary"):
                                with st.spinner("Đang cập nhật..."):
                                    success, response = api_call("PUT", f"/users/{user_id}", {
                                        "name": new_name,
                                        "notes": new_notes
                                    })
                                
                                if success:
                                    show_success("Cập nhật thành công!")
                                    time.sleep(3)
                                    st.rerun()
                                else:
                                    show_error(f"Lỗi: {response}")
                        
                        with col2:
                            if st.form_submit_button("🗑️ Xóa người dùng"):
                                with st.spinner("Đang xóa..."):
                                    success, response = api_call("DELETE", f"/users/{user_id}")
                                
                                if success:
                                    show_success(f"Đã xóa {user_info['name']} thành công!")
                                    time.sleep(3)
                                    st.rerun()
                                else:
                                    show_error(f"Lỗi: {response}")
                else:
                    show_error("Không thể tải thông tin người dùng")
        else:
            st.info("Chưa có người dùng nào")

elif menu == "📸 Ảnh khuôn mặt":
    st.header("📸 Quản lý ảnh khuôn mặt")
    
    # Lấy danh sách người dùng
    with st.spinner("Đang tải danh sách người dùng..."):
        success, users_data = api_call("GET", "/users")
    
    if success and users_data.get("users"):
        users = users_data["users"]
        user_names = [f"{user['id']} - {user['name']}" for user in users]
        
        selected_user = st.selectbox("Chọn người dùng:", user_names)
        
        if selected_user:
            user_id = selected_user.split(" - ")[0]
            
            tab1, tab2 = st.tabs(["📤 Upload ảnh", "🖼️ Xem ảnh"])
            
            with tab1:
                st.subheader("📤 Upload ảnh khuôn mặt")
                
                uploaded_files = st.file_uploader(
                    "Chọn ảnh (có thể chọn nhiều):",
                    type=['jpg', 'jpeg', 'png'],
                    accept_multiple_files=True
                )
                
                if uploaded_files:
                    # Preview ảnh
                    st.write("🔍 Ảnh đã chọn:")
                    cols = st.columns(min(len(uploaded_files), 4))
                    
                    for idx, file in enumerate(uploaded_files):
                        with cols[idx % 4]:
                            image = Image.open(file)
                            st.image(image, caption=file.name, width=150)
                    
                    if st.button("📤 Upload tất cả", type="primary"):
                        results = []
                        progress = st.progress(0)
                        status = st.empty()
                        
                        for idx, file in enumerate(uploaded_files):
                            status.text(f"Đang upload {idx + 1}/{len(uploaded_files)}: {file.name}")
                            file.seek(0)
                            files = {"face_images": (file.name, file.getvalue(), file.type)}
                            
                            success, response = api_call("POST", f"/users/{user_id}/faces", files=files)
                            results.append((file.name, success, response))
                            
                            progress.progress((idx + 1) / len(uploaded_files))
                        
                        status.empty()
                        progress.empty()
                        
                        # Phân tích chi tiết kết quả
                        success_count = 0
                        error_count = 0
                        
                        for filename, api_success, response in results:
                            if api_success and response.get("status") == "success":
                                # Kiểm tra chi tiết từng ảnh trong results
                                file_results = response.get("results", [])
                                for file_result in file_results:
                                    if file_result.get("status") == "success":
                                        success_count += 1
                                    else:
                                        error_count += 1
                                        show_error(f"Lỗi {filename}: {file_result.get('message', 'Lỗi không xác định')}")
                            else:
                                error_count += 1
                                show_error(f"Lỗi API {filename}: {response}")
                        
                        # Hiển thị thông báo tổng hợp
                        total_files = len(uploaded_files)
                        if success_count > 0:
                            show_success(f"Upload thành công {success_count}/{total_files} ảnh!")
                        if error_count > 0:
                            show_error(f"Thất bại {error_count}/{total_files} ảnh!")
                        if success_count == 0:
                            show_error("Không có ảnh nào được upload thành công!")
                        
                        if success_count > 0:
                            time.sleep(10)
                            st.rerun()
            
            with tab2:
                st.subheader("🖼️ Ảnh hiện có")
                
                with st.spinner("Đang tải danh sách ảnh..."):
                    success, faces_data = api_call("GET", f"/users/{user_id}/faces")
                
                if success and faces_data.get("face_images"):
                    face_images = faces_data["face_images"]
                    st.write(f"📊 Tổng: {len(face_images)} ảnh")
                    
                    # Hiển thị danh sách ảnh
                    for face in face_images:
                        col1, col2, col3 = st.columns([3, 2, 1])
                        
                        with col1:
                            filename = os.path.basename(face['image_path']).replace('.jpg', '')
                            st.write(f"🖼️ {filename}")
                            #st.write(f"🖼️ Ảnh #{face['id']}")
                        
                        with col2:
                            st.write(f"📅 {face['created_at'][:16]}")
                        
                        with col3:
                            if st.button("🗑️", key=f"del_{face['id']}"):
                                with st.spinner("Đang xóa..."):
                                    success, response = api_call("DELETE", f"/faces/{face['id']}")
                                
                                if success:
                                    show_success("Đã xóa ảnh!")
                                    time.sleep(3)
                                    st.rerun()
                                else:
                                    show_error(f"Lỗi: {response}")
                else:
                    st.info("Chưa có ảnh nào")
    else:
        st.warning("Chưa có người dùng. Tạo người dùng trước!")

elif menu == "🎥 Camera trực tiếp":
    st.header("🎥 Camera trực tiếp")
    
    # Kiểm tra API
    success, _ = api_call("GET", "/")
    
    if success:
        st.info("📹 Xem camera trực tiếp với nhận diện khuôn mặt")
        
        # Hiển thị stream
        stream_url = f"{API_BASE_URL}/camera/stream"
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            try:
                st.markdown(f"""
                <img src="{stream_url}" 
                     style="width: 100%; border-radius: 10px; border: 2px solid #ddd;">
                """, unsafe_allow_html=True)
            except:
                st.error("Không thể hiển thị camera. Kiểm tra camera đã kết nối chưa.")
        
        with col2:
            st.write("**🎛️ Điều khiển**")
            
            if st.button("🔄 Làm mới"):
                st.rerun()
            
            st.write("**ℹ️ Thông tin**")
            st.write("- Khung màu xanh: Người quen")
            st.write("- Khung màu đỏ: Người lạ")
            st.write("- Số trong ngoặc: Độ tin cậy")
            
            st.write("**⚠️ Lưu ý**")
            st.write("Đảm bảo camera đã được kết nối với máy tính")
    else:
        st.error("Không thể kết nối API")

elif menu == "📋 Lịch sử":
    st.header("📋 Lịch sử nhận diện")
    
    # Chọn ngày
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_date = st.date_input("📅 Chọn ngày:", value=date.today())
    
    with col2:
        search_date = selected_date.strftime("%Y-%m-%d")
    
    # Lấy lịch sử
    with st.spinner("Đang tải lịch sử..."):
        success, history_data = api_call("GET", f"/recognition/history?date={search_date}")
    
    if success:
        history = history_data.get("history", [])
        
        if history:
            st.success(f"📊 Tìm thấy {len(history)} sự kiện ngày {search_date}")
            
            # Thống kê
            known = [h for h in history if h.get("user_id") != "unknown"]
            unknown = [h for h in history if h.get("user_id") == "unknown"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("👥 Người quen", len(known))
            
            with col2:
                st.metric("❓ Người lạ", len(unknown))
            
            with col3:
                st.metric("📊 Tổng", len(history))
            
            # Bảng chi tiết
            st.subheader("📋 Chi tiết")
            
            df = pd.DataFrame(history)
            
            if not df.empty:
                # Sắp xếp theo thời gian
                df = df.sort_values('timestamp', ascending=False)
                
                # Hiển thị bảng
                st.dataframe(df, use_container_width=True)
                
                # Download CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Tải CSV",
                    csv,
                    f"history_{search_date}.csv",
                    "text/csv"
                )
        else:
            st.info(f"📭 Không có dữ liệu ngày {search_date}")
    else:
        show_error(f"Lỗi: {history_data}")

# Footer
st.write("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.write("🏠 Smart Home Security System")

with col2:
    if st.button("🚀 Khởi động lại API"):
        start_api_server()
        time.sleep(2)
        st.rerun()

with col3:
    st.write("Powered by FastAPI + Streamlit")