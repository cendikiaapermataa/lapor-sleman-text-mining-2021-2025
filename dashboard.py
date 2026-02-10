import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import os

st.set_page_config(
    page_title="Sistem Analisis Lapor Sleman",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        file_path = os.path.join(current_dir, 'Dataset', 'data_dashboard_sleman.csv')

        df = pd.read_csv(file_path)

        df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')

        df['Tahun'] = df['Tanggal'].dt.year
        df['Bulan'] = df['Tanggal'].dt.month_name()
        
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("⚠️ File 'data_dashboard_sleman.csv' TIDAK DITEMUKAN!")
    st.warning("Pastikan file CSV hasil notebook dan file 'dashboard.py' ini ada di folder yang sama.")
    st.stop()


with st.sidebar:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, 'images', 'logo kominfo.png')
    # --- Logo & Judul ---
    st.image(img_path, width=80)
    st.title("Lapor Sleman")
    st.write("Sistem Pendukung Keputusan")
    st.markdown("---")
    
    # --- Menu Navigasi ---
    menu = st.radio(
        "📂 PILIH MENU:", 
        ["🏠 Beranda (Summary)", "🔍 Analisis Masalah", "📝 Data Arsip"],
        index=0
    )
    
    st.markdown("---")
    
    # --- Filter Data (Dropdown) ---
    st.write("🛠️ **FILTER DATA**")
    
    all_years = sorted(df['Tahun'].dropna().unique().astype(int))
    # Opsi filter tahun + opsi semua tahun
    opsi_tahun = ["Semua Tahun"] + [str(y) for y in all_years]
    
    pilih_tahun = st.selectbox("Pilih Tahun Laporan:", opsi_tahun)
    
    # --- Logika Filter ---
    if pilih_tahun == "Semua Tahun":
        df_filtered = df.copy() 
        st.info("Menampilkan data: 2021 - 2025")
    else:
        df_filtered = df[df['Tahun'] == int(pilih_tahun)]
        st.success(f"Menampilkan data: Tahun {pilih_tahun}")

# ==========================================
# 4. HALAMAN 1: BERANDA (SUMMARY)
# ==========================================
if menu == "🏠 Beranda (Summary)":
    st.header("📊 Ringkasan Eksekutif")
    st.markdown("---")
    
    # --- A. Scorecards (CSS Anti-Potong) ---
    total_aduan = len(df_filtered)
    kategori_top = df_filtered['Kategori_Final'].mode()[0] if total_aduan > 0 else "-"
    media_top = df_filtered['Media'].mode()[0] if total_aduan > 0 else "-"
    bulan_sibuk = df_filtered['Bulan'].mode()[0] if total_aduan > 0 else "-"
    
    # Style CSS: Word-wrap aktif, Flexbox center, Tinggi tetap
    card_style = """
    <div style="
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #e6e6e6;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        height: 110px; 
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    ">
        <p style="
            font-size: 12px; 
            color: #666; 
            margin: 0 0 8px 0; 
            text-transform: uppercase; 
            letter-spacing: 0.5px;
        ">LABEL_TEXT</p>
        <p style="
            font-size: 15px; 
            font-weight: bold; 
            color: #2c3e50; 
            margin: 0; 
            line-height: 1.2;
            word-wrap: break-word;
            width: 100%;
        ">VALUE_TEXT</p>
    </div>
    """
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        html = card_style.replace("LABEL_TEXT", "Total Laporan").replace("VALUE_TEXT", f"{total_aduan:,}")
        st.markdown(html, unsafe_allow_html=True)
    with col2:
        html = card_style.replace("LABEL_TEXT", "Isu Dominan").replace("VALUE_TEXT", kategori_top)
        st.markdown(html, unsafe_allow_html=True)
    with col3:
        html = card_style.replace("LABEL_TEXT", "Media Favorit").replace("VALUE_TEXT", media_top)
        st.markdown(html, unsafe_allow_html=True)
    with col4:
        html = card_style.replace("LABEL_TEXT", "Bulan Tersibuk").replace("VALUE_TEXT", str(bulan_sibuk))
        st.markdown(html, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # --- B. Grafik Tren Waktu ---
    st.subheader("📈 Tren Laporan Masuk")
    
    if total_aduan > 0:
        trend = df_filtered.groupby(df_filtered['Tanggal'].dt.to_period("M")).size().reset_index(name='Jumlah')
        trend['Tanggal'] = trend['Tanggal'].dt.to_timestamp()
        
        fig_trend = px.line(trend, x='Tanggal', y='Jumlah', markers=True, 
                            title="Pergerakan Jumlah Aduan dari Waktu ke Waktu",
                            line_shape='spline', template='plotly_white')
        fig_trend.update_layout(height=350)
        
        # Key unik agar tidak error duplicate
        st.plotly_chart(fig_trend, use_container_width=True, key="trend_home")
    else:
        st.info("Data tidak tersedia untuk periode ini.")

# ==========================================
# 5. HALAMAN 2: ANALISIS MASALAH
# ==========================================
elif menu == "🔍 Analisis Masalah":
    st.header("🔍 Analisis Mendalam (AI Insights)")
    st.caption("Hasil clustering algoritma K-Means & Naive Bayes")
    st.markdown("---")
    
    # Cek Data Kosong
    if len(df_filtered) == 0:
        st.error("⚠️ Data Kosong di Tahun/Filter yang dipilih.")
        st.info("Saran: Coba ganti filter 'Tahun' menjadi 'Semua Tahun'.")
    else:
        # --- Bagian 1: Kategori ---
        st.subheader("1️⃣ Peta Permasalahan Daerah")
        col_kiri, col_kanan = st.columns([2, 1])
        
        with col_kiri:
            counts = df_filtered['Kategori_Final'].value_counts().reset_index()
            counts.columns = ['Kategori', 'Jumlah']
            counts = counts.sort_values(by='Jumlah', ascending=True)
            
            # Simpan isu terbanyak (Default)
            top_issue_default = counts.iloc[-1]['Kategori']
            
            fig_bar = px.bar(counts, x='Jumlah', y='Kategori', orientation='h',
                             text='Jumlah', title="Jumlah Laporan per Kategori",
                             color='Jumlah', color_continuous_scale='Reds')
            fig_bar.update_layout(xaxis_title="Jumlah Laporan", yaxis_title="")
            st.plotly_chart(fig_bar, use_container_width=True, key="bar_analisis")
            
        with col_kanan:
            fig_pie = px.pie(df_filtered, names='Kategori_Final', hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_traces(textposition='inside', textinfo='percent')
            fig_pie.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig_pie, use_container_width=True, key="pie_analisis")
        
        st.markdown("---")

        # --- Bagian 2: Media ---
        st.subheader("2️⃣ Asal Kanal Pengaduan")
        media_counts = df_filtered['Media'].value_counts().reset_index()
        media_counts.columns = ['Media', 'Jumlah']
        
        # [PENTING] Hitung top_media di sini agar tidak error
        top_media = media_counts.iloc[0]['Media']
        
        fig_media = px.bar(media_counts, x='Media', y='Jumlah', text='Jumlah',
                           color='Media', title="Distribusi Kanal Pengaduan",
                           color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig_media, use_container_width=True, key="media_analisis")
        
        st.markdown("---")
        
        # --- Bagian 3: Word Cloud ---
        st.subheader("3️⃣ Peta Kata Kunci (Word Cloud)")
        list_kategori = df['Kategori_Final'].unique()
        pilihan_wc = st.selectbox("Pilih Kategori untuk Word Cloud:", list_kategori, key="pilih_wc")
        
        teks_df = df_filtered[df_filtered['Kategori_Final'] == pilihan_wc]
        
        if len(teks_df) > 0:
            text_combined = " ".join(teks_df['Topik_Cleaned_Final'].astype(str))
            if len(text_combined) > 1:
                # Warna 'Reds' yang valid agar tidak error firebrick
                wc = WordCloud(width=1200, height=400, background_color='white', 
                               colormap='Reds', max_words=150, 
                               contour_width=1, contour_color='darkred').generate(text_combined)
                
                fig_wc, ax = plt.subplots(figsize=(12, 5))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig_wc)
            else:
                st.warning("Teks terlalu sedikit.")
        else:
            st.warning("Tidak ada data teks.")

        st.markdown("---")
        
        # =========================================================
        # 🔥 FITUR REKOMENDASI INTERAKTIF 🔥
        # =========================================================
        st.header("💡 Rekomendasi & Solusi Strategis")
        st.caption("Pilih masalah di bawah untuk melihat analisis AI berdasarkan kata kunci dominan.")
        
        # 1. FITUR PILIH MASALAH (DROPDOWN)
        try:
            index_default = list(list_kategori).index(top_issue_default)
        except:
            index_default = 0
            
        pilih_solusi = st.selectbox("🎯 Pilih Masalah yang Ingin Dianalisis:", 
                                    list_kategori, 
                                    index=index_default,
                                    key="pilih_solusi_ai")
        
        # 2. LOGIKA KATA KUNCI DINAMIS
        df_isu_terpilih = df_filtered[df_filtered['Kategori_Final'] == pilih_solusi]
        all_text_isu = " ".join(df_isu_terpilih['Topik_Cleaned_Final'].astype(str)).split()
        
        if len(all_text_isu) > 0:
            most_common = Counter(all_text_isu).most_common(5) 
            hot_topics = ", ".join([kata[0] for kata in most_common])
        else:
            hot_topics = "tidak ada data spesifik"

        # 3. TAMPILKAN SOLUSI
        with st.container():
            st.subheader(f"📢 Analisis & Solusi: {pilih_solusi}")
            
            # Logika Percabangan Solusi
            if "Jalan" in pilih_solusi:
                st.info(f"""
                **Analisis Kata Kunci:**
                Keluhan jalan pada kategori ini didominasi kata: **{hot_topics}**.
                
                **Rekomendasi Aksi:**
                1. Prioritaskan perbaikan di lokasi yang muncul pada kata kunci (misal: Godean, Kaliurang, dll).
                2. Jika kata 'lubang' dominan, lakukan penambalan cepat (patching).
                3. Jika kata 'air/drainase' muncul, segera normalisasi saluran air di sekitar jalan tersebut.
                """)
            elif "Penerangan" in pilih_solusi or "Lampu" in pilih_solusi:
                st.info(f"""
                **Analisis Kata Kunci:**
                Isu penerangan ini berpusat pada kata: **{hot_topics}**.
                
                **Rekomendasi Aksi:**
                1. Jadikan lokasi yang disebut di atas sebagai target patroli malam ini.
                2. Cek komponen timer/listrik jika kata 'mati' dan 'total' sering muncul.
                3. Pastikan stok lampu LED tersedia untuk penggantian segera.
                """)
            elif "Lingkungan" in pilih_solusi or "Sampah" in pilih_solusi:
                st.info(f"""
                **Analisis Kata Kunci:**
                Masalah lingkungan ini terkait dengan: **{hot_topics}**.
                
                **Rekomendasi Aksi:**
                1. Jika ada kata 'bakar', lakukan sosialisasi larangan pembakaran sampah.
                2. Jika kata 'pohon' dominan, siapkan tim perantingan pohon (pemangkasan).
                3. Koordinasi dengan Satpol PP/DLH untuk penertiban sesuai kata kunci.
                """)
            else:
                st.info(f"**Saran Umum:** Tindak lanjuti laporan dengan fokus pada kata kunci dominan: **{hot_topics}**.")
            
            # Tambahan info kecil
            st.caption(f"ℹ️ Solusi ini berdasarkan analisis terhadap laporan yang masuk di kategori **{pilih_solusi}**.")

            # 2. SOLUSI MEDIA & WAKTU
            col_saran1, col_saran2 = st.columns(2)
            
            with col_saran1:
                st.warning(f"📱 **Saluran Komunikasi Utama: {top_media}**")
                if top_media in ["Instagram", "Twitter", "Facebook"]:
                    st.write(f"Warga aktif di **{top_media}**. Buat konten edukasi visual (Infografis/Video) tentang progres perbaikan.")
                else:
                    st.write(f"Maksimalkan respons admin di **{top_media}**. Pertahankan waktu respon (SLA) < 24 jam.")
                    
            with col_saran2:
                st.success("📅 **Waspada Waktu Sibuk**")
                bulan_sibuk_ini = df_filtered['Bulan'].mode()[0]
                st.write(f"Data historis menunjukkan lonjakan aduan tertinggi di bulan **{bulan_sibuk_ini}**. Siapkan tim piket ekstra.")

# ==========================================
# 6. HALAMAN 3: ARSIP DATA (DENGAN DOWNLOAD)
# ==========================================
elif menu == "📝 Data Arsip":
    st.header("📂 Arsip Data Aduan")
    
    # --- FITUR DOWNLOAD CSV ---
    # Konversi data tabel saat ini ke format CSV
    # Kita encode 'utf-8' biar aman
    csv_data = df_filtered.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Data (CSV)",
        data=csv_data,
        file_name='laporan_aduan_sleman.csv',
        mime='text/csv',
        help="Klik untuk mengunduh data yang sedang ditampilkan."
    )
    
    st.markdown("---")
    
    # --- FITUR PENCARIAN ---
    search_term = st.text_input("🔍 Cari kata kunci:", key="search_arsip")
    
    if search_term:
        # Filter berdasarkan kata kunci yang diketik user
        df_display = df_filtered[df_filtered['Topik'].str.contains(search_term, case=False, na=False)]
    else:
        df_display = df_filtered
    
    # Tampilkan Tabel
    st.dataframe(df_display[['Tanggal', 'Media', 'Topik', 'Kategori_Final']], 
                 use_container_width=True, height=500)
    st.caption(f"Menampilkan {len(df_display)} baris data.")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: grey;'>
        <small>Sistem Pendukung Keputusan Dinas Kominfo Sleman | Dikembangkan dengan Streamlit & Python</small>
    </div>
    """, unsafe_allow_html=True)