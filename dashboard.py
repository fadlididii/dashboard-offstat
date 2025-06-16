import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Analisis IPM, TPT & Kemiskinan Indonesia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk styling seperti Tableau/Power BI
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    .metric-container:hover {
        transform: translateY(-5px);
    }
    .filter-section {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
    }
    .stMultiSelect > div > div {
        background-color: white;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    try:
        # Load data dari file terpisah dengan header yang tepat
        ipm_df = pd.read_csv('ipm.csv', sep=';', header=None, names=['Provinsi', 'IPM', 'Tahun'])
        tpt_df = pd.read_csv('tpt.csv', header=None, names=['Provinsi', 'TPT', 'Tahun'])
        kemiskinan_df = pd.read_csv('kemiskinan.csv', sep=';', header=None, names=['Provinsi', 'Kemiskinan', 'Tahun'])
        
        # Standardisasi nama provinsi untuk konsistensi
        def standardize_province_name(name):
            name = str(name).upper().strip()
            # Mapping untuk nama provinsi yang berbeda
            province_mapping = {
                'KEP. BANGKA BELITUNG': 'KEPULAUAN BANGKA BELITUNG',
                'KEPULAUAN BANGKA BELITUNG': 'KEPULAUAN BANGKA BELITUNG',
                'KEP. RIAU': 'KEPULAUAN RIAU',
                'KEPULAUAN RIAU': 'KEPULAUAN RIAU',
                'DKI JAKARTA': 'DKI JAKARTA',
                'DAERAH KHUSUS IBUKOTA JAKARTA': 'DKI JAKARTA',
                'DI YOGYAKARTA': 'DI YOGYAKARTA',
                'DAERAH ISTIMEWA YOGYAKARTA': 'DI YOGYAKARTA',
                'SULAWESI TENGAH': 'SULAWESI TENGAH',
                'SULAWESI TENGGARA': 'SULAWESI TENGGARA',
                'SULAWESI SELATAN': 'SULAWESI SELATAN',
                'SULAWESI UTARA': 'SULAWESI UTARA',
                'SULAWESI BARAT': 'SULAWESI BARAT',
                'GORONTALO': 'GORONTALO'
            }
            return province_mapping.get(name, name)
        
        # Terapkan standardisasi nama provinsi
        ipm_df['Provinsi'] = ipm_df['Provinsi'].apply(standardize_province_name)
        tpt_df['Provinsi'] = tpt_df['Provinsi'].apply(standardize_province_name)
        kemiskinan_df['Provinsi'] = kemiskinan_df['Provinsi'].apply(standardize_province_name)
        
        # Merge data berdasarkan Provinsi dan Tahun
        merged_df = ipm_df.merge(tpt_df, on=['Provinsi', 'Tahun'], how='inner')
        merged_df = merged_df.merge(kemiskinan_df, on=['Provinsi', 'Tahun'], how='inner')
        
        # Remove rows with missing values
        merged_df = merged_df.dropna()
        
        # Konversi tipe data numerik
        merged_df['IPM'] = pd.to_numeric(merged_df['IPM'], errors='coerce')
        merged_df['TPT'] = pd.to_numeric(merged_df['TPT'], errors='coerce')
        merged_df['Kemiskinan'] = pd.to_numeric(merged_df['Kemiskinan'], errors='coerce')
        merged_df['Tahun'] = pd.to_numeric(merged_df['Tahun'], errors='coerce')
        
        # Remove rows with invalid numeric values
        merged_df = merged_df.dropna()
        
        # Preprocessing: Konversi tahun ke integer untuk konsistensi tampilan
        merged_df['Tahun'] = merged_df['Tahun'].astype(int)
        
        return merged_df
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e}")
        st.error("Pastikan file ipm.csv, tpt.csv, dan kemiskinan.csv ada di direktori yang sama dengan dashboard.py")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Fungsi untuk membuat line chart
def create_line_chart(df, metric, title, color):
    yearly_avg = df.groupby('Tahun')[metric].mean().reset_index()
    
    fig = px.line(
        yearly_avg, 
        x='Tahun', 
        y=metric,
        title=title,
        markers=True,
        line_shape='spline'
    )
    
    fig.update_traces(
        line=dict(color=color, width=4),
        marker=dict(size=10, color=color)
    )
    
    fig.update_layout(
        title_font_size=18,
        title_x=0.5,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray',
        dtick=1,
        tickformat='d'
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

# Fungsi untuk membuat bar chart
def create_bar_chart(df, metric, title, color, top_n=10):
    avg_by_province = df.groupby('Provinsi')[metric].mean().sort_values(ascending=False).head(top_n)
    
    fig = px.bar(
        x=avg_by_province.values,
        y=avg_by_province.index,
        orientation='h',
        title=f"{title} (Top {top_n} Provinsi)",
        color=avg_by_province.values,
        color_continuous_scale=color
    )
    
    fig.update_layout(
        title_font_size=18,
        title_x=0.5,
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title=metric,
        yaxis_title="Provinsi",
        font=dict(size=12)
    )
    
    return fig

# Fungsi untuk membuat scatter plot
def create_scatter_plot(df, x_col, y_col, title):
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col,
        color='Tahun',
        hover_name='Provinsi',
        title=title,
        size_max=15,
        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
    )
    
    # Tambahkan trendline
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=np.poly1d(np.polyfit(df[x_col], df[y_col], 1))(df[x_col]),
            mode='lines',
            name='Trendline',
            line=dict(color='red', width=2, dash='dash')
        )
    )
    
    fig.update_layout(
        title_font_size=18,
        title_x=0.5,
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

# Fungsi untuk membuat heatmap korelasi
def create_correlation_heatmap(df):
    corr_matrix = df[['IPM', 'TPT', 'Kemiskinan']].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Heatmap Korelasi Antar Variabel",
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1
    )
    
    fig.update_layout(
        title_font_size=18,
        title_x=0.5,
        height=400,
        font=dict(size=12)
    )
    
    return fig

# Fungsi untuk membuat multi-line chart per provinsi
def create_province_trends(df, selected_provinces):
    if not selected_provinces:
        return None
    
    filtered_df = df[df['Provinsi'].isin(selected_provinces)]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Tren IPM', 'Tren TPT', 'Tren Kemiskinan'),
        horizontal_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set3
    
    for i, province in enumerate(selected_provinces):
        province_data = filtered_df[filtered_df['Provinsi'] == province]
        color = colors[i % len(colors)]
        
        # IPM
        fig.add_trace(
            go.Scatter(
                x=province_data['Tahun'],
                y=province_data['IPM'],
                mode='lines+markers',
                name=f'{province} - IPM',
                line=dict(color=color, width=3),
                marker=dict(size=8),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # TPT
        fig.add_trace(
            go.Scatter(
                x=province_data['Tahun'],
                y=province_data['TPT'],
                mode='lines+markers',
                name=f'{province} - TPT',
                line=dict(color=color, width=3),
                marker=dict(size=8),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Kemiskinan
        fig.add_trace(
            go.Scatter(
                x=province_data['Tahun'],
                y=province_data['Kemiskinan'],
                mode='lines+markers',
                name=f'{province} - Kemiskinan',
                line=dict(color=color, width=3),
                marker=dict(size=8),
                showlegend=False
            ),
            row=1, col=3
        )
    
    fig.update_layout(
        title_text="Tren Perkembangan Indikator per Provinsi (2022-2024)",
        title_font_size=18,
        title_x=0.5,
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Format sumbu x untuk menampilkan tahun tanpa desimal
    fig.update_xaxes(
        dtick=1,
        tickformat='d'
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Dashboard Analisis IPM, TPT & Kemiskinan Indonesia</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Filter Section
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    st.markdown("### 🔍 Filter Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_years = st.multiselect(
            "📅 Pilih Tahun",
            options=sorted(df['Tahun'].unique()),
            default=sorted(df['Tahun'].unique())
        )
    
    with col2:
        selected_provinces = st.multiselect(
            "🏛️ Pilih Provinsi",
            options=sorted(df['Provinsi'].unique()),
            default=[]
        )
    
    with col3:
        show_top_n = st.selectbox(
            "📊 Tampilkan Top N Provinsi",
            options=[5, 10, 15, 20],
            index=1
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Filter data
    filtered_df = df[df['Tahun'].isin(selected_years)]
    if selected_provinces:
        filtered_df = filtered_df[filtered_df['Provinsi'].isin(selected_provinces)]
    
    # Statistik Deskriptif
    st.markdown('<div class="section-header">📋 Statistik Deskriptif</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = ['IPM', 'TPT', 'Kemiskinan']
    colors = ['#3498db', '#e74c3c', '#f39c12']
    
    for i, metric in enumerate(metrics):
        with [col1, col2, col3][i]:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric(f"📊 {metric}", "")
            st.write(f"**Mean:** {filtered_df[metric].mean():.2f}")
            st.write(f"**Min:** {filtered_df[metric].min():.2f}")
            st.write(f"**Max:** {filtered_df[metric].max():.2f}")
            st.write(f"**Std:** {filtered_df[metric].std():.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("📈 Total Data", len(filtered_df))
        st.write(f"**Provinsi:** {filtered_df['Provinsi'].nunique()}")
        st.write(f"**Tahun:** {filtered_df['Tahun'].nunique()}")
        st.write(f"**Periode:** {filtered_df['Tahun'].min()}-{filtered_df['Tahun'].max()}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tren Temporal
    st.markdown('<div class="section-header">📈 Tren Perkembangan (2022-2024)</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_ipm = create_line_chart(df, 'IPM', 'Rata-rata IPM per Tahun', '#3498db')
        st.plotly_chart(fig_ipm, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_tpt = create_line_chart(df, 'TPT', 'Rata-rata TPT per Tahun', '#e74c3c')
        st.plotly_chart(fig_tpt, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_kemiskinan = create_line_chart(df, 'Kemiskinan', 'Rata-rata Kemiskinan per Tahun', '#f39c12')
        st.plotly_chart(fig_kemiskinan, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Bar Charts per Provinsi
    st.markdown('<div class="section-header">🏆 Ranking Provinsi</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_bar_ipm = create_bar_chart(filtered_df, 'IPM', 'IPM Tertinggi', 'Blues', show_top_n)
        st.plotly_chart(fig_bar_ipm, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_bar_tpt = create_bar_chart(filtered_df, 'TPT', 'TPT Tertinggi', 'Reds', show_top_n)
        st.plotly_chart(fig_bar_tpt, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_bar_kemiskinan = create_bar_chart(filtered_df, 'Kemiskinan', 'Kemiskinan Tertinggi', 'Oranges', show_top_n)
        st.plotly_chart(fig_bar_kemiskinan, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Scatter Plots - Hubungan Antar Variabel
    st.markdown('<div class="section-header">🔍 Analisis Hubungan Antar Variabel</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_scatter1 = create_scatter_plot(filtered_df, 'IPM', 'Kemiskinan', 'Hubungan IPM vs Kemiskinan')
        st.plotly_chart(fig_scatter1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_scatter2 = create_scatter_plot(filtered_df, 'IPM', 'TPT', 'Hubungan IPM vs TPT')
        st.plotly_chart(fig_scatter2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig_scatter3 = create_scatter_plot(filtered_df, 'TPT', 'Kemiskinan', 'Hubungan TPT vs Kemiskinan')
    st.plotly_chart(fig_scatter3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    
    
    # Tabel Data Interaktif
    st.markdown('<div class="section-header">📋 Data IPM, TPT, Kemiskinan 34 Provinsi di Indonesia (2022-2024)</div>', unsafe_allow_html=True)
    
    # Opsi sorting
    col1, col2 = st.columns(2)
    with col1:
        sort_by = st.selectbox(
            "📊 Urutkan berdasarkan",
            options=['Provinsi', 'Tahun', 'IPM', 'TPT', 'Kemiskinan']
        )
    
    with col2:
        sort_order = st.selectbox(
            "📈 Urutan",
            options=['Ascending', 'Descending']
        )
    
    # Sort data
    ascending = True if sort_order == 'Ascending' else False
    display_df = filtered_df.sort_values(by=sort_by, ascending=ascending)
    
    # Format data untuk tampilan
    display_df_formatted = display_df.copy()
    
    # Format angka dengan 2 desimal (tahun sudah integer dari preprocessing)
    display_df_formatted['IPM'] = display_df_formatted['IPM'].round(2)
    display_df_formatted['TPT'] = display_df_formatted['TPT'].round(2)
    display_df_formatted['Kemiskinan'] = display_df_formatted['Kemiskinan'].round(2)
    
    # Reorder columns: Provinsi, Tahun, Kemiskinan, TPT, IPM
    display_df_formatted = display_df_formatted[['Provinsi', 'Tahun', 'Kemiskinan', 'TPT', 'IPM']]
    
    st.dataframe(
        display_df_formatted,
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = display_df_formatted.to_csv(index=False)
    st.download_button(
        label="📥 Download Data CSV",
        data=csv,
        file_name=f"data_filtered_{len(selected_years)}tahun_{len(selected_provinces) if selected_provinces else 'semua'}provinsi.csv",
        mime="text/csv"
    )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #7f8c8d; font-size: 14px;'>📊 Dashboard Eksplorasi Data IPM, TPT & Kemiskinan Indonesia (2022-2024) | Dibuat dengan Streamlit & Plotly</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()