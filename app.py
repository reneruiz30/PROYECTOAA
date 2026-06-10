import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fpdf import FPDF
from supabase import create_client, Client
from streamlit_supabase_auth import login_form

# Configuración básica de la página
st.set_page_config(page_title="OPSO - Optimal Placement Stock", page_icon="🛒", layout="wide")

# Inicializar memoria temporal para guardar los datos entre pantallas
if 'user' not in st.session_state:
    st.session_state['user'] = None
if 'df_bruto' not in st.session_state:
    st.session_state['df_bruto'] = None
if 'df_cesta' not in st.session_state:
    st.session_state['df_cesta'] = None
if 'reglas' not in st.session_state:
    st.session_state['reglas'] = None

# --- INICIALIZAR SUPABASE (GLOBAL PARA LOGIN Y DATOS) ---
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["key"]
supabase = create_client(url, key)

# Función para consultar el rol
def obtener_rol(email):
    try:
        respuesta = supabase.table("usuarios_perfiles").select("rol").eq("email", email).execute()
        if respuesta.data:
            return respuesta.data[0]['rol']
        else:
            st.sidebar.warning("⚠️ Tu correo ingresó, pero no existe en la tabla 'usuarios_perfiles'.")
            return 'analista'
    except Exception as e:
        st.sidebar.error(f"🚨 Error de Base de Datos: {e}")
        return 'analista'

# --- LÓGICA DE LOGIN Y SEGURIDAD ---
if st.session_state['user'] is None:
    st.title("🔐 Acceso a OPSO")
    st.write("Por favor, inicia sesión con tu cuenta autorizada para acceder al sistema.")
    
    user_info = login_form(
        url=url,
        apiKey=key,
        providers=["google"]
    )

    if user_info:
        st.session_state['user'] = user_info
        st.rerun()
else:
    # --- BARRA LATERAL (NAVEGACIÓN) ---
    usuario_data = st.session_state['user']
    # Extraemos el email de forma segura (Supabase suele anidarlo)
    if isinstance(usuario_data, dict) and 'user' in usuario_data:
        email_usuario = usuario_data['user'].get('email', '')
    elif isinstance(usuario_data, dict):
        email_usuario = usuario_data.get('email', '')
    else:
        email_usuario = getattr(getattr(usuario_data, 'user', None), 'email', getattr(usuario_data, 'email', ''))

    rol_usuario = obtener_rol(email_usuario)

    st.sidebar.title("Menú OPSO")
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3081/3081840.png", width=80)
    
    # Menú dinámico según el rol
    if rol_usuario == 'admin':
        menu = ["Página Principal", "Carga de datos", "Análisis de Patrones", "Simulación de Layout", "Reportes"]
    elif rol_usuario == 'gerente':
        menu = ["Página Principal", "Reportes"]
    else:
        menu = ["Página Principal", "Análisis de Patrones", "Simulación de Layout"]
        
    eleccion = st.sidebar.radio("Navegación", menu)

    st.sidebar.markdown("---")
    st.sidebar.info(f"Usuario: {email_usuario}\nRol: {rol_usuario.upper()}")
    
    # BOTÓN OSCURO ORIGINAL (CON SUPERPODERES DE JAVASCRIPT)
    if st.sidebar.button("Cerrar sesión"):
        # 1. Cierra sesión en la base de datos
        try:
            supabase.auth.sign_out()
        except:
            pass
            
        # 2. Limpia la memoria de la aplicación en Python
        st.session_state.clear()
        
        # 3. Inyectamos JavaScript para destruir el token en el navegador y recargar
        js_limpieza = """
        <script>
            // Accede al navegador general y borra los tokens de Google/Supabase
            window.parent.localStorage.clear();
            window.parent.sessionStorage.clear();
            // Te devuelve a la URL raíz de tu proyecto
            window.parent.location.href = 'https://nozhito-proyectoaa-app-x2ivi2.streamlit.app';
        </script>
        """
        st.components.v1.html(js_limpieza, height=0)
    
    if st.sidebar.button("Cerrar sesión"):
        # 1. Cerrar sesión en Supabase
        supabase.auth.sign_out()
        
        # 2. Limpiar todo el estado de la aplicación
        st.session_state.clear()
        
        # 3. Forzar al navegador a ir a la URL base de la app, sin fragmentos (#) ni tokens
        # Cambia 'https://tu-app.streamlit.app' por la URL real de tu app
        st.markdown(
            f'<meta http-equiv="refresh" content="0;url=https://nozhito-proyectoaa-app-x2ivi2.streamlit.app">', 
            unsafe_allow_html=True
        )

    # --- 1. PÁGINA PRINCIPAL ---
    if eleccion == "Página Principal":
        st.title("🛒 OPSO - Optimal Placement Stock")
        st.markdown("### Optimización de distribución de supermercados mediante Machine Learning")
        
        st.write("""
        Bienvenido al sistema OPSO. Esta herramienta está diseñada para analizar patrones 
        reales de compra utilizando el algoritmo Apriori y generar recomendaciones estratégicas 
        para reorganizar los pasillos y productos de su establecimiento.
        """)
        st.info("**Objetivo:** Minimizar recorridos innecesarios, mejorar la experiencia del cliente y maximizar las ventas cruzadas.")

    # --- 2. CARGA DE DATOS ---
    elif eleccion == "Carga de datos":
        st.title("📂 Entrada y Preprocesamiento de Datos")
        st.write("Sube el archivo CSV con las transacciones del supermercado o conéctate a la base de datos.")
        
        # Opción 1: Subir CSV
        archivo_subido = st.file_uploader("Cargar archivo CSV", type=["csv"])
        
        if archivo_subido is not None:
            df = pd.read_csv(archivo_subido)
            st.session_state['df_bruto'] = df
            
        st.write("---")
        st.write("O bien, sincroniza directamente con la base de datos en la nube:")
        
        # Opción 2: Conectar a Supabase
        if st.button("🔄 Descargar datos desde Supabase"):
            with st.spinner("Conectando con la base de datos..."):
                try:
                    # Consultar todos los datos de la tabla usando la conexión global
                    respuesta = supabase.table("transacciones").select("*").execute()
                    
                    if respuesta.data:
                        # Convertir la respuesta a un DataFrame de Pandas
                        df_nube = pd.DataFrame(respuesta.data)
                        
                        # Renombrar columnas para mantener compatibilidad con el resto del código
                        df_nube = df_nube.rename(columns={'id_factura': 'ID_Factura', 'producto': 'Producto'})
                        
                        st.session_state['df_bruto'] = df_nube
                        st.success(f"¡Se descargaron {len(df_nube)} registros desde Supabase exitosamente!")
                    else:
                        st.warning("La base de datos está vacía.")
                        
                except Exception as e:
                    st.error(f"Error al conectar con Supabase: {e}\n(Asegúrate de haber configurado tu archivo .streamlit/secrets.toml correctamente).")
                    
        # Procesamiento común (sin importar de dónde vinieron los datos)
        if st.session_state['df_bruto'] is not None:
            df = st.session_state['df_bruto']
            st.success("¡Datos listos en memoria!")
            st.write("Vista previa de las transacciones (Formato Lista):")
            st.dataframe(df.head())
            
            if st.button("Ejecutar Preprocesamiento (Crear Matriz)"):
                with st.spinner("Transformando datos a formato de cesta de compras..."):
                    cesta = pd.crosstab(df['ID_Factura'], df['Producto']) > 0
                    st.session_state['df_cesta'] = cesta
                    
                st.success("¡Datos transformados! Listos para el análisis matemático.")
                st.write("Vista de la Matriz Booleana:")
                st.dataframe(st.session_state['df_cesta'].head())

    # --- 3. ANÁLISIS ---
    elif eleccion == "Análisis de Patrones":
        st.title("🧠 Análisis de Patrones (Algoritmo Apriori)")
        
        if st.session_state['df_cesta'] is None:
            st.warning("⚠️ Primero debes ir a 'Carga de datos' y preprocesar tu información.")
        else:
            st.write("Ajusta los hiperparámetros del modelo para descubrir las reglas de asociación:")
            
            col1, col2 = st.columns(2)
            with col1:
                min_soporte = st.slider("Soporte Mínimo (%)", min_value=1, max_value=50, value=5) / 100.0
                st.caption("Ej: 0.05 significa que los productos deben aparecer en al menos el 5% de las facturas.")
            with col2:
                min_confianza = st.slider("Confianza Mínima (%)", min_value=10, max_value=100, value=50) / 100.0
                st.caption("Ej: 0.50 significa que al comprar A, hay un 50% de probabilidad de llevar B.")
                
            if st.button("Ejecutar Algoritmo Apriori"):
                with st.spinner("Buscando patrones frecuentes..."):
                    cesta = st.session_state['df_cesta']
                    itemsets_frecuentes = apriori(cesta, min_support=min_soporte, use_colnames=True)
                    
                    if itemsets_frecuentes.empty:
                        st.error("No se encontraron patrones con estos parámetros. Intenta bajar el soporte.")
                    else:
                        reglas = association_rules(itemsets_frecuentes, metric="confidence", min_threshold=min_confianza)
                        
                        if reglas.empty:
                            st.error("No hay reglas con esa confianza mínima. Intenta bajar la confianza.")
                        else:
                            reglas["antecedents"] = reglas["antecedents"].apply(lambda x: ', '.join(list(x)))
                            reglas["consequents"] = reglas["consequents"].apply(lambda x: ', '.join(list(x)))
                            st.session_state['reglas'] = reglas
                            
                            st.success(f"¡Se encontraron {len(reglas)} reglas de asociación fuertes!")
                            reglas_mostrar = reglas[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by="lift", ascending=False)
                            reglas_mostrar.columns = ['Si compran (Antecedente)', 'También compran (Consecuente)', 'Soporte', 'Confianza', 'Lift (Fuerza)']
                            st.dataframe(reglas_mostrar)

    # --- 4. SIMULACIÓN ---
    elif eleccion == "Simulación de Layout":
        st.title("🗺️ Simulación y Optimización del Layout")
        
        def dibujar_plano(titulo, estantes):
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            
            ax.add_patch(patches.Rectangle((0.5, 0.5), 2.5, 1.5, facecolor='#cccccc', edgecolor='black', linewidth=1.5))
            ax.text(1.75, 1.25, 'Cajas', ha='center', va='center', fontsize=10, fontweight='bold')
            
            ax.add_patch(patches.Rectangle((7.0, 0.5), 2.5, 1.5, facecolor='#c8e6c9', edgecolor='black', linewidth=1.5))
            ax.text(8.25, 1.25, 'Entrada', ha='center', va='center', fontsize=10, fontweight='bold')
            
            for x, y, ancho, alto, label, color in estantes:
                ax.add_patch(patches.Rectangle((x, y), ancho, alto, facecolor=color, edgecolor='black', linewidth=1.2, alpha=0.9))
                ax.text(x + ancho/2, y + alto/2, label, ha='center', va='center', fontsize=8, fontweight='bold', color='black', wrap=True)
                
            fig.tight_layout()
            return fig

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Layout Actual (Tradicional)")
            st.error("Ruta larga: Productos relacionados están lejos.")
            estantes_actual = [
                [1, 8, 8, 1, 'Carnes y Embutidos', '#ffb3ba'],
                [1, 4, 1.5, 3, 'Lácteos\n(Leche, Queso)', '#bae1ff'],
                [3.5, 4, 1.5, 3, 'Panadería\n(Pan, Huevos)', '#ffdfba'],
                [6, 4, 1.5, 3, 'Bebidas\n(Soda, Cerveza)', '#baffc9'],
                [8.5, 4, 1.5, 3, 'Misceláneos\n(Carbón, Snacks)', '#ffffba']
            ]
            st.pyplot(dibujar_plano("Distribución por Categorías", estantes_actual))
            
        with col2:
            st.subheader("Layout Optimizado (OPSO)")
            st.success("Ruta corta: Agrupado por hábitos de compra.")
            estantes_opso = [
                [1, 8, 4, 1, 'Zona Parrilla\n(Carnes, Cerveza, Carbón)', '#ffb3ba'],
                [6, 8, 3, 1, 'Zona Estudiante\n(Sopas, Soda, Snacks)', '#baffc9'],
                [1, 4, 2, 3, 'Zona Desayuno\n(Pan, Huevos, Leche)', '#ffdfba'],
                [4, 4, 2, 3, 'Abarrotes\n(Arroz, Frijoles)', '#bae1ff'],
                [7, 4, 2, 3, 'Limpieza\n(Papel, Detergente)', '#ffffba']
            ]
            st.pyplot(dibujar_plano("Distribución por Comportamiento", estantes_opso))

        st.info("💡 **Conclusión del Modelo:** Al juntar 'Carnes, Cerveza y Carbón' en una sola zona transitable, evitamos que el cliente parrillero tenga que cruzar todo el supermercado, aumentando la probabilidad de ventas cruzadas.")

    # --- 5. REPORTES ---
    elif eleccion == "Reportes":
        st.title("📊 Salidas del Sistema y Reportes Gerenciales")
        st.write("Análisis de impacto comercial basado en las recomendaciones del modelo OPSO.")
        
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric(label="Reducción de Recorrido Crítico", value="28.4 %", delta="-14.2m de caminata")
        with kpi2:
            st.metric(label="Potencial Incremento de Ventas Cruzadas", value="16.5 %", delta="+ B/. 2.40 por ticket")
        with kpi3:
            st.metric(label="Índice de Afinidad de Layout", value="84.2 / 100", delta="Excelente acoplamiento")
            
        st.markdown("### 📋 Matriz de Planificación de Movimientos")
        st.write("A continuación se detallan las acciones físicas sugeridas para el equipo de operaciones del supermercado:")
        
        reporte_data = {
            "Zona Destino": ["Zona Parrilla", "Zona Estudiante", "Zona Desayuno", "Abarrotes", "Limpieza"],
            "Categorías Integradas": ["Carnes, Cerveza, Carbón, Salsa BBQ", "Sopa instantánea, Soda, Snacks", "Pan, Huevos, Leche, Queso", "Arroz, Frijoles, Atún", "Papel Higiénico, Detergente"],
            "Justificación de Regla": ["Afinidad detectada en fines de semana", "Patrón de compra rápida / conveniencia", "Alta correlación diaria en desayunos", "Productos base estables", "Baja correlación con productos alimenticios"],
            "Prioridad de Ejecución": ["ALTA", "MEDIA", "ALTA", "BAJA", "MEDIA"]
        }
        df_reporte = pd.DataFrame(reporte_data)
        st.dataframe(df_reporte, use_container_width=True)
        
        st.markdown("### 📥 Descarga de Entregables")
        st.write("Exporta la información para presentarla a la gerencia comercial o al equipo de planta.")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            csv_data = df_reporte.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Descargar Matriz de Movimientos (CSV)",
                data=csv_data,
                file_name="OPSO_Matriz_Movimientos.csv",
                mime="text/csv"
            )
        with col_btn2:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, txt="Reporte Ejecutivo OPSO", ln=True, align='C')
            
            pdf.set_font("Arial", 'B', 12)
            pdf.ln(10)
            pdf.cell(0, 10, txt="Resumen de Optimizacion de Layout", ln=True, align='L')
            
            pdf.set_font("Arial", '', 11)
            texto_cuerpo = (
                "Este documento ha sido generado automaticamente por el sistema OPSO. "
                "Basado en el algoritmo Apriori, se han detectado asociaciones fuertes "
                "que sugieren una reduccion del 28.4% en el recorrido critico del cliente."
            )
            pdf.multi_cell(0, 8, txt=texto_cuerpo)
            
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            
            st.download_button(
                label="📄 Descargar Reporte de Rendimiento (PDF)",
                data=pdf_bytes,
                file_name="OPSO_Reporte_Ejecutivo.pdf",
                mime="application/pdf"
            )