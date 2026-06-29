import ast
import datetime
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st
from fpdf import FPDF
from mlxtend.frequent_patterns import apriori, association_rules
from streamlit_supabase_auth import login_form, logout_button
from supabase import create_client

# ── Configuración de página ─────────────────────────────────────────────────
st.set_page_config(page_title="OPSO - Optimal Placement Stock", page_icon="🛒", layout="wide")


# ── CSS externo ─────────────────────────────────────────────────────────────
def load_css(path: Path) -> None:
    css = path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css(Path(__file__).parent / "style.css")


# ── Session state ───────────────────────────────────────────────────────────
for key in ("user", "df_bruto", "df_cesta", "reglas"):
    if key not in st.session_state:
        st.session_state[key] = None


# ── Supabase ────────────────────────────────────────────────────────────────
_url = st.secrets["supabase"]["url"]
_key = st.secrets["supabase"]["key"]
supabase = create_client(_url, _key)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def dibujar_plano(titulo: str, estantes: list, indices_ruta: list | None = None):
    """Genera un plano matplotlib del supermercado con ruta opcional."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    ax.add_patch(patches.Rectangle((0.5, 0.5), 2.5, 1.5, facecolor="#cccccc", edgecolor="black", linewidth=1.5))
    ax.text(1.75, 1.25, "Cajas", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.add_patch(patches.Rectangle((7.0, 0.5), 2.5, 1.5, facecolor="#c8e6c9", edgecolor="black", linewidth=1.5))
    ax.text(8.25, 1.25, "Entrada", ha="center", va="center", fontsize=10, fontweight="bold")

    centros_x, centros_y = [], []
    for i, (x, y, ancho, alto, label, color) in enumerate(estantes):
        en_ruta = indices_ruta and i in indices_ruta
        ax.add_patch(patches.Rectangle(
            (x, y), ancho, alto,
            facecolor=color,
            edgecolor="red" if en_ruta else "black",
            linewidth=3 if en_ruta else 1.2,
            alpha=0.9,
        ))
        ax.text(x + ancho / 2, y + alto / 2, label,
                ha="center", va="center", fontsize=8, fontweight="bold", color="black", wrap=True)
        centros_x.append(x + ancho / 2)
        centros_y.append(y + alto / 2)

    if indices_ruta:
        rx = [8.25] + [centros_x[i] for i in indices_ruta] + [1.75]
        ry = [1.25] + [centros_y[i] for i in indices_ruta] + [1.25]
        ax.plot(rx, ry, color="#FF4B4B", linestyle="--", linewidth=2.5, marker="o")

    fig.tight_layout()
    return fig


def calcular_kpis(reglas: pd.DataFrame) -> dict:
    lift_prom      = reglas["lift"].mean()
    conf_prom      = reglas["confidence"].mean()
    fuertes        = len(reglas[reglas["lift"] > 1.5])
    return {
        "lift_promedio":       round(lift_prom, 2),
        "indice_afinidad":     min(round((lift_prom / 3.0) * 100, 1), 100.0),
        "potencial_ventas":    round(conf_prom * 100, 1),
        "reduccion_recorrido": min(round(fuertes * 2.5, 1), 50.0),
        "ticket_extra":        round(conf_prom * 15, 2),
        "n_reglas":            len(reglas),
        "n_reglas_fuertes":    fuertes,
    }


def obtener_rol(email: str, id_usuario: str) -> str:
    if not email:
        return "analista"
    try:
        res = supabase.table("usuarios_perfiles").select("rol").eq("email", email).execute()
        if res.data:
            return res.data[0]["rol"]
        if id_usuario:
            supabase.table("usuarios_perfiles").insert({
                "id": id_usuario, "email": email, "rol": "analista"
            }).execute()
        return "analista"
    except Exception as e:
        st.sidebar.error(f"🚨 Error BD: {e}")
        return "analista"


@st.dialog("⚠️ Confirmar cierre de sesión")
def confirmar_logout():
    st.warning("¿Estás seguro de que deseas cerrar tu sesión?")
    st.write("Tendrás que volver a iniciar sesión para acceder al sistema.")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Paso 1: Desvincular credenciales")
        logout_button(url=_url, apiKey=_key)
    with col2:
        st.caption("Paso 2: Confirmar salida")
        if st.button("✅ Confirmar y salir", use_container_width=True):
            st.session_state.clear()
            st.query_params.clear()
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# LOGIN
# ════════════════════════════════════════════════════════════════════════════
if st.session_state["user"] is None:

    # Logo centrado
    st.markdown("""
    <div class="login-shell">
        <div class="login-logo">
            <div class="login-logo-icon">🛒</div>
            <h1>OPSO</h1>
            <p>Optimal Placement Stock</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        tab_login, tab_signup, tab_google = st.tabs([
            "Iniciar sesión",
            "Crear cuenta",
            "Google",
        ])

        # ── Tab 1: Iniciar sesión con email/contraseña ──────────────────
        with tab_login:
            with st.form("form_login", border=False):
                email_login = st.text_input(
                    "Correo electrónico",
                    placeholder="nombre@gmail.com",
                )
                pass_login = st.text_input(
                    "Contraseña",
                    type="password",
                    placeholder="••••••••",
                )
                if st.form_submit_button("Iniciar sesión", use_container_width=True):
                    try:
                        res = supabase.auth.sign_in_with_password({
                            "email": email_login,
                            "password": pass_login,
                        })
                        st.session_state["user"] = (
                            {"user": res.user.model_dump()} if res.user else res
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"🚨 No se pudo iniciar sesión: {e}")

        # ── Tab 2: Crear cuenta ─────────────────────────────────────────
        with tab_signup:
            with st.form("form_signup", border=False):
                nombre_signup = st.text_input(
                    "Nombre completo",
                    placeholder="Tu nombre completo",
                )
                email_signup = st.text_input(
                    "Correo electrónico",
                    placeholder="nombre@gmail.com",
                    key="su_email",
                )
                pass_signup = st.text_input(
                    "Contraseña",
                    type="password",
                    placeholder="Mínimo 8 caracteres",
                    key="su_pass",
                )
                if st.form_submit_button("Crear cuenta", use_container_width=True):
                    if len(pass_signup) < 8:
                        st.error("⚠️ La contraseña debe tener al menos 8 caracteres.")
                    else:
                        try:
                            supabase.auth.sign_up({
                                "email": email_signup,
                                "password": pass_signup,
                                "options": {"data": {"full_name": nombre_signup}},
                            })
                            st.success("✅ Cuenta creada. Revisa tu correo para confirmarla.")
                        except Exception as e:
                            st.error(f"🚨 No se pudo crear la cuenta: {e}")

        # ── Tab 3: Google OAuth (única instancia) ───────────────────────
        with tab_google:
            st.markdown("""
            <div class="google-tab-hint">
                <p>Inicia sesión o crea tu cuenta usando Google.<br>
                Serás redirigido a la página de autorización.</p>
            </div>
            """, unsafe_allow_html=True)

            gi = login_form(url=_url, apiKey=_key, providers=["google"])
            if gi:
                st.session_state["user"] = gi
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# APP PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════
else:
    usuario_data  = st.session_state["user"]
    id_usuario    = ""
    email_usuario = ""

    if isinstance(usuario_data, dict):
        inner = usuario_data.get("user", usuario_data)
        email_usuario = inner.get("email", "")
        id_usuario    = inner.get("id", "")
    else:
        user_obj      = getattr(usuario_data, "user", None)
        base          = user_obj if user_obj else usuario_data
        email_usuario = getattr(base, "email", "")
        id_usuario    = getattr(base, "id", "")

    rol_usuario = obtener_rol(email_usuario, id_usuario)

    # ── Sidebar ──────────────────────────────────────────────────────────
    st.sidebar.title("Menú OPSO")
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3081/3081840.png", width=80)

    if rol_usuario == "admin":
        menu = ["Página Principal", "Carga de datos", "Análisis de Patrones",
                "Simulación de Layout", "Reportes", "Panel Gerencial", "Gestión de Usuarios"]
    elif rol_usuario == "gerente":
        menu = ["Página Principal", "Reportes", "Panel Gerencial"]
    else:
        menu = ["Página Principal", "Análisis de Patrones", "Simulación de Layout"]

    eleccion = st.sidebar.radio("Navegación", menu)
    st.sidebar.markdown("---")
    st.sidebar.info(f"Usuario: {email_usuario}\nRol: {rol_usuario.upper()}")

    if st.sidebar.button("🚪 Cerrar sesión", use_container_width=True):
        confirmar_logout()

    # ════════════════════════════════════════════════════════════════════
    # 1. PÁGINA PRINCIPAL
    # ════════════════════════════════════════════════════════════════════
    if eleccion == "Página Principal":
        st.title("🛒 OPSO - Optimal Placement Stock")
        st.markdown("### Optimización de distribución de supermercados mediante Machine Learning")
        st.write("""
        Bienvenido al sistema OPSO. Esta herramienta analiza patrones reales de compra
        utilizando el algoritmo Apriori y genera recomendaciones estratégicas para
        reorganizar los pasillos y productos de su establecimiento.
        """)
        st.info("**Objetivo:** Minimizar recorridos innecesarios, mejorar la experiencia del cliente y maximizar las ventas cruzadas.")

    # ════════════════════════════════════════════════════════════════════
    # 2. CARGA DE DATOS
    # ════════════════════════════════════════════════════════════════════
    elif eleccion == "Carga de datos":
        st.title("📂 Entrada y Preprocesamiento de Datos")
        st.write("Sube el archivo CSV con las transacciones del supermercado o conéctate a la base de datos.")

        archivo_subido = st.file_uploader("Cargar archivo CSV", type=["csv"])
        if archivo_subido is not None:
            df = pd.read_csv(archivo_subido)
            cols_req = {"ID_Factura", "Producto"}
            if not cols_req.issubset(df.columns):
                faltantes = cols_req - set(df.columns)
                st.error(
                    f"❌ Columnas requeridas faltantes: **{', '.join(faltantes)}**.\n\n"
                    f"Columnas encontradas: {', '.join(df.columns.tolist())}."
                )
            else:
                st.session_state["df_bruto"] = df

        st.write("---")
        st.write("O bien, sincroniza directamente con la base de datos en la nube:")

        if st.button("🔄 Descargar datos desde Supabase"):
            with st.spinner("Conectando con la base de datos..."):
                try:
                    res = supabase.table("transacciones").select("*").execute()
                    if res.data:
                        df_nube = pd.DataFrame(res.data).rename(
                            columns={"id_factura": "ID_Factura", "producto": "Producto"})
                        st.session_state["df_bruto"] = df_nube
                        st.success(f"¡Se descargaron {len(df_nube)} registros desde Supabase!")
                    else:
                        st.warning("La base de datos está vacía.")
                except Exception as e:
                    st.error(f"Error al conectar con Supabase: {e}")

        if st.session_state["df_bruto"] is not None:
            df = st.session_state["df_bruto"]
            st.success("¡Datos listos en memoria!")

            st.markdown("### 📊 Top 10 Productos Más Vendidos")
            st.bar_chart(df["Producto"].value_counts().head(10), color="#E8673A")

            st.write("Vista previa de las transacciones:")
            st.dataframe(df.head())

            if st.button("Ejecutar Preprocesamiento (Crear Matriz)"):
                with st.spinner("Transformando datos..."):
                    cesta = pd.crosstab(df["ID_Factura"], df["Producto"]) > 0
                    st.session_state["df_cesta"] = cesta
                st.success("¡Datos transformados! Listos para el análisis matemático.")
                st.dataframe(st.session_state["df_cesta"].head())

    # ════════════════════════════════════════════════════════════════════
    # 3. ANÁLISIS DE PATRONES
    # ════════════════════════════════════════════════════════════════════
    elif eleccion == "Análisis de Patrones":
        st.title("🧠 Análisis de Patrones (Algoritmo Apriori)")

        if st.session_state["df_cesta"] is None:
            st.warning("⚠️ Primero ve a 'Carga de datos' y preprocesa tu información.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                min_soporte = st.slider("Soporte Mínimo (%)", 1, 50, 5) / 100.0
                st.caption("Ej: 0.05 → productos en al menos el 5% de las facturas.")
            with col2:
                min_confianza = st.slider("Confianza Mínima (%)", 10, 100, 50) / 100.0
                st.caption("Ej: 0.50 → al comprar A, 50% de probabilidad de llevar B.")

            if st.button("Ejecutar Algoritmo Apriori"):
                with st.spinner("Buscando patrones frecuentes..."):
                    itemsets = apriori(st.session_state["df_cesta"], min_support=min_soporte, use_colnames=True)
                    if itemsets.empty:
                        st.error("Sin patrones con estos parámetros. Baja el soporte.")
                    else:
                        reglas = association_rules(itemsets, metric="confidence", min_threshold=min_confianza)
                        if reglas.empty:
                            st.error("Sin reglas con esa confianza. Baja la confianza.")
                        else:
                            reglas["antecedents"] = reglas["antecedents"].apply(lambda x: ", ".join(list(x)))
                            reglas["consequents"] = reglas["consequents"].apply(lambda x: ", ".join(list(x)))
                            st.session_state["reglas"] = reglas
                            st.success(f"¡Se encontraron {len(reglas)} reglas de asociación fuertes!")

            if st.session_state["reglas"] is not None:
                rules = st.session_state["reglas"]

                st.markdown("---")
                st.markdown("### 🔍 Buscador de Asociaciones")
                filtro = st.text_input("Filtrar por producto (ej. Cerveza, Pan, Leche):", "")
                if filtro:
                    mask = (rules["antecedents"].str.contains(filtro, case=False, na=False) |
                            rules["consequents"].str.contains(filtro, case=False, na=False))
                    reglas_vis = rules[mask]
                else:
                    reglas_vis = rules

                tabla = reglas_vis[["antecedents", "consequents", "support", "confidence", "lift"]]\
                    .sort_values("lift", ascending=False).copy()
                tabla.columns = ["Si compran (Antecedente)", "También compran (Consecuente)",
                                  "Soporte", "Confianza", "Lift (Fuerza)"]

                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.markdown("###  Soporte vs Confianza")
                    st.info("El tamaño de la burbuja representa el Lift.")
                    if not tabla.empty:
                        st.scatter_chart(data=reglas_vis, x="support", y="confidence", size="lift", color="#E8673A")
                    else:
                        st.warning("No hay datos para graficar con ese filtro.")

                with col_g2:
                    st.markdown("###  Red de Compras Frecuentes")
                    st.info("Conexiones entre productos asociados.")
                    if not tabla.empty:
                        G = nx.DiGraph()
                        for _, row in tabla.head(20).iterrows():
                            G.add_edge(row["Si compran (Antecedente)"],
                                       row["También compran (Consecuente)"],
                                       weight=row["Lift (Fuerza)"])
                        fig_net, ax_net = plt.subplots(figsize=(6, 6))
                        fig_net.patch.set_facecolor("#1e1e2e")
                        ax_net.set_facecolor("#1e1e2e")
                        nx.draw(G, nx.spring_layout(G, k=0.8, seed=42),
                                with_labels=True, node_color="#E8673A",
                                node_size=2000, font_size=9, font_weight="bold",
                                font_color="#ffffff", edge_color="#7c7c9e",
                                ax=ax_net, arrows=True, arrowsize=15, alpha=0.9)
                        st.pyplot(fig_net)
                    else:
                        st.warning("No hay conexiones para graficar.")

                st.markdown("### 📋 Resultados Detallados")
                st.dataframe(tabla, use_container_width=True)
                st.download_button(
                    label="📥 Descargar Reglas (CSV)",
                    data=tabla.to_csv(index=False).encode("utf-8"),
                    file_name="OPSO_Reglas_Apriori.csv",
                    mime="text/csv",
                )

    # ════════════════════════════════════════════════════════════════════
    # 4. SIMULACIÓN DE LAYOUT
    # ════════════════════════════════════════════════════════════════════
    elif eleccion == "Simulación de Layout":
        st.title(" Simulación y Optimización del Layout")

        COORDS = [
            [1, 8, 4, 1, "#ffb3ba"], [6, 8, 3, 1, "#baffc9"],
            [1, 4, 2, 3, "#ffdfba"], [4, 4, 2, 3, "#bae1ff"],
            [7, 4, 2, 3, "#ffffba"],
        ]

        # Layout tradicional
        if st.session_state["df_bruto"] is not None:
            prods = st.session_state["df_bruto"]["Producto"].value_counts().index.tolist()
            estantes_actual = [
                [c[0], c[1], c[2], c[3], f"Pasillo:\n{prods[i]}" if i < len(prods) else f"Vacío {i+1}", "#cccccc"]
                for i, c in enumerate(COORDS)
            ]
        else:
            estantes_actual = [
                [1, 8, 8, 1, "Carnes y Embutidos", "#cccccc"],
                [1, 4, 1.5, 3, "Lácteos\n(Leche, Queso)", "#cccccc"],
                [3.5, 4, 1.5, 3, "Panadería\n(Pan, Huevos)", "#cccccc"],
                [6, 4, 1.5, 3, "Bebidas\n(Soda, Cerveza)", "#cccccc"],
                [8.5, 4, 1.5, 3, "Misceláneos", "#cccccc"],
            ]

        # Layout optimizado dinámico
        estantes_opso, mapeados, slot = [], set(), 0
        if st.session_state["reglas"] is not None and not st.session_state["reglas"].empty:
            col_lift = "lift" if "lift" in st.session_state["reglas"].columns else "Lift (Fuerza)"
            for _, row in st.session_state["reglas"].sort_values(col_lift, ascending=False).iterrows():
                if slot >= 5: break
                par = frozenset([row["antecedents"], row["consequents"]])
                if par in mapeados: continue
                mapeados.add(par)
                c = COORDS[slot]
                estantes_opso.append([c[0], c[1], c[2], c[3],
                    f"Zona Opt.:\n({row['antecedents']} + {row['consequents']})", c[4]])
                slot += 1

        if slot < 5 and st.session_state["df_bruto"] is not None:
            for prod in st.session_state["df_bruto"]["Producto"].value_counts().index:
                if slot >= 5: break
                if not any(prod in e[4] for e in estantes_opso):
                    c = COORDS[slot]
                    estantes_opso.append([c[0], c[1], c[2], c[3], f"Pasillo:\n{prod}", c[4]])
                    slot += 1

        if not estantes_opso:
            estantes_opso = [
                [1, 8, 4, 1, "Zona Parrilla\n(Carnes, Cerveza, Carbón)", "#ffb3ba"],
                [6, 8, 3, 1, "Zona Estudiante\n(Sopas, Soda, Snacks)", "#baffc9"],
                [1, 4, 2, 3, "Zona Desayuno\n(Pan, Huevos, Leche)", "#ffdfba"],
                [4, 4, 2, 3, "Abarrotes\n(Arroz, Frijoles)", "#bae1ff"],
                [7, 4, 2, 3, "Limpieza\n(Papel, Detergente)", "#ffffba"],
            ]

        st.markdown("### 🚶 Simulador de Recorrido de Compras")
        opciones = [e[4].replace("\n", " ") for e in estantes_opso]
        seleccion = st.multiselect("Seleccione los pasillos que visitará el cliente:", opciones)
        indices_ruta = [opciones.index(z) for z in seleccion]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Layout Inicial (Desorganizado)")
            st.pyplot(dibujar_plano("Tradicional", estantes_actual, indices_ruta))
        with col2:
            st.subheader("Layout Optimizado (OPSO)")
            st.pyplot(dibujar_plano("OPSO", estantes_opso, indices_ruta))

        st.markdown("---")
        if st.button("💾 Enviar Propuesta al Panel Gerencial"):
            try:
                supabase.table("layouts_historial").insert({
                    "nombre_layout": f"Propuesta OPSO - {datetime.date.today()}",
                    "asociaciones": str([e[4].replace("\n", " ") for e in estantes_opso]),
                    "creado_por": email_usuario,
                    "estado": "Pendiente",
                }).execute()
                st.success("¡Propuesta enviada! El Gerente puede revisarla en su panel.")
            except Exception as e:
                st.error(f"Error al guardar propuesta: {e}")

    # ════════════════════════════════════════════════════════════════════
    # 5. REPORTES
    # ════════════════════════════════════════════════════════════════════
    elif eleccion == "Reportes":
        st.title(" Salidas del Sistema y Reportes Gerenciales")

        reglas = st.session_state.get("reglas")

        if reglas is not None and not reglas.empty:
            kpis = calcular_kpis(reglas)
            st.write(f"Métricas basadas en **{kpis['n_reglas']} reglas** encontradas por OPSO.")
            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Reducción Recorrido (est.)", f"{kpis['reduccion_recorrido']} %",
                          delta=f"{kpis['n_reglas_fuertes']} reglas lift > 1.5")
            with k2:
                st.metric("Potencial Ventas Cruzadas", f"{kpis['potencial_ventas']} %",
                          delta=f"+ B/. {kpis['ticket_extra']:.2f} por ticket")
            with k3:
                st.metric("Índice Afinidad Layout", f"{kpis['indice_afinidad']} / 100",
                          delta=f"Lift prom: {kpis['lift_promedio']}")
        else:
            st.warning("⚠️ Ejecuta el algoritmo Apriori en 'Análisis de Patrones' para ver KPIs reales.")
            k1, k2, k3 = st.columns(3)
            with k1: st.metric("Reducción Recorrido", "— %", delta="Sin datos aún")
            with k2: st.metric("Ventas Cruzadas", "— %", delta="Sin datos aún")
            with k3: st.metric("Índice Afinidad", "— / 100", delta="Sin datos aún")

        # ROI Calculator
        st.markdown("### 💰 Calculadora de ROI")
        st.info("Estima la ganancia económica al aplicar las reglas de OPSO.")
        col_r1, col_r2 = st.columns(2)
        ticket_prom   = col_r1.number_input("Ticket Promedio (B/.)", value=25.0, step=1.0)
        facturas_mes  = col_r2.number_input("Facturas mensuales", value=1500, step=100)

        if reglas is not None and not reglas.empty:
            col_lift = "lift" if "lift" in reglas.columns else "Lift (Fuerza)"
            lift_avg = reglas[col_lift].mean()
            mejora   = max((lift_avg - 1) * 0.05, 0.02)
            proyeccion = ticket_prom * facturas_mes * mejora
            st.success(
                f"**Proyección:** Con Lift promedio {lift_avg:.2f}, estimamos un incremento de "
                f"**B/. {proyeccion:,.2f} mensuales**."
            )
        else:
            st.warning("Ejecuta Apriori para calcular la proyección financiera.")

        # Matriz de movimientos
        st.markdown("###  Matriz de Planificación de Movimientos")
        df_reporte = pd.DataFrame({
            "Zona Destino": ["Zona Parrilla", "Zona Estudiante", "Zona Desayuno", "Abarrotes", "Limpieza"],
            "Categorías Integradas": [
                "Carnes, Cerveza, Carbón, Salsa BBQ", "Sopa instantánea, Soda, Snacks",
                "Pan, Huevos, Leche, Queso", "Arroz, Frijoles, Atún", "Papel Higiénico, Detergente",
            ],
            "Justificación de Regla": [
                "Afinidad en fines de semana", "Compra rápida / conveniencia",
                "Alta correlación diaria", "Productos base estables", "Baja correlación con alimentos",
            ],
            "Prioridad": ["ALTA", "MEDIA", "ALTA", "BAJA", "MEDIA"],
        })
        st.dataframe(df_reporte, use_container_width=True)

        # Descargas
        st.markdown("### 📥 Descarga de Entregables")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.download_button(
                "💾 Descargar Matriz (CSV)",
                data=df_reporte.to_csv(index=False).encode("utf-8"),
                file_name="OPSO_Matriz_Movimientos.csv",
                mime="text/csv",
            )
        with col_b2:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Reporte Ejecutivo OPSO", ln=True, align="C")
            pdf.set_font("Arial", "B", 12)
            pdf.ln(10)
            pdf.cell(0, 10, "Resumen de Optimizacion de Layout", ln=True, align="L")
            pdf.set_font("Arial", "", 11)

            if reglas is not None and not reglas.empty:
                kpis = calcular_kpis(reglas)
                cuerpo = (
                    f"Generado por OPSO. Se detectaron {kpis['n_reglas']} reglas de asociacion "
                    f"con lift promedio {kpis['lift_promedio']}. Reduccion estimada de recorrido: "
                    f"{kpis['reduccion_recorrido']}%. Potencial ventas cruzadas: {kpis['potencial_ventas']}%."
                )
            else:
                cuerpo = ("Generado por OPSO. No se detectaron reglas al momento de la exportacion. "
                          "Ejecute el algoritmo Apriori para obtener metricas reales.")

            pdf.multi_cell(0, 8, cuerpo)
            pdf_bytes = pdf.output(dest="S")
            if isinstance(pdf_bytes, str):
                pdf_bytes = pdf_bytes.encode("latin-1")

            st.download_button(
                "📄 Descargar Reporte (PDF)",
                data=pdf_bytes,
                file_name="OPSO_Reporte_Ejecutivo.pdf",
                mime="application/pdf",
            )

    # ════════════════════════════════════════════════════════════════════
    # 6. PANEL GERENCIAL
    # ════════════════════════════════════════════════════════════════════
    elif eleccion == "Panel Gerencial":
        st.title(" Panel de Decisiones Gerenciales")
        st.write("Historial de propuestas de Layout para aprobación o rechazo.")

        try:
            res = supabase.table("layouts_historial").select("*").order("creado_en", desc=True).execute()
            if res.data:
                df_hist = pd.DataFrame(res.data)
                st.dataframe(df_hist[["creado_en", "nombre_layout", "creado_por", "estado"]],
                             use_container_width=True)

                st.markdown("### Auditar Propuesta")
                id_sel = st.selectbox("Seleccione el ID a evaluar:", df_hist["id"].tolist())
                fila   = df_hist[df_hist["id"] == id_sel].iloc[0]

                st.markdown("#### 🗂️ Distribución propuesta")
                try:
                    zonas = ast.literal_eval(fila["asociaciones"])
                    for i, zona in enumerate(zonas):
                        st.markdown(
                            f'<div class="estante-card"><b>Estante {i+1}:</b> {zona}</div>',
                            unsafe_allow_html=True,
                        )
                except Exception:
                    st.write(fila["asociaciones"])

                st.markdown("---")
                col_ap, col_re = st.columns(2)
                with col_ap:
                    if st.button("✅ Aprobar Layout"):
                        supabase.table("layouts_historial").update({"estado": "Aprobado"}).eq("id", id_sel).execute()
                        st.success("Layout aprobado y registrado.")
                        st.rerun()
                with col_re:
                    if st.button("❌ Rechazar Propuesta"):
                        supabase.table("layouts_historial").update({"estado": "Rechazado"}).eq("id", id_sel).execute()
                        st.error("Propuesta rechazada.")
                        st.rerun()
            else:
                st.info("No hay propuestas de layout pendientes.")
        except Exception as e:
            st.error(f"Error cargando el historial: {e}. Verifica que la tabla 'layouts_historial' existe.")

    # ════════════════════════════════════════════════════════════════════
    # 7. GESTIÓN DE USUARIOS
    # ════════════════════════════════════════════════════════════════════
    elif eleccion == "Gestión de Usuarios":
        st.title("👥 Panel de Gestión de Usuarios")
        st.write("Audita las cuentas registradas y reasigna roles en tiempo real.")

        try:
            res = supabase.table("usuarios_perfiles").select("*").execute()
            if res.data:
                df_u = pd.DataFrame(res.data)

                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1: st.metric("Total Registrados", len(df_u))
                with col_m2: st.metric("Administradores", len(df_u[df_u["rol"] == "admin"]))
                with col_m3: st.metric("Analistas y Gerentes", len(df_u[df_u["rol"] != "admin"]))

                st.markdown("### Directorio de Cuentas")
                st.dataframe(df_u[["email", "rol"]], use_container_width=True)

                st.markdown("---")
                st.markdown("### ⚙️ Modificar Permisos")
                correo_sel = st.selectbox("Seleccione el correo a modificar:", df_u["email"].tolist())
                rol_actual = df_u[df_u["email"] == correo_sel]["rol"].values[0]
                roles      = ["admin", "gerente", "analista"]
                nuevo_rol  = st.selectbox("Nuevo rol:", roles, index=roles.index(rol_actual) if rol_actual in roles else 2)

                if st.button("Actualizar Privilegios"):
                    supabase.table("usuarios_perfiles").update({"rol": nuevo_rol}).eq("email", correo_sel).execute()
                    st.success(f"✅ {correo_sel} → {nuevo_rol.upper()}")
                    st.rerun()
            else:
                st.warning("La tabla 'usuarios_perfiles' no retornó registros.")
        except Exception as e:
            st.error(f"🚨 Error cargando panel de usuarios: {e}")