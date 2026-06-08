import random
import toml
from supabase import create_client, Client

print("Iniciando generador de datos sintéticos...")

# 1. Leer las credenciales desde tu archivo secrets.toml
try:
    with open(".streamlit/secrets.toml", "r") as f:
        secrets = toml.load(f)
    url = secrets["supabase"]["url"]
    key = secrets["supabase"]["key"]
except FileNotFoundError:
    print("❌ Error: No se encontró el archivo .streamlit/secrets.toml")
    exit()

# Inicializar cliente de Supabase
supabase: Client = create_client(url, key)

# 2. Definir los perfiles de compra y sus productos típicos
perfiles = {
    "Familiar": ["Leche", "Pan molde", "Huevos", "Arroz", "Frijoles", "Pollo entero", "Papel Higiénico", "Detergente", "Queso amarillo"],
    "Parrillero_FinDeSemana": ["Carne de res", "Chorizo", "Carbón", "Cerveza", "Snacks", "Hielo", "Salsa BBQ"],
    "Estudiante_Rapido": ["Sopa instantánea", "Soda", "Galletas", "Pollo frito", "Café", "Cereal", "Comida congelada"],
    "Saludable": ["Pechuga de pollo", "Avena", "Yogur Griego", "Manzanas", "Espinaca", "Atún", "Té verde"]
}

productos_caja = ["Goma de mascar", "Baterías", "Agua embotellada", "Chocolates"]

registros = []
num_facturas = 800

print(f"Generando {num_facturas} facturas en memoria...")

# 3. Generar las facturas simulando clientes
for i in range(1, num_facturas + 1):
    perfil_actual = random.choice(list(perfiles.keys()))
    num_productos = random.randint(3, 6)
    productos_comprados = random.sample(perfiles[perfil_actual], num_productos)
    
    if random.random() < 0.25:
        productos_comprados.append(random.choice(productos_caja))
        
    for prod in productos_comprados:
        registros.append({
            "id_factura": f"F-{i:04d}",
            "producto": prod
            # Nota: No enviamos la 'fecha' porque Supabase la genera automáticamente con DEFAULT NOW()
        })

print(f"Total de productos individuales a insertar: {len(registros)}")
print("Conectando con Supabase e insertando datos (esto puede tomar unos segundos)...")

# 4. Insertar en Supabase en bloques (batches) de 500 para evitar saturar la red
tamanio_lote = 500
for i in range(0, len(registros), tamanio_lote):
    lote = registros[i:i + tamanio_lote]
    try:
        # Ejecutar el insert en la tabla 'transacciones'
        supabase.table("transacciones").insert(lote).execute()
        print(f"✅ Lote insertado exitosamente: {i} al {i + len(lote)}")
    except Exception as e:
        print(f"❌ Error al insertar el lote: {e}")

print("🎉 ¡Proceso finalizado! Tu base de datos en Supabase ya tiene información.")