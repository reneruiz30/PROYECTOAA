import pandas as pd
import random

# 1. Definir los perfiles de compra y sus productos típicos
perfiles = {
    "Familiar": ["Leche", "Pan molde", "Huevos", "Arroz", "Frijoles", "Pollo entero", "Papel Higiénico", "Detergente", "Queso amarillo"],
    "Parrillero_FinDeSemana": ["Carne de res", "Chorizo", "Carbón", "Cerveza", "Snacks", "Hielo", "Salsa BBQ"],
    "Estudiante_Rapido": ["Sopa instantánea", "Soda", "Galletas", "Pollo frito", "Café", "Cereal", "Comida congelada"],
    "Saludable": ["Pechuga de pollo", "Avena", "Yogur Griego", "Manzanas", "Espinaca", "Atún", "Té verde"]
}

# Productos de "compra impulsiva" que cualquiera podría llevar cerca de la caja
productos_caja = ["Goma de mascar", "Baterías", "Agua embotellada", "Chocolates"]

facturas = []
num_facturas = 800 # Generaremos 800 tickets de compra para tener un buen volumen de datos

# 2. Generar las facturas simulando clientes
for i in range(1, num_facturas + 1):
    # Elegir un perfil al azar para esta compra
    perfil_actual = random.choice(list(perfiles.keys()))
    
    # Seleccionar entre 3 y 6 productos de ese perfil
    num_productos = random.randint(3, 6)
    productos_comprados = random.sample(perfiles[perfil_actual], num_productos)
    
    # 25% de probabilidad de que el cliente compre algo impulsivo en la caja
    if random.random() < 0.25:
        productos_comprados.append(random.choice(productos_caja))
        
    # Registrar cada producto bajo el mismo ID de factura
    for prod in productos_comprados:
        facturas.append({
            "ID_Factura": f"F-{i:04d}",
            "Producto": prod
        })

# 3. Convertir a DataFrame y exportar a CSV
df_facturas = pd.DataFrame(facturas)
nombre_archivo = "datos_supermercado.csv"
df_facturas.to_csv(nombre_archivo, index=False)

print(f"¡Éxito! Se ha generado el archivo '{nombre_archivo}' con {len(df_facturas)} filas (productos comprados) agrupados en {num_facturas} facturas.")