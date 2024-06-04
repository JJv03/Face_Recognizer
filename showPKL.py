# The following program:
# Shows the content of a .pkl file
import pickle

# Ruta al archivo .pkl
file_path = 'output/encodings.pkl'

# Abre el archivo en modo de lectura binaria y carga el contenido
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Muestra los datos deserializados
print(data)
