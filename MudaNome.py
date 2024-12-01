import os
import shutil

# Caminho da pasta com os arquivos
caminho_pasta = "archive/train/eye"

# Lista os arquivos na pasta
arquivos = [f for f in os.listdir(caminho_pasta) if os.path.isfile(os.path.join(caminho_pasta, f))]

# Renomeia os arquivos
for indice, nome_antigo in enumerate(arquivos):
    # Define o caminho completo do arquivo antigo
    caminho_antigo = os.path.join(caminho_pasta, nome_antigo)
    
    # Obtem a extensão do arquivo
    extensao = os.path.splitext(nome_antigo)[1]
    
    # Define o novo nome e caminho completo
    novo_nome = f"imagem_iris[{indice}]{extensao}"
    caminho_novo = os.path.join(caminho_pasta, novo_nome)
    
    try:
        # Usa shutil para mover (renomear) o arquivo
        shutil.move(caminho_antigo, caminho_novo)
        print(f"Renomeado: {nome_antigo} -> {novo_nome}")
    except Exception as e:
        print(f"Erro ao renomear {nome_antigo}: {e}")

print("Processo concluído!")
