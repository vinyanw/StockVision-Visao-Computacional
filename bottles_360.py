# ... (Mantenha as etapas 1 e 2 de captura iguais ao código anterior) ...

# 3. Etapa de Processamento e Inferência
dimensoes = {
    "altura": 0,      # Y
    "largura": 0,     # X (visto na frente/trás)
    "profundidade": 0 # Z (visto nas laterais)
}

print("\n--- PROCESSANDO E INFERINDO VOLUME ---")

for face, caminho_img in imagens_capturadas.items():
    img = cv2.imread(caminho_img)
    results = model(img, conf=0.5, verbose=False)
    
    # Listas para guardar as coordenadas de cada garrafa detectada na face
    xs = []
    ys = []
    
    for result in results:
        for box in result.boxes:
            if model.names[int(box.cls[0])] in ["bottle", "cup"]:
                # Pega o centro da garrafa (x, y)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                xs.append((x1 + x2) / 2)
                ys.append((y1 + y2) / 2)
    
    # Se não detectou nada, pula para não quebrar o cálculo
    if not xs:
        continue
        
    # Algoritmo simples para estimar linhas e colunas por agrupamento de coordenadas
    # Agrupa elementos próximos para entender a matriz daquela face
    def contar_linhas_colunas(coordenadas, tolerancia=50):
        coordenadas.sort()
        if not coordenadas: return 0
        grupos = 1
        for i in range(1, len(coordenadas)):
            if coordenadas[i] - coordenadas[i-1] > tolerancia:
                grupos += 1
        return grupos

    # Linhas verticais representam a ALTURA (Y)
    linhas_detectadas = contar_linhas_colunas(ys)
    if linhas_detectadas > dimensoes["altura"]:
        dimensoes["altura"] = linhas_detectadas

    # Colunas representam a LARGURA (X) ou PROFUNDIDADE (Z)
    colunas_detectadas = contar_linhas_colunas(xs)
    if face in ["Frente", "Tras"]:
        if colunas_detectadas > dimensoes["largura"]:
            dimensoes["largura"] = colunas_detectadas
    else: # Direita ou Esquerda
        if colunas_detectadas > dimensoes["profundidade"]:
            dimensoes["profundidade"] = colunas_detectadas

# 4. Cálculo da Inferência 3D
largura = dimensoes["largura"]
altura = dimensoes["altura"]
profundidade = dimensoes["profundidade"]
total_inferido = largura * altura * profundidade

# 5. Resultado Final
print("\n" + "="*40)
print("       RELATÓRIO DE INFERÊNCIA 3D       ")
print("="*40)
print(f"Dimensões estimadas do fardo:")
print(f" --Largura (X): {largura} garrafas")
print(f" -- Altura  (Y): {altura} camadas")
print(f" -- Profundidade (Z): {profundidade} garrafas")
print("-"*40)
print(f"QUANTIDADE TOTAL INFERIDA: {total_inferido} garrafas")
print(f"   (Cálculo baseado em: {largura}x{altura}x{profundidade})")
print("="*40)