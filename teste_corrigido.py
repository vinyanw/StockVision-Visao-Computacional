import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob

# 1. BUSCA DINÂMICA DO MODELO MAIS RECENTE
# Corrigido o caminho: removido o 'Distribuidoras-Bottles.yolov8' do meio, pois o runs fica na raiz do repositório
base_path = "/home/yan-gorillaz/Documents/Yan/Github/Bottles-Visão-Computacional/runs/detect/train*/weights/best.pt"
arquivos_encontrados = glob.glob(base_path)

if arquivos_encontrados:
    # Pega o arquivo modificado mais recentemente entre todos os treinos (train, train2, train3...)
    caminho_best = max(arquivos_encontrados, key=os.path.getmtime)
    print(f"🧠 Carregando o modelo mais recente encontrado em: {caminho_best}")
    model = YOLO(caminho_best)
else:
    raise FileNotFoundError("❌ Nenhum arquivo 'best.pt' foi encontrado nas pastas runs/detect/train*")

# [CORREÇÃO]: A linha antiga fixa que forçava carregar o 'train-2' foi REMOVIDA daqui para não anular a busca inteligente acima.

# 2. Dicionário do Inventário Geral (Acumulador)
# Padronizado com os mesmos nomes usados na lógica do CHECK
inventario_geral = {
    "Agua Areia Branca": 0,
    "Reformatado Kuat": 0
}

cap = cv2.VideoCapture(2)

print("="*50)
print("SISTEMA DE INVENTARIO - MODELO CUSTOMIZADO")
print("-> Pressione [ESPACO] para dar o CHECK no fardo.")
print("-> Pressione [Q] para fechar o programa.")
print("="*50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Interface visual em tempo real (Painel inferior e superior)
    cv2.rectangle(frame, (10, h - 80), (w - 10, h - 10), (0, 0, 0), -1)
    cv2.putText(frame, "[ESPACO] para dar CHECK no Fardo | [Q] para Sair", 
                (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    y_painel = 30
    cv2.rectangle(frame, (w - 250, 10), (w - 10, 110), (50, 50, 50), -1)
    cv2.putText(frame, "INVENTARIO ATUAL:", (w - 240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    for produto, qtd in inventario_geral.items():
        y_painel += 20
        cv2.putText(frame, f"{produto}: {qtd}", (w - 240, y_painel), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Estacao de Contagem - Visao Superior", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # ---- LÓGICA DO CHECK ----
    if key == ord(' '):
        print("\n📸 [CHECK] Analisando fardo atual com IA customizada...")
        
        results = model(frame, conf=0.5, verbose=False)
        
        # Dicionário temporário para contar o que está NESTE fardo específico
        contagem_fardo = {
            "Agua Areia Branca": 0,
            "Reformatado Kuat": 0
        }
        
        frame_resultado = frame.copy()
        detectou_algo = False

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]  # Pega o nome da classe do seu modelo

                # Mapeia o nome do modelo para o nome do inventário
                produto_identificado = None
                if "agua" in class_name.lower() or "areia" in class_name.lower():
                    produto_identificado = "Agua Areia Branca"
                elif "kuat" in class_name.lower() or "refri" in class_name.lower():
                    produto_identificado = "Reformatado Kuat"

                # Se for um dos produtos desejados, contabiliza e desenha na tela
                if produto_identificado:
                    contagem_fardo[produto_identificado] += 1
                    detectou_algo = True
                    
                    # Desenha o quadrado na tela
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame_resultado, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_resultado, produto_identificado, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        if not detectou_algo:
            print("⚠️ Nenhuma tampa de Água ou Kuat detectada pelo modelo.")
            continue

        # Soma as quantidades deste fardo para o Inventário Geral definitivo
        for produto, qtd in contagem_fardo.items():
            if qtd > 0:
                inventario_geral[produto] += qtd
                print(f"✓ SUCESSO: Detectadas {qtd} unidades de {produto}!")
        
        # Mostra o resultado por 2 segundos
        cv2.imshow("Estacao de Contagem - Visao Superior", frame_resultado)
        cv2.waitKey(2000)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Relatório final no terminal
print("\n" + "="*40)
print("       FECHAMENTO DE INVENTARIO FINAL     ")
print("="*40)
for produto, qtd in inventario_geral.items():
    print(f" 📦 {produto:15}: {qtd} unidades")
print("="*40)