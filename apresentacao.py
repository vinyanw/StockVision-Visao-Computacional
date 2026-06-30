import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob

# ---------------------------------------------------------
# 1. CARREGAMENTO DINÂMICO DO SEU MODELO
# ---------------------------------------------------------
base_path = "/home/yan-gorillaz/Documents/Yan/Github/Bottles-Visão-Computacional/runs/detect/train*/weights/best.pt"
arquivos_encontrados = glob.glob(base_path)

if arquivos_encontrados:
    caminho_best = max(arquivos_encontrados, key=os.path.getmtime)
    print(f"🧠 Modelo carregado: {caminho_best}")
    model = YOLO(caminho_best)
else:
    raise FileNotFoundError("❌ Arquivo 'best.pt' não encontrado.")

# ---------------------------------------------------------
# 2. BANCO DE DADOS (Inventário)
# ---------------------------------------------------------
inventario_geral = {
    "Agua Areia Branca": {"qtd": 0, "ml_unidade": 510},
    "Refrigerante Kuat": {"qtd": 0, "ml_unidade": 250}
}

# Cores do StockVision
COR_FUNDO = (37, 16, 26)         # Roxo escuro
COR_ROXO_LIGHT = (243, 67, 94)   # Azul Violeta
COR_TEXTO = (240, 240, 245)      # Branco
COR_DESTAQUE = (170, 255, 0)     # Verde Neon

def escrever_texto(img, texto, x, y, cor, escala=0.7, espessura=2):
    """Desenha texto estável sem quebrar caracteres."""
    cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, cor, espessura, cv2.LINE_AA)

# ---------------------------------------------------------
# 3. INICIALIZAÇÃO DA CÂMERA (CORREÇÃO DE ÍNDICE)
# ---------------------------------------------------------
# Mudado para 0 (mude para 1 se for a webcam externa, mas o sistema diz que 2 ou 3 não existem)
cap = cv2.VideoCapture(0) 

cv2.namedWindow("StockVision", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("StockVision", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("\n==================================================")
print(" 🚀 STOCKVISION MVP INSTALADO")
print(" -> [ESPACO]: Escanear fardo atual")
print(" -> [Q]: Sair do programa")
print("==================================================")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # --- RENDERIZAÇÃO DA INTERFACE (HUD) ---
    # Painel Inferior (Instruções)
    cv2.rectangle(frame, (0, h - 70), (w, h), COR_FUNDO, -1)
    escrever_texto(frame, "[ ESPACO ] Escanear Lote   |   [ Q ] Sair", 30, h - 30, COR_ROXO_LIGHT, escala=0.9)

    # Painel Lateral (Resumo)
    cv2.rectangle(frame, (w - 450, 0), (w, 200), COR_FUNDO, -1)
    escrever_texto(frame, "StockVision - Camera 01", w - 430, 40, COR_ROXO_LIGHT, escala=1.0, espessura=2)
    
    y_offset = 100
    for produto, dados in inventario_geral.items():
        ml_total = dados['qtd'] * dados['ml_unidade']
        vol_print = f"{ml_total/1000:.2f} L" if ml_total >= 1000 else f"{ml_total} ml"
        
        texto = f"{produto}: {dados['qtd']} un ({vol_print})"
        escrever_texto(frame, texto, w - 430, y_offset, COR_TEXTO, escala=0.75)
        y_offset += 45

    cv2.imshow("StockVision", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # --- PROCESSO DE SCANNER ---
    if key == ord(' '):
        # Animação simplificada de varredura (Linha de Scanner)
        for step in range(0, h, int(h/12)):
            frame_scan = frame.copy()
            cv2.line(frame_scan, (0, step), (w, step), COR_DESTAQUE, 4)
            escrever_texto(frame_scan, "ESCANEANDO...", 30, 50, COR_DESTAQUE, escala=1.1, espessura=3)
            cv2.imshow("StockVision", frame_scan)
            cv2.waitKey(20)

        # Inferência Inteligente
        results = model(frame, conf=0.5, verbose=False)
        contagem_fardo = {"Agua Areia Branca": 0, "Refrigerante Kuat": 0}
        achou = False
        frame_res = frame.copy()

        for result in results:
            for box in result.boxes:
                cls_name = model.names[int(box.cls[0])].lower().strip()
                
                prod = None
                if "agua" in cls_name or "areia" in cls_name or "água" in cls_name:
                    prod = "Agua Areia Branca"
                elif "kuat" in cls_name or "refri" in cls_name:
                    prod = "Refrigerante Kuat"

                if prod:
                    contagem_fardo[prod] += 1
                    achou = True
                    
                    # Desenho do Quadrado Elegante (Mais fino)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame_res, (x1, y1), (x2, y2), COR_DESTAQUE, 2)
                    escrever_texto(frame_res, prod, x1 + 5, y1 - 10, COR_DESTAQUE, escala=0.5, espessura=1)

        if not achou:
            frame_err = frame.copy()
            escrever_texto(frame_err, "LOTE NAO RECONHECIDO", int(w/2) - 200, int(h/2), (0, 0, 255), escala=1.1, espessura=3)
            cv2.imshow("StockVision", frame_err)
            cv2.waitKey(1200)
            continue

        # Alimenta o estoque
        for prod, qtd in contagem_fardo.items():
            if qtd > 0:
                inventario_geral[prod]['qtd'] += qtd
                cv2.rectangle(frame_res, (0, 0), (w, 80), COR_FUNDO, -1)
                escrever_texto(frame_res, f"+ {qtd}x {prod} COMPUTADO!", 30, 50, COR_DESTAQUE, escala=1.0, espessura=2)

        cv2.imshow("StockVision", frame_res)
        cv2.waitKey(2000)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()