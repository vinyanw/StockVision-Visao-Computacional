import cv2
import numpy as np
from ultralytics import YOLO

# 1. CARREGA O SEU MODELO TREINADO
# Se o arquivo estiver em outra pasta, coloque o caminho completo até o best.pt
model = YOLO("/home/yan-gorillaz/Documents/Yan/Github/Bottles-Visão-Computacional/Distribuidoras-Bottles.yolov8/runs/detect/train-2/weights/best.pt")

# 2. Dicionário do Inventário Geral (Acumulador)
inventario_geral = {
    "Agua Mineral": 0,
    "Refrigerante": 0,
    "Cerveja": 0
}

# Altere para 1 ou 2 se precisar mudar para a webcam externa
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
        
        # Executa a detecção usando o seu modelo
        results = model(frame, conf=0.5, verbose=False)
        
        contagem_fardo_atual = 0

        # Cria uma cópia do frame para desenhar os resultados e mostrar na tela temporariamente
        frame_resultado = frame.copy()

        for result in results:
            for box in result.boxes:
                # Como você treinou o modelo apenas para as suas tampas, 
                # qualquer detecção válida aqui será computada
                contagem_fardo_atual += 1
                
                # Desenha os quadradinhos verdes nas tampas que o modelo encontrar
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame_resultado, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_resultado, "Tampa", (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        if contagem_fardo_atual == 0:
            print("⚠️ Nenhuma tampa detectada pelo modelo. Verifique o enquadramento.")
            continue

        # COMO VOCÊ SÓ TEM UM TIPO DE PRODUTO TREINADO POR ENQUANTO:
        # Vamos assumir que tudo o que ele encontrar agora será "Agua Mineral"
        tipo_detectado = "Agua Mineral"

        # Atualiza o inventário
        inventario_geral[tipo_detectado] += contagem_fardo_atual
        
        print(f"✓ SUCESSO: Detectadas {contagem_fardo_atual} tampas de {tipo_detectado}!")
        
        # Mostra o frame com os quadrados verdes por 2 segundos para você conferir a detecção
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