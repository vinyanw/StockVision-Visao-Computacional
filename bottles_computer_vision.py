import cv2
from ultralytics import YOLO

# 1. Carrega o modelo YOLOv8
model = YOLO("yolov8n.pt")

# 2. Inicializa a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível abrir a webcam.")
    exit()

print("Pressione 'q' para sair do programa.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao receber o frame da câmera.")
        break

    # Executa a detecção no frame atual
    results = model(frame, conf=0.5, verbose=False)

    # VARIÁVEL DO CONTADOR: Reseta a contagem a cada novo frame
    contador_recipientes = 0

    # Processa os resultados encontrados
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Se for um dos recipientes que queremos, incrementa o contador
            if class_name in ["bottle", "cup"]:
                contador_recipientes += 1

                # Desenha a caixa delimitadora
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"{class_name.upper()} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)

    # 3. EXIBE O CONTADOR NA TELA
    # Texto que será exibido (ex: "Total de Itens: 3")
    texto_contador = f"Total de Itens: {contador_recipientes}"
    
    # Desenha uma barra ou fundo preto no topo para destacar o texto do contador
    cv2.rectangle(frame, (10, 10), (250, 50), (0, 0, 0), -1)
    
    # Escreve o valor do contador na tela (cor branca, posição x=20, y=40)
    cv2.putText(frame, texto_contador, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2)

    # Exibe o frame na tela
    cv2.imshow("Detector e Contador de Recipientes", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()