import tkinter as tk
from tkinter import ttk
from PIL import Image

def conferma_scelta():
    global selectedOption
    selectedOption = dropdown.get()
    root.destroy()

# Finestra principale
root = tk.Tk()
root.title("Demo espressioni facciali")
root.geometry("512x256")

# Etichetta per il risultato
label_result = tk.Label(root, text = "Benvenuto nella nostra demo!\nQui puoi scegliere di testare il modello per il riconoscimento di un espressione facciale specifica\no addirittura utilizzarlo per valutatre un tua stessa espressione in tempo reale.\nScegli un opzione:")

label_result.pack(pady=(20, 10))

# Opzioni del menù
options = ["Demo interattiva", "Rabbia", "Disgusto", "Paura", "Felicità", "Neutralità", "Tristezza", "Stupore"]

# Menù a tendina
dropdown = ttk.Combobox(root, values=options)
dropdown.current(0)
dropdown.pack(pady=5)

# Bottone di conferma
btn_conferma = tk.Button(root, text="Conferma", command=conferma_scelta)
btn_conferma.pack(pady=10)

# Avvia l'interfaccia
root.mainloop()



import torch
from torchvision import transforms
from torchvision.models import efficientnet_b1
model = efficientnet_b1(num_classes = 7)

model.load_state_dict(torch.load("weights/efficientnet_FER2013_finetuning-100.pth"))
model.eval()

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224), #è un ritaglio dell'imagine
    transforms.RandomHorizontalFlip(), #è un flip orizzontale con probabilità 0.5
    transforms.ToTensor(), #converte le immagini PIL in tensori di PyTorch
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



from glob import glob
from os.path import basename
classes = glob('FER-2013/train/*') #estraiamo le classi dalla cartella del dataset
classes = [basename(c) for c in classes]



#definisce la demo
def demoInterattiva():
    import cv2

    #apre la videocamera
    cap = cv2.VideoCapture(0)

    print("Premi invio per scattare una foto")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1)
        if key == 13:
            cv2.imwrite("scatto.jpg", frame)
            print("immagine salvata")
            break
    
    #chiude la telecamera e la finestra sulla quale era attiva
    cap.release()
    cv2.destroyAllWindows()
    
    img = Image.open("scatto.jpg")

    img
    
    input_tensor = test_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1).item()
        print(f"Espressione facciale: {classes[pred]}")

def test(emozione, folder = "immaginiDemo"):
    import os
    immagine = emozione + ".jpg"
    imagePath = os.path.join(folder, immagine)
    
    img = Image.open(imagePath).convert('RGB')
    img

    input_tensor = test_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1).item()
        print(f"Espressione facciale: {classes[pred]}")
    

match selectedOption:
    case "Demo interattiva":
        demoInterattiva()
    case _:
        test(selectedOption)